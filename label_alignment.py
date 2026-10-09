"""
label_alignment.py
===================
Aligns the ground-truth labels mixer.py writes into metadata.json with the
frame grid feature_extraction.py produces, and reshapes them into a fixed
multi-track tensor suitable for training (e.g. PIT-based multi-ACCDOA).
 
This module fixes three concrete mismatches between features and labels:
 
  1. TIME-GRID mismatch
     Audio features are on the STFT frame grid (frame spacing =
     hop_length / fs, e.g. ~10.7 ms for hop_length=512, fs=48000).
     Labels (`trajectory_samples`) are on the `metadata_dt_s` grid
     (default 50 ms / 20 Hz, see scenario.py/mixer.py), and that grid's
     phase is anchored to each event's own onset -- not to the chunk's
     start -- so it is not even aligned across different simultaneous
     events, let alone with the audio frame grid.
     Fix: labels are treated as (sparse) samples of a continuous,
     physically smooth trajectory and are re-sampled onto the exact audio
     frame times via per-axis linear interpolation (drones move at most a
     few cm per audio hop, so linear interpolation on this dense a grid
     introduces negligible error). TDOA is NOT interpolated directly;
     it is recomputed from the interpolated position with
     `propagation.true_tdoa_seconds` (the exact same function mixer.py
     itself uses to generate the original labels), so the TDOA label
     stays geometrically exact and internally consistent with the
     position label at every frame, rather than being two independently
     interpolated (and therefore possibly slightly inconsistent)
     quantities.
 
  2. ABSOLUTE vs. CHUNK-RELATIVE time origin
     `feature_extraction.py`'s `frame_times_s` is relative to the start
     of the chunk. `onset_s` / `offset_s` / `trajectory_samples[*]["t"]`
     in metadata.json are in absolute EPISODE time (see mixer.py). This
     module converts frame_times_s -> absolute time internally
     (`metadata["episode_time_range_s"][0] + frame_times_s`) exactly
     once, in one place, so the caller never has to remember to do it.
 
  3. VARIABLE POLYPHONY (multiple simultaneous sources)
     A chunk can have 0..N active drones/interferers at once, and this
     varies frame to frame as events start/end (see scenario.py /
     mixer.py's `_events_active_in_chunk`). A single "the" TDOA value
     per mic-pair per frame cannot represent that. This module outputs
     a fixed-size, PADDED multi-track tensor per frame:
     (n_frames, max_tracks, ...) + an (n_frames, max_tracks) activity
     mask, one track slot per simultaneously-active event, so downstream
     training code (e.g. Hungarian/PIT-matched multi-ACCDOA loss) has a
     well-defined target shape regardless of how many sources are active
     in a given frame.
     Honesty note: this fixes the *label representation* mismatch. It
     does NOT (and physically cannot) split the single mixed GCC-PHAT
     correlation function computed in feature_extraction.py back into
     one clean correlation per source when sources overlap in time --
     that separation is exactly what the downstream model has to learn.
 
Typical usage
-------------
    from feature_extraction import extract_features_from_chunk, FeatureConfig
    from label_alignment import align_labels_to_frames
 
    feats = extract_features_from_chunk(chunk_dir, FeatureConfig())
    labels = align_labels_to_frames(
        feats["metadata"], feats["frame_times_s"], feats["mic_names"], max_tracks=3
    )
    # labels["drone"]["position_xyz_m"]   : (n_frames, max_tracks, 3)
    # labels["drone"]["tdoa_s"]           : (n_frames, max_tracks, n_pairs)
    # labels["drone"]["class_id"]         : (n_frames, max_tracks)
    # labels["drone"]["activity"]         : (n_frames, max_tracks) bool
    # labels["drone"]["pair_names"]       : same order as feats["mic_pair_names"]
 
Or, for the fully-merged, guaranteed-consistent result in one call:
    from label_alignment import load_aligned_example
    example = load_aligned_example(chunk_dir)
"""
 
from typing import Dict, List, Optional
 
import numpy as np
 
from propagation import MediumConditions, true_tdoa_seconds
from feature_extraction import (
    FeatureConfig,
    extract_features_from_chunk,
    mic_pair_names,
    _mic_pairs,
)
 
CATEGORY_KEY_MAP = {
    "drone": "drones",
    "known_unwanted": "known_unwanted",
}
 
 
def _interp_track(
    times_abs: np.ndarray,
    positions: np.ndarray,
    target_times_abs: np.ndarray,
    onset_s: float,
    offset_s: float,
):
    """
    Linearly interpolates one event's sparse position samples onto
    `target_times_abs` (all in absolute episode time). Returns
    (positions_interp (n_target, 3), active_mask (n_target,) bool),
    where active_mask is True only where target time falls inside this
    event's own [onset_s, offset_s] window (clipped separately by the
    caller to the chunk being processed).
    """
    target_times_abs = np.asarray(target_times_abs, dtype=np.float64)
    active = (target_times_abs >= onset_s) & (target_times_abs <= offset_s)
 
    out = np.zeros((len(target_times_abs), 3), dtype=np.float64)
    if len(times_abs) == 0:
        return out, np.zeros_like(active)
 
    times_abs = np.asarray(times_abs, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    order = np.argsort(times_abs)
    times_sorted = times_abs[order]
    pos_sorted = positions[order]
 
    if len(times_sorted) == 1:
        out[:] = pos_sorted[0]
    else:
        for k in range(3):
            out[:, k] = np.interp(target_times_abs, times_sorted, pos_sorted[:, k])
 
    return out, active
 
 
def _align_category(
    events_meta: List[Dict],
    frame_times_abs: np.ndarray,
    mic_names: List[str],
    mic_positions: Dict[str, np.ndarray],
    medium: MediumConditions,
    max_tracks: int,
    valid_frame_mask: np.ndarray,
) -> Dict:
    n_frames = len(frame_times_abs)
    pair_names = mic_pair_names(mic_names)
    n_pairs = len(pair_names)
 
    position = np.zeros((n_frames, max_tracks, 3), dtype=np.float64)
    tdoa = np.zeros((n_frames, max_tracks, n_pairs), dtype=np.float64)
    class_id = -np.ones((n_frames, max_tracks), dtype=np.int64)
    activity = np.zeros((n_frames, max_tracks), dtype=bool)
 
    n_events = len(events_meta)
    if n_events > max_tracks:
        # Never silently drop data without saying so: this is a real
        # capacity limit the caller needs to know about (raise, don't warn
        # quietly, since it silently corrupts training targets otherwise).
        raise ValueError(
            f"{n_events} simultaneous/overlapping events in this chunk exceed "
            f"max_tracks={max_tracks}. Increase max_tracks or filter the config's "
            f"n_events_range so a chunk can never exceed it."
        )
 
    for track_idx, event in enumerate(events_meta):
        onset_s, offset_s = event["onset_s"], event["offset_s"]
        samples = event["trajectory_samples"]
        if not samples:
            continue
 
        times_abs = np.array([s["t"] for s in samples], dtype=np.float64)
        positions = np.array([s["position_xyz_m"] for s in samples], dtype=np.float64)
        this_class_id = samples[0]["class_id"]
 
        pos_interp, active = _interp_track(times_abs, positions, frame_times_abs, onset_s, offset_s)
        active = active & valid_frame_mask
 
        position[:, track_idx, :] = pos_interp
        activity[:, track_idx] = active
        class_id[active, track_idx] = this_class_id
 
        # Recompute TDOA from the interpolated position (exact same
        # function mixer.py used to generate the original ground truth),
        # rather than interpolating the sparse tdoa_s values directly --
        # keeps position and TDOA labels geometrically consistent at
        # every frame. Uses numeric index pairs (not string-splitting
        # pair_names) so this stays correct even if a mic name itself
        # contains a hyphen (allowed via array_geometry.custom_points).
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            continue
        for p_idx, (i, j) in enumerate(_mic_pairs(mic_names)):
            mic_a = np.asarray(mic_positions[mic_names[i]])
            mic_b = np.asarray(mic_positions[mic_names[j]])
            for f_idx in active_idx:
                tdoa[f_idx, track_idx, p_idx] = true_tdoa_seconds(
                    pos_interp[f_idx], mic_a, mic_b, medium
                )
 
    return {
        "position_xyz_m": position,
        "tdoa_s": tdoa,
        "class_id": class_id,
        "activity": activity,
        "pair_names": pair_names,
        "n_tracks_used": n_events,
    }
 
 
def align_labels_to_frames(
    metadata: Dict,
    frame_times_s: np.ndarray,
    mic_names: List[str],
    max_tracks: int = 3,
    categories: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    metadata : the parsed metadata.json for ONE chunk (as returned inside
        extract_features_from_chunk()'s output, key "metadata").
    frame_times_s : CHUNK-RELATIVE frame times (feats["frame_times_s"] from
        feature_extraction.py) -- absolute-time conversion is handled here.
    mic_names : must match metadata["mic_channel_order"] (and therefore the
        channel order used to build feats["gcc_phat"]).
    max_tracks : fixed number of track slots to pad each category's labels
        to. Must be >= the max simultaneous events possible given your
        episode.drone_events / episode.interferer_events config
        (n_events_range in example_config.yaml) -- otherwise this raises
        rather than silently truncating.
    categories : which metadata categories to align; defaults to both
        "drone" and "known_unwanted" (the two categories mixer.py labels).
    """
    categories = categories or list(CATEGORY_KEY_MAP.keys())
 
    chunk_t0 = metadata["episode_time_range_s"][0]
    chunk_t1 = metadata["episode_time_range_s"][1]
    frame_times_abs = np.asarray(frame_times_s, dtype=np.float64) + chunk_t0
 
    # Frames whose (chunk-relative) time falls outside this chunk's own
    # valid audio span (can happen at the edges depending on STFT framing/
    # padding conventions) should never be marked "active" for any label.
    valid_frame_mask = (frame_times_abs >= chunk_t0) & (frame_times_abs <= chunk_t1)
 
    mic_positions = {k: np.array(v) for k, v in metadata["mic_positions_m"].items()}
    medium = MediumConditions(
        temperature_c=metadata["medium"]["temperature_c"],
        relative_humidity_pct=metadata["medium"]["relative_humidity_pct"],
        pressure_kpa=metadata["medium"]["pressure_kpa"],
    )
 
    out = {}
    for category in categories:
        meta_key = CATEGORY_KEY_MAP[category]
        events_meta = metadata.get(meta_key, [])
        out[category] = _align_category(
            events_meta, frame_times_abs, mic_names, mic_positions, medium,
            max_tracks, valid_frame_mask,
        )
    return out
 
 
def load_aligned_example(
    chunk_dir: str,
    feat_cfg: FeatureConfig = FeatureConfig(),
    max_tracks: int = 3,
    categories: Optional[List[str]] = None,
) -> Dict:
    """
    One-call convenience: extract features AND aligned labels for one
    chunk, guaranteed consistent (same mic_names, same frame grid, same
    pair order) because both come from the same feats dict.
    """
    feats = extract_features_from_chunk(chunk_dir, feat_cfg)
    labels = align_labels_to_frames(
        feats["metadata"], feats["frame_times_s"], feats["mic_names"],
        max_tracks=max_tracks, categories=categories,
    )
    return {"features": feats, "labels": labels}
 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Sanity-check feature/label alignment for one chunk.")
    parser.add_argument("--chunk_dir", required=True)
    parser.add_argument("--max_tracks", type=int, default=3)
    args = parser.parse_args()
 
    example = load_aligned_example(args.chunk_dir, max_tracks=args.max_tracks)
    feats, labels = example["features"], example["labels"]
 
    print(f"n_frames (features) : {feats['logmel'].shape[1]}")
    for category, lab in labels.items():
        n_active_frames = lab["activity"].any(axis=1).sum()
        print(f"[{category}] n_tracks_used={lab['n_tracks_used']}, "
              f"frames with >=1 active source: {n_active_frames}/{lab['activity'].shape[0]}, "
              f"tdoa_s shape={lab['tdoa_s'].shape}")
 
