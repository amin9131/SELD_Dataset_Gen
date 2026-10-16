"""
mixer.py
========
Renders one Episode (continuous flight, possibly minutes long in
principle) ONCE as a full continuous multichannel signal, then slices
that continuous signal into consecutive fixed-length chunks. Each chunk
is written out as one training sample, but because chunks are literal
contiguous slices of one continuously-rendered recording, stitching
consecutive chunks back together reproduces the original continuous
audio and continuous ground-truth trajectory exactly -- this is what
enables both single-frame localization training AND sequence/tracking
training from the same generated data.
 
Each source is a SourceEvent (scenario.py): it only exists -- and is only
added into the mix -- during its own [start_s, end_s) window, so the
number of simultaneously-active sources varies naturally across the
episode as events start and end (random polyphony, entries/exits), rather
than a fixed count present for the whole episode.
 
Signal categories and how each is spatialized:
  drone            -> moving point source (Trajectory) active only during
                       its own event window, full propagation physics via
                       moving_propagation.py. This is the localization
                       TARGET.
  known_unwanted   -> same, but labeled separately as a hard-negative /
                       interferer class.
  wind             -> diffuse OR directional background noise, continuous
                       for the whole episode, not a localizable point
                       source.
  ambient_unknown  -> same as wind: diffuse or directional background.
"""
 
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random as pyrandom
 
import numpy as np
 
from array_geometry import MicPositions, as_matrix
from audio_pool import AudioPool, fit_to_duration, normalize_rms, rms
from propagation import MediumConditions, GroundReflectionConfig, true_tdoa_seconds
from moving_propagation import propagate_moving_source_to_array
from diffuse_noise import add_diffuse_noise, add_directional_noise
from scenario import Episode, SourceEvent
 
 
@dataclass
class AssetPools:
    drone: AudioPool
    wind: AudioPool
    ambient_unknown: AudioPool
    known_unwanted: AudioPool
 
 
@dataclass
class MixerConfig:
    fs: int = 48000
    apply_air_absorption: bool = True
    ground_reflection: GroundReflectionConfig = None
    moving_block_size: int = 2048
    metadata_dt_s: float = 0.05   # ground-truth trajectory/TDOA sampling rate (20 Hz default)
    event_fade_s: float = 0.05    # short onset/offset fade to avoid audible clicks at entry/exit
    # --- DCASE-style label activity -----------------------------------
    # A source carries a label ONLY while it is actually audible at the array.
    #   label_time_reference "reception": labels live on the time axis of the
    #     RECORDING. A source emitted at t is heard R/c later (up to ~120 ms at
    #     40 m), and its label position is where it was when the heard sound
    #     was emitted. "emission": legacy -- labels on the emission clock.
    #   label_vad_db: energy threshold (dB below the clip RMS) of the VAD run
    #     on the dry source clip (after fades). Frames below it are silence ->
    #     no label. None disables the VAD (whole event window is active).
    label_time_reference: str = "reception"
    label_vad_db: Optional[float] = 30.0
    label_vad_frame_s: float = 0.02
    label_min_gap_s: float = 0.10       # silent gaps shorter than this are bridged
    label_min_segment_s: float = 0.10   # active runs shorter than this are dropped
    class_map: Dict[str, int] = None  # {class_name: class_id}, built once for the whole dataset
                                        # (see dataset_generator.py) so ids stay consistent
                                        # across every episode/chunk
 
 
def _zeros_per_mic(mic_positions: MicPositions, n_samples: int) -> Dict[str, np.ndarray]:
    return {name: np.zeros(n_samples) for name in mic_positions}
 
 
def _add_at_offset(accum: Dict[str, np.ndarray], contribution: Dict[str, np.ndarray], start_sample: int) -> None:
    """Adds a (shorter) per-mic contribution into the full-episode buffer
    at the given absolute sample offset, clipping if it would run past
    the end of the episode buffer (can happen from float rounding of the
    event's own duration)."""
    for name in accum:
        c = contribution[name]
        end_sample = start_sample + len(c)
        room = len(accum[name]) - start_sample
        if room <= 0:
            continue
        if len(c) > room:
            c = c[:room]
            end_sample = start_sample + len(c)
        accum[name][start_sample:end_sample] += c
 
 
def _add_1d_at_offset(dest: np.ndarray, contribution: np.ndarray, start_sample: int) -> None:
    """Same as `_add_at_offset` but for a single 1D array (used to track
    the drone reference-loudness signal, which is just one channel)."""
    end_sample = start_sample + len(contribution)
    room = len(dest) - start_sample
    if room <= 0:
        return
    if len(contribution) > room:
        contribution = contribution[:room]
        end_sample = start_sample + len(contribution)
    dest[start_sample:end_sample] += contribution
 
 
def _apply_fade(signal: np.ndarray, fs: int, fade_s: float) -> np.ndarray:
    """Short raised-cosine fade-in/out at the boundaries of an event's
    clip, so entries/exits don't produce an audible click."""
    n_fade = int(round(fade_s * fs))
    n = len(signal)
    n_fade = min(n_fade, n // 2)
    if n_fade <= 0:
        return signal
    out = signal.copy()
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_fade)))
    out[:n_fade] *= ramp
    out[-n_fade:] *= ramp[::-1]
    return out
 
 
def _per_pair_tdoa(position: np.ndarray, mic_names: List[str], mic_mat: np.ndarray,
                    medium: MediumConditions) -> Dict[str, float]:
    pairs = {}
    for a in range(len(mic_names)):
        for b in range(a + 1, len(mic_names)):
            tau = true_tdoa_seconds(position, mic_mat[a], mic_mat[b], medium)
            pairs[f"{mic_names[a]}-{mic_names[b]}"] = tau
    return pairs
 
 
def _render_event(
    event: SourceEvent,
    pool: AudioPool,
    fs: int,
    mic_positions: MicPositions,
    medium: MediumConditions,
    mixer_cfg: MixerConfig,
    py_rng: pyrandom.Random,
    return_dry: bool = False,
):
    """Draws a clip sized to this event's own duration, fades it in/out,
    and propagates it along the event's trajectory. The trajectory lives in
    event-LOCAL time (t=0 at event.start_s), so it is rendered with
    t_start=0.0; the caller places the result at the correct absolute
    episode sample offset (see _add_at_offset). Returns
    (per_mic_signal_dict, class_name, source_path)."""
    duration = event.end_s - event.start_s
    clip = pool.random_clip(py_rng)
    sig = fit_to_duration(clip, duration, py_rng)
    sig = normalize_rms(sig, target_rms=1.0)
    sig = _apply_fade(sig, fs, mixer_cfg.event_fade_s)
 
    per_mic = propagate_moving_source_to_array(
        sig, fs, event.trajectory, 0.0, mic_positions, medium,
        apply_air_absorption=mixer_cfg.apply_air_absorption,
        ground_reflection=mixer_cfg.ground_reflection,
        block_size=mixer_cfg.moving_block_size,
    )
    if return_dry:
        return per_mic, clip.class_name, clip.source_path, sig
    return per_mic, clip.class_name, clip.source_path
 
 
# ---------------------------------------------------------------------------
# DCASE-style label activity: WHEN is a source audible, and at what position
# ---------------------------------------------------------------------------

def _active_segments(sig: np.ndarray, fs: int, cfg: MixerConfig) -> List[Tuple[float, float]]:
    """
    Energy VAD on the dry (faded, RMS-normalised) source clip. Returns the
    audible stretches as [(start_s, end_s), ...] in EVENT-LOCAL emission time.
    Gaps shorter than cfg.label_min_gap_s are bridged, runs shorter than
    cfg.label_min_segment_s dropped. An empty list = nothing audible = no label.
    DCASE analogue: the per-file on/offset annotations that keep silent parts
    of a sound-event file from being labelled.
    """
    n = len(sig)
    dur = n / float(fs)
    if cfg.label_vad_db is None or n == 0:
        return [(0.0, dur)] if n else []
    fl = max(1, int(round(cfg.label_vad_frame_s * fs)))
    nf = n // fl
    if nf == 0:
        return [(0.0, dur)]
    r = np.sqrt((sig[: nf * fl].reshape(nf, fl) ** 2).mean(axis=1))
    act = r > rms(sig) * 10.0 ** (-cfg.label_vad_db / 20.0)

    runs, i = [], 0
    while i < nf:
        if act[i]:
            j = i
            while j + 1 < nf and act[j + 1]:
                j += 1
            runs.append([i * fl / fs, ((j + 1) * fl / fs) if j < nf - 1 else dur])
            i = j + 1
        else:
            i += 1
    merged: List[List[float]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < cfg.label_min_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= cfg.label_min_segment_s]


def _delay_to_ref_s(traj, t_local: float, ref: np.ndarray, v: float) -> float:
    t = min(max(t_local, 0.0), traj.duration_s)
    return float(np.linalg.norm(traj.position_at(t) - ref)) / v


def _emission_time_for_reception(traj, tau_local: float, ref: np.ndarray, v: float, n_iter: int = 4) -> float:
    """Emission time te solving te + |p(te) - ref| / v = tau_local (fixed-point
    iteration; converges fast because source speed << speed of sound)."""
    te = tau_local
    for _ in range(n_iter):
        te = tau_local - _delay_to_ref_s(traj, te, ref, v)
    return te


def _segments_to_episode_time(event: SourceEvent, segs_local, ref, v, time_reference: str):
    """Event-local EMISSION segments -> absolute episode time on the label clock."""
    out = []
    for a, b in segs_local:
        if time_reference == "reception":
            a = a + _delay_to_ref_s(event.trajectory, a, ref, v)
            b = b + _delay_to_ref_s(event.trajectory, b, ref, v)
        out.append((event.start_s + a, event.start_s + b))
    return out


def render_episode(
    episode: Episode,
    mic_positions: MicPositions,
    pools: AssetPools,
    mixer_cfg: MixerConfig,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Renders the full episode once, then returns a list of chunk dicts:
        [{"audio": {mic_name: np.ndarray}, "fs": int, "metadata": {...}}, ...]
    one per fixed-length consecutive chunk (any leftover tail shorter than
    one chunk is dropped).
    """
    py_rng = pyrandom.Random(episode.seed)
    fs = mixer_cfg.fs
    medium = MediumConditions(
        temperature_c=episode.temperature_c,
        relative_humidity_pct=episode.relative_humidity_pct,
        pressure_kpa=episode.pressure_kpa,
    )
    mic_names, mic_mat = as_matrix(mic_positions)
 
    total_n = int(round(episode.episode_duration_s * fs))
    mixed_full = _zeros_per_mic(mic_positions, total_n)
    drone_reference_full = np.zeros(total_n)
 
    drone_render = []       # per event: {"class_name", "source_file", "segments_abs"}
    interferer_render = []
    ref_pos = mic_mat.mean(axis=0)   # labels are referenced to the array centroid
 
    # --- drones: each event only exists during its own [start_s, end_s) window ---
    for event in episode.drones:
        per_mic, class_name, source_path, dry = _render_event(
            event, pools.drone, fs, mic_positions, medium, mixer_cfg, py_rng, return_dry=True
        )
        start_sample = int(round(event.start_s * fs))
        _add_at_offset(mixed_full, per_mic, start_sample)
        _add_1d_at_offset(drone_reference_full, per_mic[mic_names[0]], start_sample)
        segs = _segments_to_episode_time(event, _active_segments(dry, fs, mixer_cfg), ref_pos,
                                         medium.v_sound, mixer_cfg.label_time_reference)
        drone_render.append({"class_name": class_name, "source_file": source_path, "segments_abs": segs})
 
    # --- known-but-unwanted interferers: same event-window treatment ---
    for event in episode.interferers:
        per_mic, class_name, source_path, dry = _render_event(
            event, pools.known_unwanted, fs, mic_positions, medium, mixer_cfg, py_rng, return_dry=True
        )
        start_sample = int(round(event.start_s * fs))
        _add_at_offset(mixed_full, per_mic, start_sample)
        segs = _segments_to_episode_time(event, _active_segments(dry, fs, mixer_cfg), ref_pos,
                                         medium.v_sound, mixer_cfg.label_time_reference)
        interferer_render.append({"class_name": class_name, "source_file": source_path, "segments_abs": segs})
 
    # --- reference loudness for noise SNR scaling: RMS over only the
    #     samples where at least one drone event is actually active, so
    #     long silent stretches (between events) don't dilute it ---
    active_mask = np.abs(drone_reference_full) > 1e-12
    reference_rms = rms(drone_reference_full[active_mask]) if active_mask.any() else 1e-3
 
    # --- background noise, continuous over the full episode length ---
    if not pools.wind.is_empty():
        if episode.wind_cfg.mode == "diffuse":
            def draw_wind(n):
                c = pools.wind.random_clip(py_rng)
                return normalize_rms(fit_to_duration(c, n / fs, py_rng), 1.0)
            mixed_full = add_diffuse_noise(mixed_full, draw_wind, reference_rms, episode.wind_cfg,
                                            mic_positions, fs, medium.v_sound, rng)
        else:
            clip = pools.wind.random_clip(py_rng)
            sig = normalize_rms(fit_to_duration(clip, episode.episode_duration_s, py_rng), 1.0)
            mixed_full = add_directional_noise(
                mixed_full, sig, fs, mic_positions, reference_rms,
                episode.wind_cfg, medium.v_sound, rng,
            )
 
    if not pools.ambient_unknown.is_empty():
        if episode.ambient_cfg.mode == "diffuse":
            def draw_amb(n):
                c = pools.ambient_unknown.random_clip(py_rng)
                return normalize_rms(fit_to_duration(c, n / fs, py_rng), 1.0)
            mixed_full = add_diffuse_noise(mixed_full, draw_amb, reference_rms, episode.ambient_cfg,
                                            mic_positions, fs, medium.v_sound, rng)
        else:
            clip = pools.ambient_unknown.random_clip(py_rng)
            sig = normalize_rms(fit_to_duration(clip, episode.episode_duration_s, py_rng), 1.0)
            mixed_full = add_directional_noise(
                mixed_full, sig, fs, mic_positions, reference_rms,
                episode.ambient_cfg, medium.v_sound, rng,
            )
 
    # --- clipping protection at EPISODE level (one gain for the whole
    #     recording). Rescaling each chunk separately would put gain steps
    #     at chunk boundaries and break the continuity guarantee. ---
    peak = max(float(np.max(np.abs(mixed_full[n]))) for n in mic_names)
    episode_gain = 1.0
    if peak > 0.99:
        episode_gain = 0.99 / peak
        for n in mic_names:
            mixed_full[n] = mixed_full[n] * episode_gain
 
    # --- slice the continuous episode into consecutive fixed-length chunks ---
    chunk_n = int(round(episode.chunk_duration_s * fs))
    n_chunks = total_n // chunk_n
    dt_meta = mixer_cfg.metadata_dt_s
    class_map = mixer_cfg.class_map or {}
 
    chunks = []
    for c in range(n_chunks):
        s0, s1 = c * chunk_n, (c + 1) * chunk_n
        t0, t1 = s0 / fs, s1 / fs
        audio_chunk = {name: mixed_full[name][s0:s1] for name in mic_names}
 
        drones_meta = _events_active_in_chunk(
            episode.drones, drone_render, t0, t1, dt_meta, class_map,
            category="drone", mic_names=mic_names, mic_mat=mic_mat, medium=medium,
            include_tdoa=True, ref_pos=ref_pos, time_reference=mixer_cfg.label_time_reference,
        )
        interferers_meta = _events_active_in_chunk(
            episode.interferers, interferer_render, t0, t1, dt_meta, class_map,
            category="known_unwanted", mic_names=mic_names, mic_mat=mic_mat, medium=medium,
            include_tdoa=True, ref_pos=ref_pos, time_reference=mixer_cfg.label_time_reference,
        )
 
        metadata = {
            "fs": fs,
            "chunk_duration_s": episode.chunk_duration_s,
            "episode_time_range_s": [t0, t1],
            "chunk_index": c,
            "n_chunks_in_episode": n_chunks,
            "mic_positions_m": {k: v.tolist() for k, v in mic_positions.items()},
            "medium": {
                "temperature_c": episode.temperature_c,
                "relative_humidity_pct": episode.relative_humidity_pct,
                "pressure_kpa": episode.pressure_kpa,
                "v_sound_m_s": medium.v_sound,
            },
            "class_map": class_map,
            "drones": drones_meta,
            "known_unwanted": interferers_meta,
            "n_active_drones": len(drones_meta),
            "n_active_known_unwanted": len(interferers_meta),
            "wind": {"mode": episode.wind_cfg.mode, "snr_db": episode.wind_cfg.snr_db},
            "ambient_unknown": {"mode": episode.ambient_cfg.mode, "snr_db": episode.ambient_cfg.snr_db},
            "label_time_reference": mixer_cfg.label_time_reference,
            "label_vad_db": mixer_cfg.label_vad_db,
            "seed": episode.seed,
            "episode_gain": episode_gain,
        }
 
        chunks.append({"audio": audio_chunk, "fs": fs, "metadata": metadata})
 
    return chunks
 
 
def _events_active_in_chunk(
    events: List[SourceEvent],
    render_info: List[Dict],
    t0: float,
    t1: float,
    dt_meta: float,
    class_map: Dict[str, int],
    category: str,
    mic_names: List[str],
    mic_mat: np.ndarray,
    medium: MediumConditions,
    include_tdoa: bool,
    ref_pos: np.ndarray = None,
    time_reference: str = "reception",
) -> List[Dict]:
    """
    Metadata list for ONLY the events that are actually AUDIBLE inside
    [t0, t1): an event whose audible segments (VAD, see _active_segments)
    do not overlap this chunk is not listed at all -- so in silence there is
    no label, and a source that appears/disappears is labelled exactly over
    the span it is heard. Times are on the label clock (see
    MixerConfig.label_time_reference); `trajectory_samples[*]["t"]` is that
    clock and `position_xyz_m` is the position of the source when the sound
    heard at that time was emitted (reception mode), so position, TDOA and
    audio always describe the same wavefront. Every sample has an "active"
    flag; `active_segments_s` lists the audible spans (unclipped).
    """
    ref_pos = mic_mat.mean(axis=0) if ref_pos is None else ref_pos
    out = []
    for event, info in zip(events, render_info):
        segs = info["segments_abs"]
        if not any(min(t1, b) > max(t0, a) for a, b in segs):
            continue  # nothing of this event is audible in this chunk

        onset_s, offset_s = segs[0][0], segs[-1][1]
        ov0, ov1 = max(t0, onset_s), min(t1, offset_s)
        class_name = info["class_name"]
        class_id = class_map.get(class_name, -1)

        samples = []
        for tau in np.arange(ov0, ov1, dt_meta):
            tau_local = float(tau) - event.start_s
            te = (_emission_time_for_reception(event.trajectory, tau_local, ref_pos, medium.v_sound)
                  if time_reference == "reception" else tau_local)
            te = min(max(te, 0.0), event.trajectory.duration_s)
            pos = event.trajectory.position_at(te)
            samp = {
                "t": float(tau),
                "position_xyz_m": pos.tolist(),
                "active": any(a <= tau <= b for a, b in segs),
                "class_name": class_name,
                "class_id": class_id,
            }
            if include_tdoa:
                samp["tdoa_s"] = _per_pair_tdoa(pos, mic_names, mic_mat, medium)
            samples.append(samp)

        out.append({
            "category": category,
            "class_name": class_name,
            "class_id": class_id,
            "source_file": info["source_file"],
            "onset_s": onset_s,               # first audible instant (label clock)
            "offset_s": offset_s,             # last audible instant (label clock)
            "active_segments_s": [[a, b] for a, b in segs],
            "emission_onset_s": event.start_s,
            "emission_offset_s": event.end_s,
            "trajectory_samples": samples,
        })

    return out
