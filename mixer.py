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
from typing import Dict, List
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
    return per_mic, clip.class_name, clip.source_path
 
 
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
 
    drone_render = []       # per drone event: {"class_name", "source_file"}
    interferer_render = []
 
    # --- drones: each event only exists during its own [start_s, end_s) window ---
    for event in episode.drones:
        per_mic, class_name, source_path = _render_event(
            event, pools.drone, fs, mic_positions, medium, mixer_cfg, py_rng
        )
        start_sample = int(round(event.start_s * fs))
        _add_at_offset(mixed_full, per_mic, start_sample)
        _add_1d_at_offset(drone_reference_full, per_mic[mic_names[0]], start_sample)
        drone_render.append({"class_name": class_name, "source_file": source_path})
 
    # --- known-but-unwanted interferers: same event-window treatment ---
    for event in episode.interferers:
        per_mic, class_name, source_path = _render_event(
            event, pools.known_unwanted, fs, mic_positions, medium, mixer_cfg, py_rng
        )
        start_sample = int(round(event.start_s * fs))
        _add_at_offset(mixed_full, per_mic, start_sample)
        interferer_render.append({"class_name": class_name, "source_file": source_path})
 
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
            include_tdoa=True,
        )
        interferers_meta = _events_active_in_chunk(
            episode.interferers, interferer_render, t0, t1, dt_meta, class_map,
            category="known_unwanted", mic_names=mic_names, mic_mat=mic_mat, medium=medium,
            include_tdoa=True,
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
) -> List[Dict]:
    """Builds the metadata list for only the events that actually overlap
    [t0, t1) -- this is what makes the number of "active" sources per
    chunk vary naturally with each event's onset/offset."""
    out = []
    for event, info in zip(events, render_info):
        ov0, ov1 = max(t0, event.start_s), min(t1, event.end_s)
        if ov0 >= ov1:
            continue  # this event does not overlap this chunk at all
 
        local0, local1 = ov0 - event.start_s, ov1 - event.start_s
        samples = event.trajectory.sample(local0, local1, dt_meta)
 
        class_name = info["class_name"]
        class_id = class_map.get(class_name, -1)
        for samp in samples:
            samp["t"] = samp["t"] + event.start_s  # back to absolute episode time
            samp["class_name"] = class_name
            samp["class_id"] = class_id
            if include_tdoa:
                samp["tdoa_s"] = _per_pair_tdoa(np.array(samp["position_xyz_m"]), mic_names, mic_mat, medium)
 
        out.append({
            "category": category,
            "class_name": class_name,
            "class_id": class_id,
            "source_file": info["source_file"],
            "onset_s": event.start_s,
            "offset_s": event.end_s,
            "trajectory_samples": samples,
        })
 
    return out
 
