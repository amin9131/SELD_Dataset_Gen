"""
diffuse_noise.py
=================
Injection models for background noise categories that are NOT treated as
localizable point sources: wind and unidentified ambient noise.
 
Two mixing modes are supported per noise category:
 
  "diffuse"      -> noise arrives as a SPATIALLY-COHERENT diffuse field
                     across the array, following the standard spherically
                     isotropic diffuse-field coherence law (see
                     diffuse_field.py): nearby microphones see strongly
                     correlated noise, far-apart microphones see nearly
                     independent noise -- exactly like real outdoor wind/
                     ambient noise, and the physically correct "hard case"
                     for evaluating beamforming/array-processing
                     algorithms (as opposed to per-mic-independent noise,
                     which is artificially easy to suppress).
 
  "directional"  -> noise is treated as a far-field plane wave arriving
                     from one random direction (only a relative delay is
                     applied across the array, no 1/r spreading, since
                     the "source" is effectively at infinity). Useful
                     when the ambient noise has a dominant, roughly
                     constant bearing (e.g. steady wind from one side,
                     distant road noise).
 
Both modes let you independently pick a target SNR (in dB) relative to a
reference LOUDNESS (normally the drone signal's RMS during the windows
where at least one drone is actually active -- see mixer.py, which computes
this once per episode and passes it in as a plain float, `reference_rms`,
rather than a full array. This matters once sources can enter/exit at
random times: a raw reference *array* that's silent most of the episode
would otherwise dilute the RMS and mis-scale the noise).
"""
 
from dataclasses import dataclass
from typing import Dict, List, Literal
 
import numpy as np
 
from audio_pool import rms
from array_geometry import as_matrix
from diffuse_field import generate_diffuse_field
 
 
@dataclass
class NoiseInjectionConfig:
    mode: Literal["diffuse", "directional"] = "diffuse"
    snr_db: float = 10.0          # target SNR of this noise vs. the reference loudness
    direction_azimuth_deg: float = None   # only used for "directional" (None -> random)
    direction_elevation_deg: float = None  # only used for "directional" (None -> random)
    coherent: bool = True          # only used for "diffuse": if True (default), synthesize a
                                     # physically realistic spatially-coherent diffuse field
                                     # (see diffuse_field.py); if False, fall back to the old,
                                     # simpler per-mic-independent noise (no shared structure
                                     # at all -- an easier, less realistic case, kept for
                                     # comparison / ablation studies)
 
 
def _scale_for_target_snr(noise: np.ndarray, reference_rms: float, snr_db: float) -> np.ndarray:
    """Scale `noise` so that 10*log10(P_ref / P_noise) == snr_db."""
    target_noise_rms = reference_rms / (10 ** (snr_db / 20.0))
    return noise * (target_noise_rms / rms(noise))
 
 
def add_diffuse_noise(
    per_mic_signals: Dict[str, np.ndarray],
    noise_pool_draw_fn,
    reference_rms: float,
    cfg: NoiseInjectionConfig,
    mic_positions: Dict[str, np.ndarray],
    fs: float,
    v_sound: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Add background noise across the whole array for the FULL length of
    `per_mic_signals` (background noise is continuous for the whole
    episode, unlike the intermittent point sources).
 
    If `cfg.coherent` (default True): synthesizes a physically realistic
    SPATIALLY-COHERENT diffuse field matching the spherically isotropic
    coherence law for this exact microphone geometry (see
    diffuse_field.py) -- built from independent real-audio-derived
    per-microphone draws, mixed to the correct coherence structure, then
    scaled to the target SNR per channel.
 
    If `cfg.coherent` is False (or the array has only one microphone):
    falls back to the older, simpler model -- a fully INDEPENDENT random
    clip/segment per microphone, with no shared spatial structure at all
    (kept for comparison/ablation only; real wind/ambient noise is not
    actually like this).
 
    `noise_pool_draw_fn(n_samples) -> np.ndarray` must return a mono
    real-audio-derived segment of the requested length (see mixer.py for
    how this is wired to AudioPool + fit_to_duration).
    """
    mic_names = list(per_mic_signals.keys())
    n_samples = len(per_mic_signals[mic_names[0]])
 
    independent = np.stack([noise_pool_draw_fn(n_samples) for _ in mic_names], axis=0)
 
    if cfg.coherent and len(mic_names) > 1:
        _, mic_mat = as_matrix(mic_positions)
        noise_per_mic = generate_diffuse_field(mic_names, mic_mat, independent, fs=fs, v_sound=v_sound)
    else:
        noise_per_mic = {name: independent[i] for i, name in enumerate(mic_names)}
 
    out = {}
    for mic_name in mic_names:
        noise = _scale_for_target_snr(noise_per_mic[mic_name], reference_rms, cfg.snr_db)
        out[mic_name] = per_mic_signals[mic_name] + noise
    return out
 
 
def add_directional_noise(
    per_mic_signals: Dict[str, np.ndarray],
    noise_clip: np.ndarray,
    fs: float,
    mic_positions: Dict[str, np.ndarray],
    reference_rms: float,
    cfg: NoiseInjectionConfig,
    v_sound: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Add a far-field plane-wave noise: the SAME noise clip is delayed
    per-microphone according to a random (or configured) arrival
    direction, with no distance attenuation (source at infinity).
    Models a dominant wind direction or a distant, roughly-constant-
    bearing environmental noise source. Runs for the FULL episode length.
    """
    az = cfg.direction_azimuth_deg
    el = cfg.direction_elevation_deg
    if az is None:
        az = rng.uniform(0, 360)
    if el is None:
        el = rng.uniform(-10, 60)  # mostly near-horizontal for wind/ambient
 
    az_r, el_r = np.deg2rad(az), np.deg2rad(el)
    # unit vector pointing FROM the array TOWARD the noise source
    direction = np.array([
        np.cos(el_r) * np.cos(az_r),
        np.cos(el_r) * np.sin(az_r),
        np.sin(el_r),
    ])
 
    names = list(mic_positions.keys())
    mat = np.stack([mic_positions[n] for n in names], axis=0)
    centroid = mat.mean(axis=0)
 
    n_samples = len(next(iter(per_mic_signals.values())))
    scaled_noise = _scale_for_target_snr(noise_clip[:n_samples], reference_rms, cfg.snr_db)
 
    spec = np.fft.rfft(scaled_noise, n=n_samples + 8)
    freqs = np.fft.rfftfreq(n_samples + 8, d=1.0 / fs)
 
    out = {}
    for name in names:
        # relative path-length difference vs. centroid, projected onto direction
        rel_pos = mic_positions[name] - centroid
        extra_path = -np.dot(rel_pos, direction)  # sign: source is far along `direction`
        tau = extra_path / v_sound
        phase = np.exp(-1j * 2 * np.pi * freqs * tau)
        delayed = np.fft.irfft(spec * phase, n=n_samples + 8)[:n_samples]
        out[name] = per_mic_signals[name] + delayed
 
    return out
