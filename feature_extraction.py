"""
feature_extraction.py
======================
Feature extraction for chunks produced by dataset_generator.py / mixer.py.
 
This module is intentionally geometry-agnostic: the array in this
framework can be linear/circular/tetrahedral/custom/distributed (see
array_geometry.py), so features that assume a fixed 4-channel FOA/
Ambisonics layout (e.g. intensity vectors, SALSA) are NOT used here.
Instead, two array-agnostic features are computed, from ONE shared STFT
so they are frame-aligned with each other and with the ground-truth
`trajectory_samples` timestamps in metadata.json (see mixer.py):
 
  1. Log-mel spectrogram, per microphone channel.
     Standard spectral/timbre feature for the detection/classification
     side of a SELD model.
 
  2. Frame-wise GCC-PHAT, per microphone PAIR.
     Same phase-transform technique as the `gcc_phat` sanity check in
     validate_gcc_phat.py, but computed per-STFT-frame (instead of once
     over a whole clip) so it becomes a genuine per-frame localization
     feature that lines up with the per-pair `tdoa_s` ground truth
     mixer.py already attaches to every trajectory sample.
 
Both features are computed from the SAME scipy.signal.stft call style
already used elsewhere in this codebase (see diffuse_field.py, which
calls `stft(multichannel_array, fs=fs, nperseg=..., axis=-1)` on a
(M, n_samples) array) -- this module follows that same convention
rather than introducing a new STFT/framing convention.
 
Typical usage
-------------
    from feature_extraction import extract_features_from_chunk, FeatureConfig
 
    feats = extract_features_from_chunk("output/episode_000000/chunk_000",
                                         FeatureConfig())
    feats["logmel"]        # (n_mics, n_frames, n_mels)
    feats["gcc_phat"]      # (n_pairs, n_frames, n_lags)
    feats["mic_pair_names"]  # e.g. ["mic1-mic2", "mic1-mic3", ...]
    feats["frame_times_s"]  # (n_frames,) frame CENTER times, seconds,
                            # relative to the start of this chunk -- use
                            # this to align against trajectory_samples[t]
                            # in metadata.json (which is in absolute
                            # episode time; add metadata's
                            # episode_time_range_s[0] if you need that).
"""
 
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import soundfile as sf
from scipy.signal import stft

from array_geometry import infer_mic_groups, split_pairs_by_group
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
@dataclass
class FeatureConfig:
    n_fft: int = 2048 # 1024, 2048
    hop_length: int = 1024 # 512, 1024          # noverlap = n_fft - hop_length
    n_mels: int = 64
    fmin_hz: float = 20.0
    fmax_hz: Optional[float] = None   # None -> fs / 2
    log_eps: float = 1e-10
 
    # GCC-PHAT lag window. If None, it is derived automatically from
    # mic_positions (max pairwise distance / v_sound, + 20% margin) in
    # extract_features(); if mic_positions is also unavailable, falls
    # back to `default_max_tau_s`. Either way, the EFFECTIVE window is
    # capped to +/- n_fft // 2 samples (an n_fft-point IFFT cannot
    # resolve lags beyond that without aliasing -- see
    # compute_gcc_phat_framewise). A warning is raised if the resolved
    # max_tau_s would have needed a wider window than n_fft allows; for
    # widely-spaced arrays, increase n_fft rather than relying on this cap.
    max_tau_s: Optional[float] = None
    default_max_tau_s: float = 0.0015   # 1.5 ms fallback (~0.5 m mic spacing)
    v_sound: float = 343.0

    # --- pair handling -------------------------------------------------
    # "all"     : legacy -- ONE GCC-PHAT tensor over every mic pair, one
    #             shared lag window (sized by the longest pair).
    # "grouped" : pairs are split into INTRA-array (same sub-array) and
    #             INTER-array (different sub-arrays), each with its OWN lag
    #             window and its own tensor: feats["gcc_phat_intra"] and
    #             feats["gcc_phat_inter"]. Needed for distributed arrays:
    #             a compact sub-array only has a few meaningful lags (e.g.
    #             +/-6 samples), which a shared +/-175-sample window
    #             squeezes into a handful of bins out of ~350.
    pair_mode: str = "all"
    gcc_interp_intra: int = 8   # sub-sample interpolation of the correlation
                                # (zero-padded irfft). Integer-lag GCC-PHAT on
                                # a 4-8 cm array gives ~10 deg bearing steps.
    gcc_interp_inter: int = 1   # long baseline already has ~140 lags: no need
    inter_pairs: object = "all" # which cross-array pairs to keep: "all" | "ref" | k (int);
                                # see array_geometry.split_pairs_by_group. With 3+ sub-arrays
                                # "ref"/2 cuts the inter-branch input from 48 to 3/6 pairs.
 
 
# ---------------------------------------------------------------------------
# Mel filterbank (implemented directly with numpy -- no extra dependency
# beyond what the rest of this project already uses: numpy/scipy/soundfile)
# ---------------------------------------------------------------------------
 
def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)
 
 
def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
 
 
def build_mel_filterbank(
    n_fft: int, fs: float, n_mels: int, fmin_hz: float, fmax_hz: Optional[float]
) -> np.ndarray:
    """
    Returns a (n_mels, n_freq_bins) triangular mel filterbank matrix,
    where n_freq_bins = n_fft // 2 + 1 (matches np.fft.rfftfreq / the
    frequency axis scipy.signal.stft produces for a real input).
    """
    fmax_hz = fmax_hz or fs / 2.0
    n_freq_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, fs / 2.0, n_freq_bins)
 
    mel_min, mel_max = _hz_to_mel(np.array([fmin_hz, fmax_hz]))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
 
    fb = np.zeros((n_mels, n_freq_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        left_slope = (freqs - f_left) / max(f_center - f_left, 1e-12)
        right_slope = (f_right - freqs) / max(f_right - f_center, 1e-12)
        fb[m - 1] = np.clip(np.minimum(left_slope, right_slope), 0.0, None)
    return fb
 
 
# ---------------------------------------------------------------------------
# Shared STFT
# ---------------------------------------------------------------------------
 
def _shared_stft(multichannel: np.ndarray, fs: float, cfg: FeatureConfig):
    """
    multichannel: shape (n_mics, n_samples).
    Returns (freqs, frame_times_s, Zxx) with Zxx shape (n_mics, n_freqs, n_frames),
    using the same stft(..., axis=-1) convention as diffuse_field.py.
    """
    noverlap = cfg.n_fft - cfg.hop_length
    freqs, frame_times_s, Zxx = stft(
        multichannel, fs=fs, nperseg=cfg.n_fft, noverlap=noverlap, axis=-1
    )
    return freqs, frame_times_s, Zxx
 
 
# ---------------------------------------------------------------------------
# Feature 1: log-mel per channel
# ---------------------------------------------------------------------------
 
def compute_logmel(Zxx: np.ndarray, fs: float, cfg: FeatureConfig) -> np.ndarray:
    """
    Zxx: (n_mics, n_freqs, n_frames) complex STFT (from _shared_stft).
    Returns log-mel, shape (n_mics, n_frames, n_mels).
    """
    n_mics, n_freqs, n_frames = Zxx.shape
    fb = build_mel_filterbank(cfg.n_fft, fs, cfg.n_mels, cfg.fmin_hz, cfg.fmax_hz)  # (n_mels, n_freqs)
 
    power = np.abs(Zxx) ** 2  # (n_mics, n_freqs, n_frames)
    # (n_mics, n_frames, n_freqs) @ (n_freqs, n_mels) -> (n_mics, n_frames, n_mels)
    mel_power = np.einsum("mft,kf->mtk", power, fb)
    return np.log(mel_power + cfg.log_eps)
 
 
# ---------------------------------------------------------------------------
# Feature 2: frame-wise GCC-PHAT per mic pair
# ---------------------------------------------------------------------------
 
def _mic_pairs(mic_names: List[str]) -> List[Tuple[int, int]]:
    pairs = []
    for a in range(len(mic_names)):
        for b in range(a + 1, len(mic_names)):
            pairs.append((a, b))
    return pairs
 
 
def mic_pair_names(mic_names: List[str]) -> List[str]:
    """
    Public, single source of truth for mic-pair ORDER and naming.
    Both compute_gcc_phat_framewise() (this file) and label_alignment.py
    (which builds the matching ground-truth TDOA tensor) call this, so the
    pair axis of the GCC-PHAT feature and the pair axis of the TDOA label
    are guaranteed to be in the exact same order.
    """
    return [f"{mic_names[i]}-{mic_names[j]}" for i, j in _mic_pairs(mic_names)]
 
 
def _resolve_max_tau_s(
    cfg: FeatureConfig, mic_names: List[str], mic_positions: Optional[Dict[str, np.ndarray]]
) -> float:
    if cfg.max_tau_s is not None:
        return cfg.max_tau_s
    if mic_positions is not None:
        max_dist = 0.0
        for i in range(len(mic_names)):
            for j in range(i + 1, len(mic_names)):
                pi = np.asarray(mic_positions[mic_names[i]])
                pj = np.asarray(mic_positions[mic_names[j]])
                max_dist = max(max_dist, float(np.linalg.norm(pi - pj)))
        if max_dist > 0:
            return 1.2 * max_dist / cfg.v_sound  # +20% margin
    return cfg.default_max_tau_s
 
 
def compute_gcc_phat_framewise(
    Zxx: np.ndarray,
    fs: float,
    mic_names: List[str],
    cfg: FeatureConfig,
    mic_positions: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Zxx: (n_mics, n_freqs, n_frames) complex STFT (from _shared_stft, same
    call used for compute_logmel -- guarantees frame alignment).
 
    Uses exactly the same PHAT-normalize -> ifft -> wrap-around-lag-window
    technique as the whole-clip `gcc_phat()` sanity check in
    validate_gcc_phat.py, applied per STFT frame and vectorized over
    frames instead of over one long clip -- including that function's
    n_fft//2 cap on the lag window (an n_fft-point IFFT cannot resolve
    lags beyond +/-n_fft//2 without aliasing).
 
    Returns
    -------
    gcc : np.ndarray, shape (n_pairs, n_frames, n_lags)
    pair_names : list of "{mic_a}-{mic_b}" strings, same order as pairs
    lags_s : np.ndarray, shape (n_lags,) -- the time lag (seconds)
             each column of `gcc` corresponds to
    """
    n_mics, n_freqs, n_frames = Zxx.shape
    max_tau_s = _resolve_max_tau_s(cfg, mic_names, mic_positions)
    requested_shift = int(round(max_tau_s * fs))
    nyquist_shift = cfg.n_fft // 2  # an n_fft-point IFFT only resolves lags in [-n_fft//2, n_fft//2]
                                      # without aliasing/wrap-around -- see validate_gcc_phat.py's
                                      # gcc_phat(), which applies this exact same cap.
    max_shift = max(1, min(requested_shift, nyquist_shift))
 
    if requested_shift > nyquist_shift:
        min_n_fft = 2 * requested_shift
        warnings.warn(
            f"GCC-PHAT lag window requested (+/-{requested_shift} samples, from "
            f"max_tau_s={max_tau_s * 1000:.2f} ms) exceeds what n_fft={cfg.n_fft} can resolve "
            f"without aliasing (+/-{nyquist_shift} samples). Clamping to +/-{nyquist_shift}. "
            f"For this array geometry, use n_fft >= {min_n_fft} (e.g. the next power of two) "
            f"to cover the full lag range, or pass an explicit FeatureConfig.max_tau_s.",
            stacklevel=2,
        )
 
    pairs = _mic_pairs(mic_names)
    pair_names = mic_pair_names(mic_names)
    n_lags = 2 * max_shift + 1
    gcc = np.zeros((len(pairs), n_frames, n_lags), dtype=np.float64)
 
    for p_idx, (i, j) in enumerate(pairs):
        R = Zxx[i] * np.conj(Zxx[j])           # (n_freqs, n_frames)
        R = R / np.maximum(np.abs(R), 1e-12)    # PHAT normalization
 
        # irfft along the frequency axis, per frame, using n_fft samples
        # (matches the "n_fft" used for the STFT itself)
        cc = np.fft.irfft(R, n=cfg.n_fft, axis=0)   # (n_fft, n_frames)
 
        # same wrap-around slice as validate_gcc_phat.gcc_phat():
        # cc = concat(cc[-max_shift:], cc[:max_shift+1])
        wrapped = np.concatenate([cc[-max_shift:, :], cc[: max_shift + 1, :]], axis=0)
        gcc[p_idx] = wrapped.T  # (n_frames, n_lags)
 
    lags_s = np.arange(-max_shift, max_shift + 1) / float(fs)
    return gcc, pair_names, lags_s
 
 
def _max_tau_for_pairs(cfg, pairs, mic_names, mic_positions) -> float:
    if cfg.max_tau_s is not None:
        return cfg.max_tau_s
    if mic_positions is not None and pairs:
        dmax = max(float(np.linalg.norm(np.asarray(mic_positions[mic_names[i]])
                                        - np.asarray(mic_positions[mic_names[j]])))
                   for i, j in pairs)
        if dmax > 0:
            return 1.2 * dmax / cfg.v_sound   # +20% margin
    return cfg.default_max_tau_s


def compute_gcc_phat_for_pairs(
    Zxx: np.ndarray, fs: float, mic_names: List[str], pairs: List[Tuple[int, int]],
    cfg: FeatureConfig, max_tau_s: float, interp: int = 1,
):
    """
    Frame-wise GCC-PHAT for an explicit list of (i, j) pairs, with its own
    lag window and an optional sub-sample interpolation factor.

    interp > 1: the PHAT-normalised cross-spectrum is zero-padded to
    n_fft * interp points before the inverse FFT, which band-limited-
    interpolates the correlation function (lag step = 1 / (fs * interp)).
    Amplitude is rescaled by `interp` so values stay comparable.

    Returns (gcc (n_pairs, n_frames, n_lags), pair_names, lags_s (n_lags,)).
    """
    n_mics, n_freqs, n_frames = Zxx.shape
    n_ifft = cfg.n_fft * interp
    requested = int(round(max_tau_s * fs * interp))
    max_shift = max(1, min(requested, n_ifft // 2))
    if requested > n_ifft // 2:
        warnings.warn(f"GCC-PHAT lag window (+/-{requested} lags) exceeds what "
                      f"n_fft*interp={n_ifft} resolves; clamped to +/-{n_ifft // 2}.", stacklevel=2)
    gcc = np.zeros((len(pairs), n_frames, 2 * max_shift + 1), dtype=np.float64)
    for p_idx, (i, j) in enumerate(pairs):
        R = Zxx[i] * np.conj(Zxx[j])
        R = R / np.maximum(np.abs(R), 1e-12)
        cc = np.fft.irfft(R, n=n_ifft, axis=0) * interp          # (n_ifft, n_frames)
        wrapped = np.concatenate([cc[-max_shift:, :], cc[: max_shift + 1, :]], axis=0)
        gcc[p_idx] = wrapped.T
    names = [f"{mic_names[i]}-{mic_names[j]}" for i, j in pairs]
    lags_s = np.arange(-max_shift, max_shift + 1) / (float(fs) * interp)
    return gcc, names, lags_s


def compute_gcc_phat_grouped(Zxx, fs, mic_names, cfg, mic_positions=None, mic_groups=None) -> Dict:
    """INTRA-array and INTER-array GCC-PHAT, each with its own lag window.
    A scope with no pairs (e.g. INTER for a single array) is omitted."""
    intra, inter = split_pairs_by_group(mic_names, mic_groups, cfg.inter_pairs)
    out = {}
    for scope, pairs, interp in (("intra", intra, cfg.gcc_interp_intra),
                                 ("inter", inter, cfg.gcc_interp_inter)):
        if not pairs:
            continue
        tau = _max_tau_for_pairs(cfg, pairs, mic_names, mic_positions)
        g, names, lags = compute_gcc_phat_for_pairs(Zxx, fs, mic_names, pairs, cfg, tau, interp)
        out[scope] = {"gcc": g, "pair_names": names, "lags_s": lags}
    return out


def ordered_pairs(mic_names: List[str], pair_mode: str = "all",
                  mic_groups: Optional[List[str]] = None,
                  inter_pairs="all") -> List[Tuple[int, int]]:
    """
    Single source of truth for the ORDER of mic pairs in pair-indexed
    tensors (e.g. the TDOA label of label_alignment.py).
      "all"     -> all pairs, lexicographic (legacy order)
      "grouped" -> INTRA-array pairs first, then INTER-array pairs; i.e. the
                   concatenation of feats["gcc_phat_intra"] and
                   feats["gcc_phat_inter"] along the pair axis.
    """
    if pair_mode == "grouped":
        intra, inter = split_pairs_by_group(mic_names, mic_groups, inter_pairs)
        return intra + inter
    return _mic_pairs(mic_names)


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------
 
def extract_features(
    multichannel: np.ndarray,
    fs: float,
    mic_names: List[str],
    cfg: FeatureConfig = FeatureConfig(),
    mic_positions: Optional[Dict[str, np.ndarray]] = None,
    mic_groups: Optional[List[str]] = None,
) -> Dict:
    """
    multichannel : np.ndarray, shape (n_samples, n_mics) -- e.g. what
        soundfile.read() returns for a multichannel WAV (this matches how
        dataset_generator.py writes audio.wav: np.stack(..., axis=1)).
    mic_names : channel order, must match metadata["mic_channel_order"].
    mic_positions : optional {mic_name: (x,y,z)} (from metadata["mic_positions_m"]),
        used only to auto-size the GCC-PHAT lag window.
    """
    mc = np.asarray(multichannel, dtype=np.float64).T  # -> (n_mics, n_samples)
    freqs, frame_times_s, Zxx = _shared_stft(mc, fs, cfg)
 
    logmel = compute_logmel(Zxx, fs, cfg)

    if cfg.pair_mode == "grouped":
        grouped = compute_gcc_phat_grouped(Zxx, fs, mic_names, cfg, mic_positions, mic_groups)
        out = {
            "logmel": logmel,
            "mic_names": list(mic_names),
            "mic_groups": list(mic_groups or infer_mic_groups(mic_names)),
            "frame_times_s": frame_times_s,
            "fs": fs, "n_fft": cfg.n_fft, "hop_length": cfg.hop_length,
            "mic_pair_names": [],
        }
        for scope, d in grouped.items():          # intra first, then inter
            out[f"gcc_phat_{scope}"] = d["gcc"]    # (n_pairs_scope, n_frames, n_lags_scope)
            out[f"mic_pair_names_{scope}"] = d["pair_names"]
            out[f"gcc_lags_s_{scope}"] = d["lags_s"]
            out["mic_pair_names"] += d["pair_names"]
        return out

    gcc, pair_names, lags_s = compute_gcc_phat_framewise(Zxx, fs, mic_names, cfg, mic_positions)
 
    return {
        "logmel": logmel,                # (n_mics, n_frames, n_mels)
        "gcc_phat": gcc,                 # (n_pairs, n_frames, n_lags)
        "mic_names": list(mic_names),
        "mic_pair_names": pair_names,
        "frame_times_s": frame_times_s,   # (n_frames,) relative to chunk start
        "gcc_lags_s": lags_s,             # (n_lags,)
        "fs": fs,
        "n_fft": cfg.n_fft,
        "hop_length": cfg.hop_length,
    }
 
 
def extract_features_from_chunk(chunk_dir: str, cfg: FeatureConfig = FeatureConfig()) -> Dict:
    """
    Convenience wrapper matching dataset_generator.py's output layout:
        <chunk_dir>/audio.wav
        <chunk_dir>/metadata.json
    Reads both, resolves mic_names/mic_positions from metadata.json, and
    calls extract_features(). This is the intended way to run this module
    over a generated dataset.
    """
    audio_path = os.path.join(chunk_dir, "audio.wav")
    meta_path = os.path.join(chunk_dir, "metadata.json")
 
    multichannel, fs = sf.read(audio_path, always_2d=True)  # (n_samples, n_mics)
 
    with open(meta_path, "r") as f:
        metadata = json.load(f)
 
    mic_names = metadata["mic_channel_order"]
    mic_positions = {k: np.array(v) for k, v in metadata["mic_positions_m"].items()}
 
    feats = extract_features(multichannel, fs, mic_names, cfg, mic_positions,
                             mic_groups=metadata.get("mic_groups"))
    feats["chunk_dir"] = chunk_dir
    feats["metadata"] = metadata
    return feats
 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Extract log-mel + GCC-PHAT features from one chunk.")
    parser.add_argument("--chunk_dir", required=True, help="e.g. output/episode_000000/chunk_000")
    args = parser.parse_args()
 
    feats = extract_features_from_chunk(args.chunk_dir)
    print(f"logmel   shape: {feats['logmel'].shape}   (n_mics, n_frames, n_mels)")
    print(f"gcc_phat shape: {feats['gcc_phat'].shape}  (n_pairs, n_frames, n_lags)")
    print(f"mic pairs: {feats['mic_pair_names']}")
    print(f"n_frames : {len(feats['frame_times_s'])}, frame span: "
          f"{feats['frame_times_s'][0]:.3f}s - {feats['frame_times_s'][-1]:.3f}s")
