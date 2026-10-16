"""
feature_normalization.py
=========================
Per-dataset standardization (zero mean / unit variance) for the log-mel
and GCC-PHAT features produced by feature_extraction.py.
 
Both features are z-scored per FEATURE DIMENSION (not per time frame):
  - log-mel  : statistics per (mic, mel_bin)   -> shape (n_mics, n_mels)
  - gcc_phat : statistics per (mic_pair, lag)  -> shape (n_pairs, n_lags)
i.e. every (mic, mel_bin) / (pair, lag) slot gets its own mean/std, pooled
over BOTH the time-frame axis and every chunk in the fitting dataset. This
is the standard normalization granularity for spectrogram-like SELD
features, and fits naturally here since GCC-PHAT amplitude at different
lags/pairs, and log-mel level in different mel bands/mics, are not on a
common scale to begin with.
 
Statistics are accumulated with Welford's online algorithm (single pass,
constant memory), so fitting scales to arbitrarily large datasets without
loading every chunk into RAM at once -- following the same "don't assume
everything fits in memory" discipline audio_pool.py already uses (loads
clips on demand rather than eagerly).
 
LEAKAGE WARNING (same category of issue dataset_split.py exists to
prevent, just one layer up the pipeline): fit this ONLY on the train
split, then reuse those exact statistics for val/test. Fitting separately
per split would leak val/test distribution information into what the
model effectively sees through normalization -- this module does not
enforce that for you (it has no notion of "split"), enforcing it is the
caller's responsibility:
 
    norm = FeatureNormalizer().fit("output/train", feat_cfg)
    norm.save("output/train/feature_norm_stats.json")
    # val/test:
    norm = FeatureNormalizer.load("output/train/feature_norm_stats.json")
 
This module intentionally does NOT import torch_dataset.py (which
requires `torch`), so it stays usable in a numpy/scipy-only environment.
"""
 
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
 
import numpy as np
 
from feature_extraction import FeatureConfig, extract_features_from_chunk
 
 
def _find_chunk_dirs(dataset_dir: str) -> List[str]:
    """Same glob logic as torch_dataset.find_chunk_dirs, duplicated (not
    imported) here to keep this module independent of the torch
    dependency -- see module docstring."""
    pattern = os.path.join(dataset_dir, "episode_*", "chunk_*")
    return sorted(glob.glob(pattern))
 
 
class _WelfordAccumulator:
    """Numerically stable streaming mean/variance, per feature dimension.
    `shape` is the per-frame feature shape to accumulate over (e.g.
    (n_mics, n_mels) for log-mel)."""
 
    def __init__(self, shape):
        self.shape = shape
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
 
    def update_batch(self, frames: np.ndarray) -> None:
        """frames: shape (n_frames, *self.shape).
 
        Vectorized batch merge (Chan, Golub & LeVeque parallel-variance
        update): mathematically identical to feeding the frames one by one
        through Welford's recurrence, but without a Python-level loop over
        every frame (the old loop did n_chunks x ~376 iterations over
        arrays as large as (n_pairs, n_lags))."""
        n_b = frames.shape[0]
        if n_b == 0:
            return
        mean_b = frames.mean(axis=0)
        m2_b = ((frames - mean_b) ** 2).sum(axis=0)
        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + m2_b + delta ** 2 * (n_a * n_b / n)
        self.count = n
 
    def finalize(self, eps: float = 1e-8):
        if self.count < 2:
            return self.mean, np.ones(self.shape, dtype=np.float64)
        var = self.m2 / (self.count - 1)
        std = np.maximum(np.sqrt(var), eps)
        return self.mean, std
 
 
# GCC-PHAT feature keys: legacy single tensor, or the grouped INTRA/INTER pair
# (feature_extraction.FeatureConfig.pair_mode = "grouped"). Each key gets its own
# per-(pair, lag) statistics.
GCC_KEYS = ("gcc_phat", "gcc_phat_intra", "gcc_phat_inter")


@dataclass
class FeatureNormalizer:
    logmel_mean: Optional[np.ndarray] = None    # (n_mics, n_mels)
    logmel_std: Optional[np.ndarray] = None
    gcc_mean: Optional[np.ndarray] = None        # legacy "gcc_phat": (n_pairs, n_lags)
    gcc_std: Optional[np.ndarray] = None
    gcc_intra_mean: Optional[np.ndarray] = None  # "gcc_phat_intra"
    gcc_intra_std: Optional[np.ndarray] = None
    gcc_inter_mean: Optional[np.ndarray] = None  # "gcc_phat_inter"
    gcc_inter_std: Optional[np.ndarray] = None
    n_chunks_fitted: int = 0

    _ATTR = {"gcc_phat": ("gcc_mean", "gcc_std"),
             "gcc_phat_intra": ("gcc_intra_mean", "gcc_intra_std"),
             "gcc_phat_inter": ("gcc_inter_mean", "gcc_inter_std")}
 
    def fit(
        self,
        dataset_dir: str,
        feat_cfg: FeatureConfig = FeatureConfig(),
        max_chunks: Optional[int] = None,
    ) -> "FeatureNormalizer":
        chunk_dirs = _find_chunk_dirs(dataset_dir)
        if not chunk_dirs:
            raise FileNotFoundError(f"No chunks found under '{dataset_dir}'")
        if max_chunks is not None:
            chunk_dirs = chunk_dirs[:max_chunks]
 
        logmel_acc = None
        gcc_acc: Dict[str, _WelfordAccumulator] = {}

        for chunk_dir in chunk_dirs:
            feats = extract_features_from_chunk(chunk_dir, feat_cfg)
            logmel = feats["logmel"]   # (n_mics, n_frames, n_mels)

            if logmel_acc is None:
                logmel_acc = _WelfordAccumulator((logmel.shape[0], logmel.shape[2]))
            # move the time-frame axis to the front: (n_frames, n_mics, n_mels)
            logmel_acc.update_batch(np.moveaxis(logmel, 1, 0))

            for key in GCC_KEYS:
                if key not in feats:
                    continue
                gcc = feats[key]       # (n_pairs_scope, n_frames, n_lags_scope)
                if key not in gcc_acc:
                    gcc_acc[key] = _WelfordAccumulator((gcc.shape[0], gcc.shape[2]))
                gcc_acc[key].update_batch(np.moveaxis(gcc, 1, 0))

        self.logmel_mean, self.logmel_std = logmel_acc.finalize()
        for key, acc in gcc_acc.items():
            m_attr, s_attr = self._ATTR[key]
            m, sd = acc.finalize()
            setattr(self, m_attr, m)
            setattr(self, s_attr, sd)
        self.n_chunks_fitted = len(chunk_dirs)
        return self

    def transform_logmel(self, logmel: np.ndarray) -> np.ndarray:
        """logmel: (n_mics, n_frames, n_mels) -> same shape, z-scored."""
        self._check_fitted()
        mean = self.logmel_mean[:, None, :]   # broadcast over frame axis
        std = self.logmel_std[:, None, :]
        return (logmel - mean) / std
 
    def transform_gcc(self, gcc: np.ndarray, key: str = "gcc_phat") -> np.ndarray:
        """gcc: (n_pairs, n_frames, n_lags) -> same shape, z-scored with the
        statistics fitted for `key` ("gcc_phat" | "gcc_phat_intra" | "gcc_phat_inter")."""
        self._check_fitted()
        m_attr, s_attr = self._ATTR[key]
        mean, std = getattr(self, m_attr), getattr(self, s_attr)
        if mean is None:
            raise RuntimeError(f"FeatureNormalizer has no statistics for '{key}' "
                               f"(it was fitted on a dataset/FeatureConfig without that feature).")
        return (gcc - mean[:, None, :]) / std[:, None, :]

    def apply(self, feats: dict) -> dict:
        """Returns a shallow copy of `feats` with 'logmel' and every present GCC
        key replaced by its normalized version; all other keys untouched."""
        out = dict(feats)
        out["logmel"] = self.transform_logmel(feats["logmel"])
        for key in GCC_KEYS:
            if key in feats:
                out[key] = self.transform_gcc(feats[key], key)
        return out

    def _check_fitted(self) -> None:
        if self.logmel_mean is None or not any(
                getattr(self, self._ATTR[k][0]) is not None for k in GCC_KEYS):
            raise RuntimeError("FeatureNormalizer has not been fit()/load()ed yet.")

    def save(self, path: str) -> None:
        self._check_fitted()
        payload = {"logmel_mean": self.logmel_mean.tolist(),
                   "logmel_std": self.logmel_std.tolist(),
                   "n_chunks_fitted": self.n_chunks_fitted}
        for key in GCC_KEYS:
            m_attr, s_attr = self._ATTR[key]
            if getattr(self, m_attr) is not None:
                payload[m_attr] = getattr(self, m_attr).tolist()
                payload[s_attr] = getattr(self, s_attr).tolist()
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        with open(path, "r") as f:
            payload = json.load(f)
        obj = cls(logmel_mean=np.array(payload["logmel_mean"]),
                  logmel_std=np.array(payload["logmel_std"]),
                  n_chunks_fitted=payload.get("n_chunks_fitted", 0))
        for key in GCC_KEYS:          # old files only contain gcc_mean / gcc_std
            m_attr, s_attr = cls._ATTR[key]
            if m_attr in payload:
                setattr(obj, m_attr, np.array(payload[m_attr]))
                setattr(obj, s_attr, np.array(payload[s_attr]))
        return obj


if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(
        description="Fit feature normalization stats over a dataset (fit on the TRAIN split only)."
    )
    parser.add_argument("--dataset_dir", required=True, help="e.g. output/train")
    parser.add_argument("--out", required=True, help="e.g. output/train/feature_norm_stats.json")
    parser.add_argument("--max_chunks", type=int, default=None)
    args = parser.parse_args()
 
    norm = FeatureNormalizer().fit(args.dataset_dir, max_chunks=args.max_chunks)
    norm.save(args.out)
    print(f"Fitted on {norm.n_chunks_fitted} chunks under '{args.dataset_dir}'.")
    print(f"logmel_mean shape={norm.logmel_mean.shape}")
    for k in GCC_KEYS:
        a = getattr(norm, FeatureNormalizer._ATTR[k][0])
        if a is not None:
            print(f"{k} mean shape={a.shape}")
    print(f"Saved stats to '{args.out}'")
