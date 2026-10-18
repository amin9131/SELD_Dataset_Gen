# """
# feature_normalization.py
# =========================
# Per-dataset standardization (zero mean / unit variance) for the log-mel
# and GCC-PHAT features produced by feature_extraction.py.
 
# Both features are z-scored per FEATURE DIMENSION (not per time frame):
#   - log-mel  : statistics per (mic, mel_bin)   -> shape (n_mics, n_mels)
#   - gcc_phat : statistics per (mic_pair, lag)  -> shape (n_pairs, n_lags)
# i.e. every (mic, mel_bin) / (pair, lag) slot gets its own mean/std, pooled
# over BOTH the time-frame axis and every chunk in the fitting dataset. This
# is the standard normalization granularity for spectrogram-like SELD
# features, and fits naturally here since GCC-PHAT amplitude at different
# lags/pairs, and log-mel level in different mel bands/mics, are not on a
# common scale to begin with.
 
# Statistics are accumulated with Welford's online algorithm (single pass,
# constant memory), so fitting scales to arbitrarily large datasets without
# loading every chunk into RAM at once -- following the same "don't assume
# everything fits in memory" discipline audio_pool.py already uses (loads
# clips on demand rather than eagerly).
 
# LEAKAGE WARNING (same category of issue dataset_split.py exists to
# prevent, just one layer up the pipeline): fit this ONLY on the train
# split, then reuse those exact statistics for val/test. Fitting separately
# per split would leak val/test distribution information into what the
# model effectively sees through normalization -- this module does not
# enforce that for you (it has no notion of "split"), enforcing it is the
# caller's responsibility:
 
#     norm = FeatureNormalizer().fit("output/train", feat_cfg)
#     norm.save("output/train/feature_norm_stats.json")
#     # val/test:
#     norm = FeatureNormalizer.load("output/train/feature_norm_stats.json")
 
# This module intentionally does NOT import torch_dataset.py (which
# requires `torch`), so it stays usable in a numpy/scipy-only environment.
# """
 
# import glob
# import json
# import os
# from dataclasses import dataclass
# from typing import Dict, List, Optional
 
# import numpy as np
 
# from feature_extraction import FeatureConfig, extract_features_from_chunk
 
 
# def _find_chunk_dirs(dataset_dir: str) -> List[str]:
#     """Same glob logic as torch_dataset.find_chunk_dirs, duplicated (not
#     imported) here to keep this module independent of the torch
#     dependency -- see module docstring."""
#     pattern = os.path.join(dataset_dir, "episode_*", "chunk_*")
#     return sorted(glob.glob(pattern))
 
 
# class _WelfordAccumulator:
#     """Numerically stable streaming mean/variance, per feature dimension.
#     `shape` is the per-frame feature shape to accumulate over (e.g.
#     (n_mics, n_mels) for log-mel)."""
 
#     def __init__(self, shape):
#         self.shape = shape
#         self.count = 0
#         self.mean = np.zeros(shape, dtype=np.float64)
#         self.m2 = np.zeros(shape, dtype=np.float64)
 
#     def update_batch(self, frames: np.ndarray) -> None:
#         """frames: shape (n_frames, *self.shape).
 
#         Vectorized batch merge (Chan, Golub & LeVeque parallel-variance
#         update): mathematically identical to feeding the frames one by one
#         through Welford's recurrence, but without a Python-level loop over
#         every frame (the old loop did n_chunks x ~376 iterations over
#         arrays as large as (n_pairs, n_lags))."""
#         n_b = frames.shape[0]
#         if n_b == 0:
#             return
#         mean_b = frames.mean(axis=0)
#         m2_b = ((frames - mean_b) ** 2).sum(axis=0)
#         n_a = self.count
#         n = n_a + n_b
#         delta = mean_b - self.mean
#         self.mean = self.mean + delta * (n_b / n)
#         self.m2 = self.m2 + m2_b + delta ** 2 * (n_a * n_b / n)
#         self.count = n
 
#     def finalize(self, eps: float = 1e-8):
#         if self.count < 2:
#             return self.mean, np.ones(self.shape, dtype=np.float64)
#         var = self.m2 / (self.count - 1)
#         std = np.maximum(np.sqrt(var), eps)
#         return self.mean, std
 
 
# # GCC-PHAT feature keys: legacy single tensor, or the grouped INTRA/INTER pair
# # (feature_extraction.FeatureConfig.pair_mode = "grouped"). Each key gets its own
# # per-(pair, lag) statistics.
# GCC_KEYS = ("gcc_phat", "gcc_phat_intra", "gcc_phat_inter")


# @dataclass
# class FeatureNormalizer:
#     logmel_mean: Optional[np.ndarray] = None    # (n_mics, n_mels)
#     logmel_std: Optional[np.ndarray] = None
#     gcc_mean: Optional[np.ndarray] = None        # legacy "gcc_phat": (n_pairs, n_lags)
#     gcc_std: Optional[np.ndarray] = None
#     gcc_intra_mean: Optional[np.ndarray] = None  # "gcc_phat_intra"
#     gcc_intra_std: Optional[np.ndarray] = None
#     gcc_inter_mean: Optional[np.ndarray] = None  # "gcc_phat_inter"
#     gcc_inter_std: Optional[np.ndarray] = None
#     n_chunks_fitted: int = 0

#     _ATTR = {"gcc_phat": ("gcc_mean", "gcc_std"),
#              "gcc_phat_intra": ("gcc_intra_mean", "gcc_intra_std"),
#              "gcc_phat_inter": ("gcc_inter_mean", "gcc_inter_std")}
 
#     def fit(
#         self,
#         dataset_dir: str,
#         feat_cfg: FeatureConfig = FeatureConfig(),
#         max_chunks: Optional[int] = None,
#     ) -> "FeatureNormalizer":
#         chunk_dirs = _find_chunk_dirs(dataset_dir)
#         if not chunk_dirs:
#             raise FileNotFoundError(f"No chunks found under '{dataset_dir}'")
#         if max_chunks is not None:
#             chunk_dirs = chunk_dirs[:max_chunks]
 
#         logmel_acc = None
#         gcc_acc: Dict[str, _WelfordAccumulator] = {}

#         for chunk_dir in chunk_dirs:
#             feats = extract_features_from_chunk(chunk_dir, feat_cfg)
#             logmel = feats["logmel"]   # (n_mics, n_frames, n_mels)

#             if logmel_acc is None:
#                 logmel_acc = _WelfordAccumulator((logmel.shape[0], logmel.shape[2]))
#             # move the time-frame axis to the front: (n_frames, n_mics, n_mels)
#             logmel_acc.update_batch(np.moveaxis(logmel, 1, 0))

#             for key in GCC_KEYS:
#                 if key not in feats:
#                     continue
#                 gcc = feats[key]       # (n_pairs_scope, n_frames, n_lags_scope)
#                 if key not in gcc_acc:
#                     gcc_acc[key] = _WelfordAccumulator((gcc.shape[0], gcc.shape[2]))
#                 gcc_acc[key].update_batch(np.moveaxis(gcc, 1, 0))

#         self.logmel_mean, self.logmel_std = logmel_acc.finalize()
#         for key, acc in gcc_acc.items():
#             m_attr, s_attr = self._ATTR[key]
#             m, sd = acc.finalize()
#             setattr(self, m_attr, m)
#             setattr(self, s_attr, sd)
#         self.n_chunks_fitted = len(chunk_dirs)
#         return self

#     def transform_logmel(self, logmel: np.ndarray) -> np.ndarray:
#         """logmel: (n_mics, n_frames, n_mels) -> same shape, z-scored."""
#         self._check_fitted()
#         mean = self.logmel_mean[:, None, :]   # broadcast over frame axis
#         std = self.logmel_std[:, None, :]
#         return (logmel - mean) / std
 
#     def transform_gcc(self, gcc: np.ndarray, key: str = "gcc_phat") -> np.ndarray:
#         """gcc: (n_pairs, n_frames, n_lags) -> same shape, z-scored with the
#         statistics fitted for `key` ("gcc_phat" | "gcc_phat_intra" | "gcc_phat_inter")."""
#         self._check_fitted()
#         m_attr, s_attr = self._ATTR[key]
#         mean, std = getattr(self, m_attr), getattr(self, s_attr)
#         if mean is None:
#             raise RuntimeError(f"FeatureNormalizer has no statistics for '{key}' "
#                                f"(it was fitted on a dataset/FeatureConfig without that feature).")
#         return (gcc - mean[:, None, :]) / std[:, None, :]

#     def apply(self, feats: dict) -> dict:
#         """Returns a shallow copy of `feats` with 'logmel' and every present GCC
#         key replaced by its normalized version; all other keys untouched."""
#         out = dict(feats)
#         out["logmel"] = self.transform_logmel(feats["logmel"])
#         for key in GCC_KEYS:
#             if key in feats:
#                 out[key] = self.transform_gcc(feats[key], key)
#         return out

#     def _check_fitted(self) -> None:
#         if self.logmel_mean is None or not any(
#                 getattr(self, self._ATTR[k][0]) is not None for k in GCC_KEYS):
#             raise RuntimeError("FeatureNormalizer has not been fit()/load()ed yet.")

#     def save(self, path: str) -> None:
#         self._check_fitted()
#         payload = {"logmel_mean": self.logmel_mean.tolist(),
#                    "logmel_std": self.logmel_std.tolist(),
#                    "n_chunks_fitted": self.n_chunks_fitted}
#         for key in GCC_KEYS:
#             m_attr, s_attr = self._ATTR[key]
#             if getattr(self, m_attr) is not None:
#                 payload[m_attr] = getattr(self, m_attr).tolist()
#                 payload[s_attr] = getattr(self, s_attr).tolist()
#         with open(path, "w") as f:
#             json.dump(payload, f, indent=2)

#     @classmethod
#     def load(cls, path: str) -> "FeatureNormalizer":
#         with open(path, "r") as f:
#             payload = json.load(f)
#         obj = cls(logmel_mean=np.array(payload["logmel_mean"]),
#                   logmel_std=np.array(payload["logmel_std"]),
#                   n_chunks_fitted=payload.get("n_chunks_fitted", 0))
#         for key in GCC_KEYS:          # old files only contain gcc_mean / gcc_std
#             m_attr, s_attr = cls._ATTR[key]
#             if m_attr in payload:
#                 setattr(obj, m_attr, np.array(payload[m_attr]))
#                 setattr(obj, s_attr, np.array(payload[s_attr]))
#         return obj


# if __name__ == "__main__":
#     import argparse
 
#     parser = argparse.ArgumentParser(
#         description="Fit feature normalization stats over a dataset (fit on the TRAIN split only)."
#     )
#     parser.add_argument("--dataset_dir", required=True, help="e.g. output/train")
#     parser.add_argument("--out", required=True, help="e.g. output/train/feature_norm_stats.json")
#     parser.add_argument("--max_chunks", type=int, default=None)
#     args = parser.parse_args()
 
#     norm = FeatureNormalizer().fit(args.dataset_dir, max_chunks=args.max_chunks)
#     norm.save(args.out)
#     print(f"Fitted on {norm.n_chunks_fitted} chunks under '{args.dataset_dir}'.")
#     print(f"logmel_mean shape={norm.logmel_mean.shape}")
#     for k in GCC_KEYS:
#         a = getattr(norm, FeatureNormalizer._ATTR[k][0])
#         if a is not None:
#             print(f"{k} mean shape={a.shape}")
#     print(f"Saved stats to '{args.out}'")


"""
feature_normalization.py
=========================
Per-dataset standardization (zero mean / unit variance) for the log-mel
and GCC-PHAT features produced by feature_extraction.py.

Both features are z-scored per FEATURE DIMENSION (not per time frame):
  - log-mel  : statistics per (mic, mel_bin)   -> shape (n_mics, n_mels)
  - gcc_phat : statistics per (mic_pair, lag)  -> shape (n_pairs, n_lags)

FAST FIT (what changed vs. the original version)
-------------------------------------------------
The original fit() ran the full feature extraction (STFT + log-mel + GCC-PHAT)
on EVERY train chunk, in ONE process, and re-parsed the (large) metadata.json
of every chunk. For ~6000 chunks that is hours. Now:

  1. SUBSAMPLING  -- mean/std of a feature dimension converge long before the
     whole dataset is seen. fit() uses a random (seeded, reproducible) subset of
     `max_chunks` chunks drawn from ALL episodes (default 1500). The old
     max_chunks took the FIRST n sorted chunks = only the first few episodes,
     which is a biased sample; random selection fixes that. Use max_chunks=None
     (CLI: --max_chunks 0) to use everything.
  2. MULTIPROCESSING -- chunks are independent; each worker accumulates its own
     Welford statistics and the partial results are merged exactly
     (Chan-Golub-LeVeque). Same numbers as a sequential pass, up to float rounding.
  3. NO JSON PER CHUNK -- fit only needs the audio plus the array geometry
     (mic names / positions / groups), which is identical for the whole dataset,
     so it is read ONCE from the first chunk's metadata.json.

(The faster GCC-PHAT in feature_extraction.py speeds this up further.)

LEAKAGE WARNING: fit this ONLY on the train split, then reuse those exact
statistics for val/test:

    norm = FeatureNormalizer().fit("output/train", feat_cfg)
    norm.save("output/train/feature_norm_stats.json")
    norm = FeatureNormalizer.load("output/train/feature_norm_stats.json")

This module intentionally does NOT import torch_dataset.py (requires torch).
"""

import concurrent.futures
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

from feature_extraction import FeatureConfig, extract_features


def _find_chunk_dirs(dataset_dir: str) -> List[str]:
    """Same glob logic as torch_dataset.find_chunk_dirs, duplicated (not
    imported) here to keep this module independent of the torch dependency."""
    pattern = os.path.join(dataset_dir, "episode_*", "chunk_*")
    return sorted(glob.glob(pattern))


def _select_chunks(chunk_dirs: List[str], max_chunks: Optional[int], seed: int) -> List[str]:
    """Random subset (seeded -> reproducible) spread over ALL episodes, returned sorted."""
    if max_chunks is None or max_chunks <= 0 or max_chunks >= len(chunk_dirs):
        return chunk_dirs
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(chunk_dirs), size=max_chunks, replace=False))
    return [chunk_dirs[i] for i in idx]


class _WelfordAccumulator:
    """Numerically stable streaming mean/variance, per feature dimension.
    `shape` is the per-frame feature shape to accumulate over (e.g.
    (n_mics, n_mels) for log-mel)."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)

    def update_batch(self, frames: np.ndarray) -> None:
        """frames: shape (n_frames, *self.shape). Chan-Golub-LeVeque batch merge."""
        n_b = frames.shape[0]
        if n_b == 0:
            return
        mean_b = frames.mean(axis=0)
        m2_b = ((frames - mean_b) ** 2).sum(axis=0)
        self.merge_stats(n_b, mean_b, m2_b)

    def merge_stats(self, n_b: int, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        if n_b == 0:
            return
        n_a = self.count
        n = n_a + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + m2_b + delta ** 2 * (n_a * n_b / n)
        self.count = n

    def merge(self, other: "_WelfordAccumulator") -> None:
        self.merge_stats(other.count, other.mean, other.m2)

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
_ACC_KEYS = ("logmel",) + GCC_KEYS


# ---------------------------------------------------------------------------
# Worker side (module-level so it also works with Windows "spawn")
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(feat_cfg, mic_names, mic_positions, mic_groups) -> None:
    # one BLAS thread per worker: N processes x M BLAS threads would oversubscribe the CPU
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _W.update(cfg=feat_cfg, mic_names=mic_names, mic_positions=mic_positions, mic_groups=mic_groups)


def _accumulate_chunks(chunk_dirs: Sequence[str], cfg, mic_names, mic_positions, mic_groups) -> Dict:
    accs: Dict[str, _WelfordAccumulator] = {}
    for chunk_dir in chunk_dirs:
        audio, fs = sf.read(os.path.join(chunk_dir, "audio.wav"), always_2d=True)  # (n_samples, n_mics)
        if audio.shape[1] != len(mic_names):
            raise ValueError(f"{chunk_dir}: {audio.shape[1]} channels, expected {len(mic_names)} "
                             f"(array geometry is read from the first chunk and must be identical everywhere)")
        feats = extract_features(audio, fs, mic_names, cfg, mic_positions, mic_groups=mic_groups)

        logmel = feats["logmel"]   # (n_mics, n_frames, n_mels)
        if "logmel" not in accs:
            accs["logmel"] = _WelfordAccumulator((logmel.shape[0], logmel.shape[2]))
        accs["logmel"].update_batch(np.moveaxis(logmel, 1, 0))

        for key in GCC_KEYS:
            if key not in feats:
                continue
            gcc = feats[key]       # (n_pairs_scope, n_frames, n_lags_scope)
            if key not in accs:
                accs[key] = _WelfordAccumulator((gcc.shape[0], gcc.shape[2]))
            accs[key].update_batch(np.moveaxis(gcc, 1, 0))
    return accs


def _worker_task(batch: List[str]) -> Dict:
    return _accumulate_chunks(batch, _W["cfg"], _W["mic_names"], _W["mic_positions"], _W["mic_groups"])


def _read_array_info(chunk_dir: str):
    """Mic names / positions / groups: constant over the dataset -> parse ONE metadata.json."""
    with open(os.path.join(chunk_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    mic_names = meta["mic_channel_order"]
    mic_positions = {k: np.array(v) for k, v in meta["mic_positions_m"].items()}
    return mic_names, mic_positions, meta.get("mic_groups")


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
        max_chunks: Optional[int] = 1500,
        n_workers: Optional[int] = None,
        seed: int = 0,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> "FeatureNormalizer":
        """
        max_chunks : random subset size (None / 0 = every chunk).
        n_workers  : None -> cpu_count()-1 ; 1 -> run in this process (easiest to debug).
        """
        all_dirs = _find_chunk_dirs(dataset_dir)
        if not all_dirs:
            raise FileNotFoundError(f"No chunks found under '{dataset_dir}'")
        chunk_dirs = _select_chunks(all_dirs, max_chunks, seed)
        mic_names, mic_positions, mic_groups = _read_array_info(chunk_dirs[0])

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) - 1)
        n_workers = max(1, min(n_workers, len(chunk_dirs)))

        t0 = time.time()
        if verbose:
            print(f"[norm] fitting on {len(chunk_dirs)}/{len(all_dirs)} chunks, {n_workers} worker(s)")

        total: Dict[str, _WelfordAccumulator] = {}

        def _merge(part: Dict) -> None:
            for key, acc in part.items():
                if key not in total:
                    total[key] = _WelfordAccumulator(acc.shape)
                total[key].merge(acc)

        if n_workers == 1:
            for i in range(0, len(chunk_dirs), batch_size):
                _merge(_accumulate_chunks(chunk_dirs[i:i + batch_size], feat_cfg,
                                          mic_names, mic_positions, mic_groups))
                if verbose:
                    print(f"[norm] {min(i + batch_size, len(chunk_dirs))}/{len(chunk_dirs)} "
                          f"({time.time() - t0:.0f}s)")
        else:
            # keep NumPy/BLAS single-threaded in the children (inherited env under "spawn")
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
            batches = [chunk_dirs[i:i + batch_size] for i in range(0, len(chunk_dirs), batch_size)]
            done = 0
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, initializer=_init_worker,
                initargs=(feat_cfg, mic_names, mic_positions, mic_groups),
            ) as ex:
                for part, batch in zip(ex.map(_worker_task, batches), batches):
                    _merge(part)
                    done += len(batch)
                    if verbose:
                        print(f"[norm] {done}/{len(chunk_dirs)} ({time.time() - t0:.0f}s)")

        m, s = total["logmel"].finalize()
        self.logmel_mean, self.logmel_std = m, s
        for key in GCC_KEYS:
            if key in total:
                m_attr, s_attr = self._ATTR[key]
                m, s = total[key].finalize()
                setattr(self, m_attr, m)
                setattr(self, s_attr, s)
        self.n_chunks_fitted = len(chunk_dirs)
        if verbose:
            print(f"[norm] done in {time.time() - t0:.0f}s")
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
    parser.add_argument("--max_chunks", type=int, default=1500,
                        help="random subset size (default 1500; 0 = all chunks)")
    parser.add_argument("--n_workers", type=int, default=None, help="default: cpu_count()-1; 1 = no multiprocessing")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    norm = FeatureNormalizer().fit(args.dataset_dir, max_chunks=args.max_chunks or None,
                                   n_workers=args.n_workers, seed=args.seed)
    norm.save(args.out)
    print(f"Fitted on {norm.n_chunks_fitted} chunks under '{args.dataset_dir}'.")
    print(f"logmel_mean shape={norm.logmel_mean.shape}")
    for k in GCC_KEYS:
        a = getattr(norm, FeatureNormalizer._ATTR[k][0])
        if a is not None:
            print(f"{k} mean shape={a.shape}")
    print(f"Saved stats to '{args.out}'")