"""feature_cache.py -- precompute & cache features + aligned labels per chunk."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from dataclasses import asdict

import numpy as np

from feature_extraction import FeatureConfig
from label_alignment import load_aligned_example

GCC_KEYS = ("gcc_phat", "gcc_phat_intra", "gcc_phat_inter")
CATEGORIES = ("drone", "known_unwanted")


def cache_path(chunk_dir, feat_cfg, max_tracks):
    s = json.dumps({"f": asdict(feat_cfg), "t": max_tracks}, sort_keys=True, default=str)
    tag = hashlib.md5(s.encode()).hexdigest()[:8]
    return os.path.join(chunk_dir, f"feat_{tag}.npz")


def example_to_arrays(example):
    feats, labels = example["features"], example["labels"]
    out = {"logmel": np.asarray(feats["logmel"], dtype=np.float32)}
    for k in GCC_KEYS:
        if k in feats:
            out[k] = np.asarray(feats[k], dtype=np.float32)
    for cat, lab in labels.items():
        out[f"{cat}_position"] = np.asarray(lab["position_xyz_m"], dtype=np.float32)
        out[f"{cat}_tdoa"] = np.asarray(lab["tdoa_s"], dtype=np.float32)
        out[f"{cat}_class_id"] = np.asarray(lab["class_id"], dtype=np.int64)
        out[f"{cat}_activity"] = np.asarray(lab["activity"], dtype=bool)
    return out


def load_cached(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_one(args):
    chunk_dir, feat_cfg, max_tracks, overwrite = args
    path = cache_path(chunk_dir, feat_cfg, max_tracks)
    if os.path.exists(path) and not overwrite:
        return 0
    arrs = example_to_arrays(load_aligned_example(chunk_dir, feat_cfg, max_tracks))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)          # uncompressed: fastest to load
    os.replace(tmp, path)
    return 1


if __name__ == "__main__":
    import glob, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--max_tracks", type=int, default=3)
    ap.add_argument("--n_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = FeatureConfig()          # MUST be the same default train_seld.py uses
    dirs = sorted(glob.glob(os.path.join(a.dataset_dir, "episode_*", "chunk_*")))
    tasks = [(d, cfg, a.max_tracks, a.overwrite) for d in dirs]
    t0, done, built = time.time(), 0, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.n_workers) as ex:
        for r in ex.map(build_one, tasks, chunksize=4):
            done += 1
            built += r
            if done % 50 == 0 or done == len(tasks):
                print(f"[cache] {done}/{len(tasks)} ({time.time() - t0:.0f}s)")
    print(f"built {built} new cache files for {len(dirs)} chunks in '{a.dataset_dir}'")