"""
torch_dataset.py
=================
PyTorch Dataset/DataLoader layer over this framework's generated dataset
(output_dir/episode_XXXXXX/chunk_XXX/{audio.wav, metadata.json}), built
directly on feature_extraction.py + label_alignment.py so every training
example is guaranteed to have frame-aligned features and multi-track
labels (see label_alignment.py's module docstring for exactly what
"aligned" means and does not mean).
 
NOTE: this is the first module in the project that depends on `torch`
(not used anywhere else in the codebase) -- install it separately
(`pip install torch`) if it isn't already available in your environment.
 
Usage
-----
    from torch_dataset import SELDDataset, build_dataloader
    from feature_extraction import FeatureConfig
    from feature_normalization import FeatureNormalizer
 
    # fit ONCE on the train split, then reuse everywhere (see
    # feature_normalization.py's leakage warning)
    norm = FeatureNormalizer().fit("output/train")
    norm.save("output/train/feature_norm_stats.json")
 
    ds = SELDDataset("output/train", feat_cfg=FeatureConfig(), max_tracks=3, normalizer=norm)
    loader = build_dataloader(ds, batch_size=8, shuffle=True)
 
    for batch in loader:
        batch["logmel"]              # (B, n_mics, n_frames, n_mels)
        batch["gcc_phat"]            # (B, n_pairs, n_frames, n_lags)   [pair_mode="all"]
        batch["gcc_phat_intra"]      # (B, n_intra, n_frames, n_lags_i) [pair_mode="grouped"]
        batch["gcc_phat_inter"]      # (B, n_inter, n_frames, n_lags_o) [pair_mode="grouped"]
        batch["drone_position"]      # (B, n_frames, max_tracks, 3)
        batch["drone_tdoa"]          # (B, n_frames, max_tracks, n_pairs)
        batch["drone_class_id"]      # (B, n_frames, max_tracks)
        batch["drone_activity"]      # (B, n_frames, max_tracks) bool
        batch["known_unwanted_*"]    # same fields, other category
        batch["chunk_dir"]           # list[str], not stacked (for debugging)
"""
 
import glob
import os
from typing import Dict, List, Optional
 
import torch
from torch.utils.data import DataLoader, Dataset
 
from feature_extraction import FeatureConfig
from feature_normalization import FeatureNormalizer
from label_alignment import load_aligned_example
from feature_cache import (CATEGORIES, GCC_KEYS, cache_path,
                           example_to_arrays, load_cached)
 
 
def find_chunk_dirs(dataset_dir: str) -> List[str]:
    """
    dataset_dir is either dataset_generator.py's `output_dir`, or one of
    its split subfolders (output_dir/train, output_dir/val, output_dir/test
    -- see dataset_generator.py's split-mode layout). Expects the standard
    episode_XXXXXX/chunk_XXX layout that mixer.py + dataset_generator.py
    always write.
    """
    pattern = os.path.join(dataset_dir, "episode_*", "chunk_*")
    return sorted(glob.glob(pattern))
 
 
class SELDDataset(Dataset):
    """
    One example = one chunk. __getitem__ runs feature extraction + label
    alignment on demand (no caching) -- for large datasets, consider
    pre-computing and caching features to disk separately if I/O becomes
    the bottleneck; that is NOT implemented here.
    """
 
    def __init__(
        self,
        dataset_dir: str,
        feat_cfg: FeatureConfig = FeatureConfig(),
        max_tracks: int = 3,
        categories: Optional[List[str]] = None,
        normalizer: Optional[FeatureNormalizer] = None,
        use_cache: bool = True,
    ):
        self.chunk_dirs = find_chunk_dirs(dataset_dir)
        if not self.chunk_dirs:
            raise FileNotFoundError(
                f"No chunks found under '{dataset_dir}' "
                f"(expected the episode_XXXXXX/chunk_XXX layout dataset_generator.py writes)"
            )
        self.feat_cfg = feat_cfg
        self.max_tracks = max_tracks
        self.categories = categories
        self.normalizer = normalizer
        self.use_cache = use_cache
 
    def __len__(self) -> int:
        return len(self.chunk_dirs)
 
    # def __getitem__(self, idx: int) -> Dict:
    #     chunk_dir = self.chunk_dirs[idx]
    #     example = load_aligned_example(chunk_dir, self.feat_cfg, self.max_tracks, self.categories)
    #     feats, labels = example["features"], example["labels"]
 
    #     logmel = feats["logmel"]
    #     if self.normalizer is not None:
    #         logmel = self.normalizer.transform_logmel(logmel)

    #     item = {
    #         "logmel": torch.from_numpy(logmel).float(),      # (n_mics, n_frames, n_mels)
    #         "chunk_dir": chunk_dir,
    #     }
    #     # "gcc_phat" (legacy, pair_mode="all") or "gcc_phat_intra"/"gcc_phat_inter"
    #     # (pair_mode="grouped"): whichever feature_extraction produced.
    #     for key in ("gcc_phat", "gcc_phat_intra", "gcc_phat_inter"):
    #         if key in feats:
    #             arr = feats[key]                              # (n_pairs_*, n_frames, n_lags_*)
    #             if self.normalizer is not None:
    #                 arr = self.normalizer.transform_gcc(arr, key)
    #             item[key] = torch.from_numpy(arr).float()
    #     for category, lab in labels.items():
    #         item[f"{category}_position"] = torch.from_numpy(lab["position_xyz_m"]).float()
    #         item[f"{category}_tdoa"] = torch.from_numpy(lab["tdoa_s"]).float()
    #         item[f"{category}_class_id"] = torch.from_numpy(lab["class_id"]).long()
    #         item[f"{category}_activity"] = torch.from_numpy(lab["activity"]).bool()
    #     return item
    def __getitem__(self, idx: int) -> Dict:
        chunk_dir = self.chunk_dirs[idx]
        path = cache_path(chunk_dir, self.feat_cfg, self.max_tracks)
        if self.use_cache and os.path.exists(path):
            data = load_cached(path)
        else:
            data = example_to_arrays(
                load_aligned_example(chunk_dir, self.feat_cfg, self.max_tracks))

        logmel = data["logmel"]
        if self.normalizer is not None:
            logmel = self.normalizer.transform_logmel(logmel)
        item = {"logmel": torch.from_numpy(logmel).float(), "chunk_dir": chunk_dir}

        for key in GCC_KEYS:
            if key in data:
                arr = data[key]
                if self.normalizer is not None:
                    arr = self.normalizer.transform_gcc(arr, key)
                item[key] = torch.from_numpy(arr).float()

        for cat in (self.categories or CATEGORIES):
            item[f"{cat}_position"] = torch.from_numpy(data[f"{cat}_position"]).float()
            item[f"{cat}_tdoa"] = torch.from_numpy(data[f"{cat}_tdoa"]).float()
            item[f"{cat}_class_id"] = torch.from_numpy(data[f"{cat}_class_id"]).long()
            item[f"{cat}_activity"] = torch.from_numpy(data[f"{cat}_activity"]).bool()
        return item
 
 
def seld_collate(batch: List[Dict]) -> Dict:
    """
    Stacks a list of __getitem__ dicts into batched tensors. Requires every
    example in the batch to have IDENTICAL shapes per key -- this holds
    automatically as long as every chunk in the dataset was generated with
    the same fs/chunk_duration_s (episode.chunk_duration_s in the YAML
    config) and every call uses the same FeatureConfig/max_tracks. If
    shapes disagree, this raises immediately rather than silently
    dropping/truncating data.
    """
    keys = [k for k in batch[0].keys() if k != "chunk_dir"]
    out: Dict = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) > 1:
            raise ValueError(
                f"Inconsistent tensor shape for '{k}' across batch: {shapes}. "
                f"All chunks must share the same fs / chunk_duration_s (config), "
                f"FeatureConfig, and max_tracks for batching to work."
            )
        out[k] = torch.stack(tensors, dim=0)
    out["chunk_dir"] = [b["chunk_dir"] for b in batch]
    return out
 
 
def build_dataloader(
    dataset: SELDDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=seld_collate, **kwargs,
    )
 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Smoke-test the Dataset/DataLoader on a generated dataset.")
    parser.add_argument("--dataset_dir", required=True, help="e.g. output/ or output/train")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tracks", type=int, default=3)
    parser.add_argument("--norm_stats", default=None, help="optional path to a saved FeatureNormalizer JSON")
    args = parser.parse_args()
 
    normalizer = FeatureNormalizer.load(args.norm_stats) if args.norm_stats else None
    ds = SELDDataset(args.dataset_dir, max_tracks=args.max_tracks, normalizer=normalizer)
    print(f"Dataset: {len(ds)} chunks found under '{args.dataset_dir}'")
 
    loader = build_dataloader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    for k, v in batch.items():
        if k == "chunk_dir":
            print(f"{k}: {v}")
        else:
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
 
