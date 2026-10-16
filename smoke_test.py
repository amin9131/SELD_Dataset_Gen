"""One training step on a generated split, grouped-GCC layout.
   python smoke_test.py --train_dir output/train"""
import argparse, torch
from feature_extraction import FeatureConfig
from feature_normalization import FeatureNormalizer
from torch_dataset import SELDDataset, build_dataloader
from model import ModelConfig, build_model, infer_dims_from_batch
from pit_loss import pit_loss_single_category, merge_category_targets

ap = argparse.ArgumentParser(); ap.add_argument("--train_dir", required=True)
ap.add_argument("--n_fft", type=int, default=2048)
ap.add_argument("--inter_pairs", default="2", help="all | ref | k (cross-array pairs kept per array pair)")
a = ap.parse_args()
inter = a.inter_pairs if a.inter_pairs in ("all", "ref") else int(a.inter_pairs)

fc = FeatureConfig(pair_mode="grouped", n_fft=a.n_fft, hop_length=a.n_fft // 2, inter_pairs=inter)
norm = FeatureNormalizer().fit(a.train_dir, fc, max_chunks=50)     # fit on TRAIN only
norm.save(f"{a.train_dir}/feature_norm_stats.json")
ds = SELDDataset(a.train_dir, fc, max_tracks=3, normalizer=norm)
batch = next(iter(build_dataloader(ds, batch_size=4, shuffle=True)))
dims = infer_dims_from_batch(batch); print(dims)
model = build_model(ModelConfig(**dims, n_classes=2, max_tracks=6))   # 2 categories x 3 tracks
loss = pit_loss_single_category(model.forward_batch(batch), merge_category_targets(batch))["loss"]
loss.backward(); print("loss", float(loss), "- OK")
