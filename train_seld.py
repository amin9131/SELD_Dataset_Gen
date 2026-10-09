"""
train_seld.py
=============
Trains model.SELDModel with pit_loss.pit_loss_single_category on the
dataset produced by dataset_generator.py (train/val split folders).

  * Features are normalized with the TRAIN-split statistics only
    (--norm_stats output/train/feature_norm_stats.json); the same file is
    reused for val/test (evaluate_seld.py reads its path from the checkpoint).
  * Both label categories (drone, known_unwanted) are merged into one pool of
    2 * max_tracks slots (pit_loss.merge_category_targets); the class head
    separates them.
  * Positions are regressed in units of `pos_scale` metres (targets divided
    by pos_scale for the loss, multiplied back for metrics) so the position
    term does not dominate BCE / cross-entropy.
  * FeatureConfig() defaults from feature_extraction.py must be the same ones
    used when feature_normalization.py was run.

Outputs in --out_dir: best.pt, last.pt, history.csv, args.json.

Usage:
    python train_seld.py --train_dir output/train --val_dir output/val \
        --norm_stats output/train/feature_norm_stats.json --out_dir runs/crnn_01
    python train_seld.py ... --overfit 16 --epochs 150 --eval_every 10   # sanity check
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import Subset

from evaluate_seld import (CATEGORIES, array_centroid_from_dataset, evaluate_model,
                           make_targets, print_report, to_device)
from feature_extraction import FeatureConfig
from feature_normalization import FeatureNormalizer
from model import ModelConfig, build_model
from pit_loss import PITLossConfig, pit_loss_single_category
from torch_dataset import SELDDataset, build_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_overfit_indices(ds, n):
    """n evenly spaced chunks that contain at least one active drone."""
    ok = []
    for i, chunk_dir in enumerate(ds.chunk_dirs):
        with open(os.path.join(chunk_dir, "metadata.json")) as f:
            if json.load(f)["n_active_drones"] > 0:
                ok.append(i)
    if not ok:
        raise RuntimeError("no chunk with an active drone found")
    sel = np.linspace(0, len(ok) - 1, min(n, len(ok))).astype(int)
    return [ok[j] for j in sel]


def save_ckpt(path, model, model_cfg, feat_cfg, args, n_classes, class_map, epoch, val_res):
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "feat_cfg": asdict(feat_cfg),
        "pos_scale": args.pos_scale,
        "dataset_max_tracks": args.max_tracks,
        "n_classes": n_classes,
        "class_map": class_map,
        "norm_stats": args.norm_stats,
        "epoch": epoch,
        "val_metrics": val_res,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--class_map", default=None, help="default: <parent of train_dir>/class_map.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pos_scale", type=float, default=50.0, help="metres; >= largest source distance")
    ap.add_argument("--max_tracks", type=int, default=3, help="per category (SELDDataset max_tracks)")
    ap.add_argument("--backbone", choices=["crnn", "conformer"], default="crnn")
    ap.add_argument("--non_causal", action="store_true")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=10, help="early stopping, in evaluations")
    ap.add_argument("--overfit", type=int, default=0, help="train+eval on N chunks only (sanity check)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    cm_path = args.class_map or os.path.join(os.path.dirname(os.path.normpath(args.train_dir)), "class_map.json")
    with open(cm_path) as f:
        cm = json.load(f)
    n_classes, class_map = cm["num_classes"], cm["class_name_to_id"]

    norm = FeatureNormalizer.load(args.norm_stats)
    feat_cfg = FeatureConfig()
    train_ds = SELDDataset(args.train_dir, feat_cfg, args.max_tracks, normalizer=norm)
    if args.overfit:
        train_ds = Subset(train_ds, pick_overfit_indices(train_ds, args.overfit))
        val_ds = train_ds
        val_dir = args.train_dir
    else:
        val_ds = SELDDataset(args.val_dir, feat_cfg, args.max_tracks, normalizer=norm)
        val_dir = args.val_dir

    # model dims come from one real example (nothing is guessed)
    base = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    item = base[0]
    n_mics, n_mels = item["logmel"].shape[0], item["logmel"].shape[2]
    n_pairs, n_lags = item["gcc_phat"].shape[0], item["gcc_phat"].shape[2]
    fp = 4
    model_cfg = ModelConfig(
        n_mics=n_mics, n_pairs=n_pairs, n_mels=n_mels, n_lags=n_lags,
        n_classes=n_classes, max_tracks=len(CATEGORIES) * args.max_tracks,
        backbone=args.backbone, causal=not args.non_causal, d_model=args.d_model,
        freq_pool=fp, logmel_freq_bins=min(4, n_mels // fp ** 2),
        gcc_freq_bins=min(32, n_lags // fp ** 2),   # 351 lags -> 21 bins after two pools
    )
    model = build_model(model_cfg).to(device)
    print(f"device={device} | train chunks={len(train_ds)} val chunks={len(val_ds)} | "
          f"params={sum(p.numel() for p in model.parameters()):,} | "
          f"mics={n_mics} pairs={n_pairs} mels={n_mels} lags={n_lags} classes={n_classes} "
          f"slots={model_cfg.max_tracks}")

    kw = dict(persistent_workers=True) if args.num_workers > 0 else {}
    train_loader = build_dataloader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers,
                                    drop_last=len(train_ds) > args.batch_size, **kw)
    val_loader = build_dataloader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, **kw)
    centroid = array_centroid_from_dataset(val_dir)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    pit_cfg = PITLossConfig()

    hist_path = os.path.join(args.out_dir, "history.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "drone_f1",
                                "drone_loc_err_m", "drone_ang_err_deg", "drone_class_acc"])

    best, bad = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            b = to_device(batch, device)
            pred = model(b["logmel"], b["gcc_phat"])
            tgt, _ = make_targets(b, args.pos_scale)
            loss = pit_loss_single_category(pred, tgt, pit_cfg)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            bs = b["logmel"].shape[0]
            run += loss.item() * bs
            n += bs
        train_loss = run / max(n, 1)

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            print(f"ep {epoch:03d} | train {train_loss:.4f} | {time.time() - t0:.0f}s")
            continue

        res = evaluate_model(model, val_loader, device, args.pos_scale, centroid, args.max_tracks, pit_cfg=pit_cfg)
        val_loss, d = res["loss"], res["drone"]
        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        loc = d["loc_err_m"]["mean"]
        ang = d["ang_err_deg"]["mean"]
        print(f"ep {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | drone F1 {d['f1']:.3f} "
              f"loc {'n/a' if loc is None else f'{loc:.2f}m'} ang {'n/a' if ang is None else f'{ang:.1f}deg'} "
              f"| lr {lr:.1e} | {time.time() - t0:.0f}s")
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, lr, d["f1"], loc, ang, d["class_acc"]])

        save_ckpt(os.path.join(args.out_dir, "last.pt"), model, model_cfg, feat_cfg, args,
                  n_classes, class_map, epoch, res)
        if val_loss < best:
            best, bad = val_loss, 0
            save_ckpt(os.path.join(args.out_dir, "best.pt"), model, model_cfg, feat_cfg, args,
                      n_classes, class_map, epoch, res)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no val improvement for {bad} evaluations)")
                break

    print(f"\nbest val loss {best:.4f} -> {os.path.join(args.out_dir, 'best.pt')}")
    ck = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False)
    print_report(ck["val_metrics"])


if __name__ == "__main__":
    main()
