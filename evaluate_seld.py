"""
evaluate_seld.py
================
Metrics + CLI for the SELDModel trained by train_seld.py.

Per chunk, predicted track slots are matched to ground-truth slots with the
SAME Hungarian assignment pit_loss.py uses; metrics are then computed on
the matched pairs, split by category (drone / known_unwanted):

  precision / recall / F1   frame-level activity (sigmoid(logit) > act_thr)
  loc_err_m                 ||p_hat - p||  on frames where both are active (TP)
  ang_err_deg               angle between (p_hat - c) and (p - c), c = array centroid
  az_err_deg / el_err_deg   azimuth (wrapped) / elevation errors w.r.t. c
  range_err_m               | ||p_hat - c|| - ||p - c|| |
  class_acc                 argmax(class_logit) == class_id, on TP frames

Positions are predicted in units of `pos_scale` metres (see train_seld.py)
and multiplied back to metres here.

Usage:
    python evaluate_seld.py --ckpt runs/crnn_01/best.pt --data_dir output/val
    python evaluate_seld.py --ckpt runs/crnn_01/best.pt --data_dir output/test
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from feature_extraction import FeatureConfig
from feature_normalization import FeatureNormalizer
from model import ModelConfig, build_model
from pit_loss import PITLossConfig, merge_category_targets, pit_loss_single_category
from torch_dataset import SELDDataset, build_dataloader, find_chunk_dirs

CATEGORIES = ("drone", "known_unwanted")   # slot order used by merge_category_targets


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_targets(batch, pos_scale):
    """Returns (targets for the loss: position / pos_scale, targets in metres for metrics)."""
    tgt_m = merge_category_targets(batch, CATEGORIES)
    tgt = dict(tgt_m)
    tgt["position"] = tgt_m["position"] / pos_scale
    return tgt, tgt_m


def array_centroid_from_dataset(dataset_dir):
    chunk = find_chunk_dirs(dataset_dir)[0]
    with open(os.path.join(chunk, "metadata.json")) as f:
        meta = json.load(f)
    return np.mean(np.array(list(meta["mic_positions_m"].values())), axis=0)


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90))}


@torch.no_grad()
def evaluate_model(model, loader, device, pos_scale, centroid, n_slots_per_category,
                   act_thr=0.5, pit_cfg=None):
    pit_cfg = pit_cfg or PITLossConfig()
    model.eval()
    acc = {c: defaultdict(list) for c in CATEGORIES}
    cnt = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
    loss_sum, n_items = 0.0, 0

    for batch in loader:
        b = to_device(batch, device)
        pred = model(b["logmel"], b["gcc_phat"])
        tgt, tgt_m = make_targets(b, pos_scale)
        out = pit_loss_single_category(pred, tgt, pit_cfg)
        B = b["logmel"].shape[0]
        loss_sum += out["loss"].item() * B
        n_items += B

        p_pos = (pred["position"] * pos_scale).cpu().numpy()                       # (B,T,P,3) metres
        p_act = (torch.sigmoid(pred["activity_logit"]) > act_thr).cpu().numpy()    # (B,T,P)
        p_cls = pred["class_logit"].argmax(-1).cpu().numpy()                       # (B,T,P)
        t_pos = tgt_m["position"].cpu().numpy()
        t_act = tgt_m["activity"].cpu().numpy()
        t_cls = tgt_m["class_id"].cpu().numpy()

        for bi, (rows, cols) in enumerate(out["assignments"]):
            n_gt = t_act[bi].sum(-1)                                                # (T,) polyphony
            for r, c in zip(rows, cols):
                cat = CATEGORIES[min(int(c) // n_slots_per_category, len(CATEGORIES) - 1)]
                pa, ta = p_act[bi, :, r], t_act[bi, :, c]
                tp = pa & ta
                cnt[cat]["tp"] += int(tp.sum())
                cnt[cat]["fp"] += int((pa & ~ta).sum())
                cnt[cat]["fn"] += int((~pa & ta).sum())
                idx = np.where(tp)[0]
                if idx.size == 0:
                    continue

                ph = p_pos[bi, idx, r] - centroid
                th = t_pos[bi, idx, c] - centroid
                rn_p = np.linalg.norm(ph, axis=1)
                rn_t = np.linalg.norm(th, axis=1)
                cosang = (ph * th).sum(1) / np.maximum(rn_p * rn_t, 1e-9)
                az_p, az_t = np.arctan2(ph[:, 1], ph[:, 0]), np.arctan2(th[:, 1], th[:, 0])
                el_p = np.degrees(np.arcsin(np.clip(ph[:, 2] / np.maximum(rn_p, 1e-9), -1, 1)))
                el_t = np.degrees(np.arcsin(np.clip(th[:, 2] / np.maximum(rn_t, 1e-9), -1, 1)))

                a = acc[cat]
                a["loc"].extend(np.linalg.norm(ph - th, axis=1))
                a["ang"].extend(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
                a["az"].extend(np.abs(np.degrees(np.arctan2(np.sin(az_p - az_t), np.cos(az_p - az_t)))))
                a["el"].extend(np.abs(el_p - el_t))
                a["rng"].extend(np.abs(rn_p - rn_t))
                a["cls_ok"].extend(p_cls[bi, idx, r] == t_cls[bi, idx, c])
                a["n_gt"].extend(n_gt[idx])
                a["gt_range"].extend(rn_t)

    res = {"loss": loss_sum / max(n_items, 1), "n_chunks": n_items}
    for cat in CATEGORIES:
        tp, fp, fn = cnt[cat]["tp"], cnt[cat]["fp"], cnt[cat]["fn"]
        a = acc[cat]
        loc = np.asarray(a["loc"])
        poly = np.asarray(a["n_gt"])
        rng = np.asarray(a["gt_range"])
        res[cat] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "n_tp_frames": tp,
            "loc_err_m": _stats(a["loc"]),
            "ang_err_deg": _stats(a["ang"]),
            "az_err_deg": _stats(a["az"]),
            "el_err_deg": _stats(a["el"]),
            "range_err_m": _stats(a["rng"]),
            "class_acc": float(np.mean(a["cls_ok"])) if a["cls_ok"] else None,
            "loc_err_by_polyphony": {
                k: _stats(loc[sel]) for k, sel in
                {"1": poly == 1, "2": poly == 2, ">=3": poly >= 3}.items()
            } if loc.size else {},
            "loc_err_by_range_m": {
                f"{lo}-{'inf' if hi > 1e8 else hi}": _stats(loc[(rng >= lo) & (rng < hi)])
                for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 1e9)]
            } if loc.size else {},
        }
    return res


def _fmt(s, nd=2):
    return "n/a" if s["mean"] is None else f"{s['mean']:.{nd}f} (med {s['median']:.{nd}f}, p90 {s['p90']:.{nd}f})"


def print_report(res):
    print(f"loss={res['loss']:.4f}  chunks={res['n_chunks']}")
    for cat in CATEGORIES:
        r = res[cat]
        acc = "n/a" if r["class_acc"] is None else f"{r['class_acc']:.3f}"
        print(f"[{cat}] P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"| class_acc={acc} | TP frames={r['n_tp_frames']}")
        print(f"    loc_err_m    : {_fmt(r['loc_err_m'])}")
        print(f"    ang_err_deg  : {_fmt(r['ang_err_deg'], 1)}   az: {_fmt(r['az_err_deg'], 1)}   el: {_fmt(r['el_err_deg'], 1)}")
        print(f"    range_err_m  : {_fmt(r['range_err_m'])}")
        for k, s in r["loc_err_by_polyphony"].items():
            print(f"    loc_err by #active={k:>3}: {_fmt(s)}  n={s['n']}")
        for k, s in r["loc_err_by_range_m"].items():
            print(f"    loc_err by range {k:>7} m: {_fmt(s)}  n={s['n']}")


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**ck["model_cfg"])).to(device)
    model.load_state_dict(ck["model_state"])
    return model, ck, FeatureConfig(**ck["feat_cfg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True, help="e.g. output/val or output/test")
    ap.add_argument("--norm_stats", default=None, help="default: the TRAIN stats path stored in the checkpoint")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--act_thr", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, feat_cfg = load_checkpoint(args.ckpt, device)
    norm = FeatureNormalizer.load(args.norm_stats or ck["norm_stats"])
    ds = SELDDataset(args.data_dir, feat_cfg=feat_cfg, max_tracks=ck["dataset_max_tracks"], normalizer=norm)
    loader = build_dataloader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    centroid = array_centroid_from_dataset(args.data_dir)

    res = evaluate_model(model, loader, device, ck["pos_scale"], centroid,
                         ck["dataset_max_tracks"], act_thr=args.act_thr)
    print_report(res)

    split = os.path.basename(os.path.normpath(args.data_dir))
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"metrics_{split}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
