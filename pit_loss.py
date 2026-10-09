"""
pit_loss.py
===========
Permutation-Invariant Training (PIT) loss for the multi-track labels
label_alignment.py produces: position_xyz_m / tdoa_s / class_id / activity,
each shaped (n_frames, max_tracks, ...) -- see that file's module
docstring, point 3 ("VARIABLE POLYPHONY"), for why this shape exists.
 
Why PIT is required here (not optional)
----------------------------------------
Track slot index in the ground truth is assigned by event ORDER within a
chunk (label_alignment.py: `for track_idx, event in enumerate(events_meta)`)
-- an arbitrary bookkeeping order, not a meaningful identity. A model's
output head also produces some fixed number of track slots, in whatever
order it settles into during training. There is no reason slot 0 of the
model's output should correspond to slot 0 of the label. Training with a
naive fixed-index loss (e.g. plain MSE between pred[:, k] and
label[:, k]) would arbitrarily punish/reward the model for track
ORDERING, not for getting the sources right, and typically prevents
convergence entirely once more than one track is active.
 
PIT resolves this: for every training example, find the assignment
(predicted slot -> ground-truth slot) that minimizes total loss -- via
the Hungarian algorithm (scipy.optimize.linear_sum_assignment) -- then
compute the actual (differentiable) loss under that assignment.
 
Matching granularity: ONE permutation per (batch element, category),
pooled over ALL frames of the chunk -- not a separate permutation per
frame. This matches label_alignment.py's own track semantics: one track
slot = one SourceEvent (scenario.py), which spans many contiguous frames
within a chunk, so track identity is already meant to stay fixed across
the chunk's duration. Permuting per-frame would throw that consistency
away for no benefit (and would make it impossible to later thread
per-frame detections into a continuous track).
 
Honesty note carried over from label_alignment.py: PIT resolves the
LABEL/matching-format side of the "variable polyphony" problem. It does
not, and cannot, un-mix the physically-mixed input audio -- the model
still has to learn to separate overlapping sources from the shared
log-mel / GCC-PHAT features (see feature_extraction.py); PIT just makes
sure the loss doesn't fight the model over arbitrary track-order
bookkeeping while it does so.
"""
 
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
 
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
 
 
@dataclass
class PITLossConfig:
    position_weight: float = 1.0
    activity_weight: float = 1.0
    class_weight: float = 1.0
    tdoa_weight: float = 0.0   # off by default -- optional auxiliary signal;
                                # tdoa_s is already an exact function of
                                # position_xyz_m + mic geometry (see
                                # label_alignment.py), so this is redundant
                                # with the position loss unless you want to
                                # explicitly encourage the model to also
                                # match the TDOA feature space directly.
    class_loss_ignore_index: int = -1  # matches label_alignment.py's padding value
 
 
def _pairwise_cost_matrix(
    pred_position: torch.Tensor,          # (n_pred, n_frames, 3)
    pred_activity_logit: torch.Tensor,    # (n_pred, n_frames)
    target_position: torch.Tensor,        # (n_tgt, n_frames, 3)
    target_activity: torch.Tensor,        # (n_tgt, n_frames) bool
    cfg: PITLossConfig,
    pred_class_logit: Optional[torch.Tensor] = None,   # (n_pred, n_frames, n_classes)
    target_class_id: Optional[torch.Tensor] = None,    # (n_tgt, n_frames) long
    pred_tdoa: Optional[torch.Tensor] = None,           # (n_pred, n_frames, n_pairs)
    target_tdoa: Optional[torch.Tensor] = None,         # (n_tgt, n_frames, n_pairs)
) -> torch.Tensor:
    """
    Cost matrix (n_pred, n_tgt) for ONE (batch element, category), pooled
    over every frame in the chunk. Differentiable w.r.t. every `pred_*`
    input -- only the Hungarian solver itself (outside this function)
    needs a detached numpy copy.
    """
    n_pred = pred_position.shape[0]
    n_tgt = target_position.shape[0]
    cost = pred_position.new_zeros(n_pred, n_tgt)
 
    tgt_active_f = target_activity.float()  # (n_tgt, n_frames)
 
    for p in range(n_pred):
        for t in range(n_tgt):
            mask = tgt_active_f[t]                      # (n_frames,)
            n_active = mask.sum().clamp(min=1.0)
 
            # activity: every frame, this target's ground truth vs. this
            # predicted slot's activity logit
            act_loss = F.binary_cross_entropy_with_logits(
                pred_activity_logit[p], target_activity[t].float(), reduction="mean"
            )
 
            # position: MSE, only counted on frames where this target is active
            pos_err = ((pred_position[p] - target_position[t]) ** 2).sum(dim=-1)  # (n_frames,)
            pos_loss = (pos_err * mask).sum() / n_active
 
            c = cfg.activity_weight * act_loss + cfg.position_weight * pos_loss
 
            if pred_class_logit is not None and target_class_id is not None:
                valid = mask.bool() & (target_class_id[t] != cfg.class_loss_ignore_index)
                if valid.any():
                    ce = F.cross_entropy(
                        pred_class_logit[p][valid], target_class_id[t][valid], reduction="mean"
                    )
                else:
                    ce = pred_position.new_tensor(0.0)
                c = c + cfg.class_weight * ce
 
            if cfg.tdoa_weight > 0 and pred_tdoa is not None and target_tdoa is not None:
                tdoa_err = ((pred_tdoa[p] - target_tdoa[t]) ** 2).mean(dim=-1)  # (n_frames,)
                tdoa_loss = (tdoa_err * mask).sum() / n_active
                c = c + cfg.tdoa_weight * tdoa_loss
 
            cost[p, t] = c
 
    return cost
 
 
def pit_loss_single_category(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    cfg: PITLossConfig = PITLossConfig(),
) -> Dict:
    """
    pred (model output; you provide the head that produces these):
      "position"       : (B, n_frames, n_pred_tracks, 3)
      "activity_logit" : (B, n_frames, n_pred_tracks)          -- raw logits, pre-sigmoid
      "class_logit"    : (B, n_frames, n_pred_tracks, n_classes), optional
      "tdoa"           : (B, n_frames, n_pred_tracks, n_pairs), optional
 
    target (from label_alignment.py / torch_dataset.py, one category):
      "position" : (B, n_frames, max_tracks, 3)
      "activity" : (B, n_frames, max_tracks) bool
      "class_id" : (B, n_frames, max_tracks) long, optional
      "tdoa"     : (B, n_frames, max_tracks, n_pairs), optional
 
    n_pred_tracks is expected to equal max_tracks (design your model's
    output head with exactly `max_tracks` slots, matching the
    SELDDataset(..., max_tracks=...) it was trained against). Unequal
    counts are handled (Hungarian matches min(n_pred, n_tgt) pairs) but
    any unmatched slots contribute nothing to the loss, which likely
    is not what you want -- keep them equal.
 
    Returns {"loss": scalar tensor, "assignments": list of (pred_idx, tgt_idx)
    numpy index arrays, one pair per batch element}.
    """
    B = pred["position"].shape[0]
    losses = []
    assignments: List[Tuple] = []
 
    for b in range(B):
        pred_position = pred["position"][b].permute(1, 0, 2)            # (n_pred, n_frames, 3)
        pred_activity_logit = pred["activity_logit"][b].permute(1, 0)   # (n_pred, n_frames)
        target_position = target["position"][b].permute(1, 0, 2)
        target_activity = target["activity"][b].permute(1, 0)
 
        pred_class_logit = pred.get("class_logit")
        target_class_id = target.get("class_id")
        pred_class_logit = pred_class_logit[b].permute(1, 0, 2) if pred_class_logit is not None else None
        target_class_id = target_class_id[b].permute(1, 0) if target_class_id is not None else None
 
        pred_tdoa = pred.get("tdoa")
        target_tdoa = target.get("tdoa")
        pred_tdoa = pred_tdoa[b].permute(1, 0, 2) if pred_tdoa is not None else None
        target_tdoa = target_tdoa[b].permute(1, 0, 2) if target_tdoa is not None else None
 
        cost = _pairwise_cost_matrix(
            pred_position, pred_activity_logit, target_position, target_activity, cfg,
            pred_class_logit, target_class_id, pred_tdoa, target_tdoa,
        )
 
        row_idx, col_idx = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append((row_idx, col_idx))
        losses.append(cost[row_idx, col_idx].sum())
 
    total = torch.stack(losses).mean()
    return {"loss": total, "assignments": assignments}
 
 
def pit_loss_multi_category(
    pred_by_category: Dict[str, Dict[str, torch.Tensor]],
    target_by_category: Dict[str, Dict[str, torch.Tensor]],
    cfg: PITLossConfig = PITLossConfig(),
    category_weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Convenience wrapper for both categories label_alignment.py produces
    ("drone" and "known_unwanted"): runs pit_loss_single_category per
    category and returns a combined weighted sum, alongside each
    category's own breakdown (useful for logging drone-only vs.
    interferer-only loss curves separately).
    """
    category_weights = category_weights or {c: 1.0 for c in pred_by_category}
    per_category = {}
    total = 0.0
    for category, pred in pred_by_category.items():
        target = target_by_category[category]
        result = pit_loss_single_category(pred, target, cfg)
        per_category[category] = result
        total = total + category_weights.get(category, 1.0) * result["loss"]
 
    return {"loss": total, "per_category": per_category}
 
 
def target_from_batch(batch: Dict, category: str) -> Dict[str, torch.Tensor]:
    """
    Pulls the flattened `{category}_position` / `{category}_tdoa` /
    `{category}_class_id` / `{category}_activity` keys torch_dataset.py's
    SELDDataset produces back into the nested dict pit_loss_single_category
    expects. Purely a naming-convention adapter between the two files.
    """
    return {
        "position": batch[f"{category}_position"],
        "tdoa": batch[f"{category}_tdoa"],
        "class_id": batch[f"{category}_class_id"],
        "activity": batch[f"{category}_activity"],
    }
 
 
def merge_category_targets(
    batch: Dict, categories=("drone", "known_unwanted")
) -> Dict[str, torch.Tensor]:
    """
    Concatenates several categories' targets along the TRACK axis into one
    pool of len(categories) * max_tracks slots, for a model with a single
    set of output heads (model.py's SELDModel): its class_logit -- not a
    separate head per category -- is what tells a drone slot from a
    known_unwanted slot (class ids are global, see class_map.json). Use
    with ModelConfig(max_tracks=len(categories) * dataset_max_tracks):
 
        loss = pit_loss_single_category(model(...), merge_category_targets(batch))["loss"]
 
    pit_loss_multi_category() is the alternative for a model that outputs
    a separate {category: pred_dict} per category.
    """
    parts = [target_from_batch(batch, c) for c in categories]
    return {k: torch.cat([p[k] for p in parts], dim=2) for k in parts[0]}
 
 
if __name__ == "__main__":
    # Self-check with synthetic tensors (no dataset needed): verifies the
    # loss runs end-to-end, is differentiable, and is genuinely
    # permutation-invariant (shuffling the target track order must not
    # change the loss value) -- in the same spirit as
    # validate_gcc_phat.py's standalone sanity checks for the rest of
    # this codebase.
    torch.manual_seed(0)
    B, n_frames, max_tracks, n_pairs, n_classes = 2, 20, 3, 6, 4
 
    pred = {
        "position": torch.randn(B, n_frames, max_tracks, 3, requires_grad=True),
        "activity_logit": torch.randn(B, n_frames, max_tracks, requires_grad=True),
        "class_logit": torch.randn(B, n_frames, max_tracks, n_classes, requires_grad=True),
    }
    target = {
        "position": torch.randn(B, n_frames, max_tracks, 3),
        "activity": torch.randint(0, 2, (B, n_frames, max_tracks)).bool(),
        "class_id": torch.randint(0, n_classes, (B, n_frames, max_tracks)),
    }
 
    result = pit_loss_single_category(pred, target)
    print(f"loss = {result['loss'].item():.4f}")
    result["loss"].backward()
    print(f"grad flows to pred['position']: {pred['position'].grad is not None}")
 
    # permutation-invariance check: shuffle target's track axis, loss must match
    perm = torch.randperm(max_tracks)
    target_shuffled = {k: v[:, :, perm, ...] if v.dim() >= 3 else v for k, v in target.items()}
    result2 = pit_loss_single_category(
        {k: v.detach().requires_grad_(True) if v.requires_grad else v for k, v in pred.items()},
        target_shuffled,
    )
    print(f"loss (shuffled targets) = {result2['loss'].item():.4f}  "
          f"(should match the value above, up to float rounding)")
 
