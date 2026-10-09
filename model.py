"""
model.py
========
Neural architecture for joint localization + detection (+ tracking, via
streaming_tracker.py at inference time). Consumes exactly the tensors
torch_dataset.py's SELDDataset produces (logmel, gcc_phat) and produces
exactly the `pred` dict pit_loss.py's pit_loss_single_category /
pit_loss_multi_category expect:
 
    pred["position"]        : (B, n_frames, max_tracks, 3)
    pred["activity_logit"]  : (B, n_frames, max_tracks)
    pred["class_logit"]     : (B, n_frames, max_tracks, n_classes)
 
Architecture
------------
    logmel (B,n_mics,T,n_mels)      gcc_phat (B,n_pairs,T,n_lags)
              |                                |
      [FeatureCNNEncoder]             [FeatureCNNEncoder]     <- 2x causal Conv2d + BN + ReLU + max-pool,
              |                                |                  feature axis reduced to a few bins that
              +---------------- concat --------+                  are KEPT (not averaged away)
                              |
                       [Linear -> d_model]
                              |
                   [Temporal backbone]          <- SWITCHABLE: "crnn" (causal GRU) or
                              |                     "conformer" (causal self-attn + causal
                              |                     depthwise conv), see ModelConfig.backbone
              +---------------+---------------+
              |               |               |
      [activity_head]  [position_head]  [class_head]
      (B,T,max_tracks) (B,T,max_tracks,3) (B,T,max_tracks,n_classes)
 
Both feature branches are geometry-agnostic (same design principle as
feature_extraction.py itself): n_mics and n_pairs are just input CHANNEL
counts to the CNN, so the same architecture works for any array from
array_geometry.py (linear/circular/tetrahedral/custom/distributed)
without modification -- only ModelConfig's dims need to match the
dataset actually used (see `infer_dims_from_batch` below).
 
Causality and streaming
------------------------
`ModelConfig.causal=True` (the default) makes every convolution and
every attention operation causal: output at frame t depends only on
input frames <= t within the SAME chunk. This matters directly for
streaming_tracker.py, which is built around the assumption that each
chunk's prediction is a self-contained, forward-only computation over
that chunk's own audio (no look-ahead into audio the tracker hasn't
received yet).
 
Honest limitation: causal =/= stateful across chunks. Each call to
`model(logmel, gcc_phat)` processes one chunk independently -- there is
NO persistent hidden state carried from chunk N to chunk N+1 inside the
model itself (unlike a true streaming RNN deployment). Cross-chunk
continuity is handled entirely at the TRACK level by
streaming_tracker.py's explicit TrackState (position/velocity
extrapolation), not by the encoder. Adding real cross-chunk hidden-state
passthrough to the temporal backbone (so e.g. the GRU's final hidden
state from chunk N seeds chunk N+1) is a reasonable future extension,
not implemented here -- it would require the training loop to also
process chunks of one episode in strict order with state detached
between optimizer steps, which torch_dataset.py's current shuffled,
independent-chunk sampling does not support (see the SELDDataset
docstring: chunks are treated as independent examples).
"""
 
import math
from dataclasses import dataclass
from typing import Dict, Optional
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
@dataclass
class ModelConfig:
    # --- structural dims: DO NOT guess these, infer from real data ---
    # (see infer_dims_from_batch() below) -- they depend on array geometry
    # (n_mics, n_pairs) and FeatureConfig (n_mels, n_lags), which vary by
    # dataset/config. No universally "sensible" default exists for these.
    n_mics: int
    n_pairs: int
    n_mels: int
    n_lags: int
 
    # --- dataset-dependent but with a reasonable default ---
    # n_classes=2 matches dataset_generator.py's DEFAULT flat-folder setup
    # (no per-class subfolders under assets/drone or assets/known_unwanted):
    # exactly 2 classes, "drone" and "known_unwanted" (see audio_pool.py's
    # `default_class_name` fallback). If you use subfolders for finer-grained
    # classes, override this with len(class_map) from output/class_map.json.
    n_classes: int = 2
 
    # max_tracks=3 matches label_alignment.py / SELDDataset's own default,
    # already used throughout the project -- keep these equal, or your
    # model's output track-slot count won't match what pit_loss.py's
    # `target["position"]` etc. are shaped for.
    # NOTE: SELDModel has ONE set of heads, so with both label categories
    # (drone + known_unwanted) use max_tracks = 2 * dataset_max_tracks (=6)
    # together with pit_loss.merge_category_targets(batch); the class head
    # separates drone slots from interferer slots.
    max_tracks: int = 3
 
    # --- feature-axis downsampling inside each CNN encoder ---
    # The feature axis (mel bins / GCC-PHAT lags) is max-pooled and then
    # reduced to a FIXED number of bins that are kept (flattened into the
    # channel dim), NOT averaged away: for GCC-PHAT the lag at which the
    # peak sits IS the localization information.
    freq_pool: int = 4           # max-pool factor on the feature axis after each conv block
    logmel_freq_bins: int = 4    # bins kept per channel from the log-mel branch
    gcc_freq_bins: int = 21 #32      # bins kept per channel from the GCC-PHAT branch
 
    # --- architecture knobs ---
    backbone: str = "crnn"        # "crnn" or "conformer" -- switchable, see module docstring
    causal: bool = True           # see "Causality and streaming" above
    d_model: int = 128
    cnn_channels: int = 64        # output channels of EACH feature-branch CNN encoder
    n_temporal_layers: int = 2
    n_heads: int = 4              # conformer self-attention heads (ignored for crnn)
    conv_kernel_size: int = 15    # conformer conv-module depthwise kernel (ignored for crnn)
    dropout: float = 0.1
 
    def __post_init__(self):
        if self.backbone not in ("crnn", "conformer"):
            raise ValueError(f"backbone must be 'crnn' or 'conformer', got '{self.backbone}'")
        if self.backbone == "conformer" and self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}) for the conformer backbone"
            )
 
 
def infer_dims_from_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Reads n_mics/n_mels/n_pairs/n_lags directly off a real batch from
    torch_dataset.py's SELDDataset/build_dataloader, instead of guessing.
    Usage:
        batch = next(iter(loader))
        dims = infer_dims_from_batch(batch)
        cfg = ModelConfig(**dims, n_classes=len(class_map), max_tracks=3)
    """
    return {
        "n_mics": batch["logmel"].shape[1],
        "n_mels": batch["logmel"].shape[3],
        "n_pairs": batch["gcc_phat"].shape[1],
        "n_lags": batch["gcc_phat"].shape[3],
    }
 
 
# ---------------------------------------------------------------------------
# Causal Conv2d building block (shared by both feature-branch encoders)
# ---------------------------------------------------------------------------
 
class _TimeCausalConv2d(nn.Module):
    """
    Conv2d over a (T, F) input where T is the (causal-sensitive) time axis
    and F is a plain feature axis (mel bins or GCC-PHAT lag bins -- NOT a
    time axis, so it is always padded symmetrically regardless of
    `causal`). When causal=True, output frame t depends only on input
    frames <= t (left-padding only on the time axis, matching kernel_size-1
    so the T dimension length is preserved).
    """
 
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.kt = kernel_size
        self.pad_f = kernel_size // 2  # symmetric on the feature axis
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=(0, self.pad_f))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        pad_t = self.kt - 1 if self.causal else self.kt // 2
        pad_t_right = 0 if self.causal else self.kt // 2
        x = F.pad(x, (0, 0, pad_t, pad_t_right))  # (left_F, right_F, left_T, right_T)
        return self.conv(x)
 
 
class FeatureCNNEncoder(nn.Module):
    """
    Two causal Conv2d+BN+ReLU blocks over (T, feature_axis). After each
    block the feature axis is max-pooled by `freq_pool`; the result is then
    adaptively average-pooled to a fixed `out_freq_bins` and FLATTENED into
    the channel dimension ((C, bins) -> C*bins) and projected to
    `out_channels` per frame.
 
    Why not average the feature axis away entirely: convolutions are
    translation-equivariant along the feature axis, so a global average over
    it is translation-INVARIANT -- the network could no longer tell where on
    the axis a pattern sits. For GCC-PHAT that position (the lag of the
    correlation peak) is the TDOA itself. Keeping a few spatial bins before
    the projection preserves it (max-pooling also keeps narrow peaks alive).
 
    Input channel count is whatever the caller passes (n_mics for log-mel,
    n_pairs for GCC-PHAT) -- this keeps the whole model array-geometry
    -agnostic (see module docstring). The feature axis must be at least
    freq_pool**2 long.
    """
 
    def __init__(self, in_channels: int, out_channels: int, causal: bool = True,
                 dropout: float = 0.1, freq_pool: int = 4, out_freq_bins: int = 4):
        super().__init__()
        self.conv1 = _TimeCausalConv2d(in_channels, out_channels, kernel_size=3, causal=causal)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = _TimeCausalConv2d(out_channels, out_channels, kernel_size=3, causal=causal)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.freq_pool = nn.MaxPool2d(kernel_size=(1, freq_pool)) if freq_pool > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.adapt = nn.AdaptiveAvgPool2d((None, out_freq_bins))  # fixed number of feature bins
        self.proj = nn.Linear(out_channels * out_freq_bins, out_channels)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, F_in) -> (B, T, C_out)
        x = self.freq_pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.freq_pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adapt(x)                                  # (B, C_out, T, bins)
        B, C, T, Fb = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fb)    # keep the bins: (B, T, C_out*bins)
        return F.relu(self.proj(x))                        # (B, T, C_out)
 
 
# ---------------------------------------------------------------------------
# Temporal backbone, option A: CRNN (causal GRU)
# ---------------------------------------------------------------------------
 
class CRNNBackbone(nn.Module):
    """Causal (unidirectional) multi-layer GRU when `causal=True`; a
    bidirectional GRU (full non-causal context, offline-only) otherwise."""
 
    def __init__(self, d_model: int, n_layers: int, causal: bool = True, dropout: float = 0.1):
        super().__init__()
        self.causal = causal
        bidirectional = not causal
        gru_hidden = d_model // 2 if bidirectional else d_model
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=gru_hidden, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (B, T, d_model) -- bidirectional GRU already concatenates to d_model
        return out
 
 
# ---------------------------------------------------------------------------
# Temporal backbone, option B: simplified causal Conformer
# ---------------------------------------------------------------------------
 
class _PositionalEncoding(nn.Module):
    """Standard sinusoidal absolute positional encoding. Simplification
    note: the original Conformer paper uses RELATIVE positional encoding
    inside attention; this uses simpler absolute encoding added to the
    input instead, which is adequate for chunk-length sequences (a few
    hundred frames) but is a known simplification versus the full
    Conformer design."""
 
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]
 
 
class _CausalDepthwiseConvModule(nn.Module):
    """Conformer's conv module: pointwise -> GLU -> causal depthwise
    conv1d (left-padded only when causal) -> BN -> SiLU -> pointwise."""
 
    def __init__(self, d_model: int, kernel_size: int, causal: bool = True, dropout: float = 0.1):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x.transpose(1, 2)               # (B, d_model, T)
        x = self.pointwise1(x)
        x = F.glu(x, dim=1)                  # (B, d_model, T)
        pad = self.kernel_size - 1 if self.causal else self.kernel_size // 2
        pad_right = 0 if self.causal else self.kernel_size // 2
        x = F.pad(x, (pad, pad_right))
        x = self.depthwise(x)
        x = F.silu(self.bn(x))
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)             # (B, T, d_model)
 
 
class ConformerBlock(nn.Module):
    """Half-step FFN -> causal self-attention -> causal conv module ->
    half-step FFN -> LayerNorm (standard Conformer macro-structure)."""
 
    def __init__(self, d_model: int, n_heads: int, conv_kernel_size: int, causal: bool = True, dropout: float = 0.1):
        super().__init__()
        self.causal = causal
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_module = _CausalDepthwiseConvModule(d_model, conv_kernel_size, causal, dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
 
        attn_mask = None
        if self.causal:
            T = x.shape[1]
            attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
 
        x = x + self.conv_module(self.conv_norm(x))
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)
 
 
class ConformerBackbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, conv_kernel_size: int,
                 causal: bool = True, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = _PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel_size, causal, dropout) for _ in range(n_layers)
        ])
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        return x
 
 
# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
 
class SELDModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
 
        self.logmel_encoder = FeatureCNNEncoder(
            cfg.n_mics, cfg.cnn_channels, cfg.causal, cfg.dropout,
            freq_pool=cfg.freq_pool, out_freq_bins=cfg.logmel_freq_bins)
        self.gcc_encoder = FeatureCNNEncoder(
            cfg.n_pairs, cfg.cnn_channels, cfg.causal, cfg.dropout,
            freq_pool=cfg.freq_pool, out_freq_bins=cfg.gcc_freq_bins)
        self.input_proj = nn.Linear(2 * cfg.cnn_channels, cfg.d_model)
 
        if cfg.backbone == "crnn":
            self.temporal = CRNNBackbone(cfg.d_model, cfg.n_temporal_layers, cfg.causal, cfg.dropout)
        else:  # "conformer", validated in ModelConfig.__post_init__
            self.temporal = ConformerBackbone(
                cfg.d_model, cfg.n_temporal_layers, cfg.n_heads, cfg.conv_kernel_size, cfg.causal, cfg.dropout
            )
 
        self.activity_head = nn.Linear(cfg.d_model, cfg.max_tracks)
        self.position_head = nn.Linear(cfg.d_model, cfg.max_tracks * 3)
        self.class_head = nn.Linear(cfg.d_model, cfg.max_tracks * cfg.n_classes)
 
    def forward(self, logmel: torch.Tensor, gcc_phat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        logmel   : (B, n_mics, n_frames, n_mels)   -- e.g. batch["logmel"] from torch_dataset.py
        gcc_phat : (B, n_pairs, n_frames, n_lags)  -- e.g. batch["gcc_phat"]
 
        Returns a dict with exactly the keys/shapes pit_loss.py's
        pit_loss_single_category expects for `pred`.
        """
        a = self.logmel_encoder(logmel)   # (B, T, cnn_channels)
        b = self.gcc_encoder(gcc_phat)    # (B, T, cnn_channels)
        x = torch.cat([a, b], dim=-1)     # (B, T, 2*cnn_channels)
        x = self.input_proj(x)            # (B, T, d_model)
        x = self.temporal(x)              # (B, T, d_model)
 
        B, T, _ = x.shape
        activity_logit = self.activity_head(x)                                    # (B, T, max_tracks)
        position = self.position_head(x).view(B, T, self.cfg.max_tracks, 3)        # (B, T, max_tracks, 3)
        class_logit = self.class_head(x).view(B, T, self.cfg.max_tracks, self.cfg.n_classes)
 
        return {"position": position, "activity_logit": activity_logit, "class_logit": class_logit}
 
 
def build_model(cfg: ModelConfig) -> SELDModel:
    return SELDModel(cfg)
 
 
if __name__ == "__main__":
    # Self-check with synthetic tensors (no dataset needed), in the same
    # spirit as validate_gcc_phat.py / pit_loss.py / streaming_tracker.py's
    # own standalone checks:
    #   1. output shapes match what pit_loss.py expects
    #   2. gradients flow into both feature encoders
    #   3. CAUSALITY: with causal=True, changing a FUTURE frame's input
    #      must not change an EARLIER frame's output (up to float error) --
    #      this is the property streaming_tracker.py's whole design
    #      depends on.
    torch.manual_seed(0)
    B, T = 2, 40
    n_mics, n_pairs, n_mels, n_lags, n_classes = 4, 6, 64, 129, 2
 
    for backbone in ("crnn", "conformer"):
        print(f"--- backbone={backbone} ---")
        cfg = ModelConfig(
            n_mics=n_mics, n_pairs=n_pairs, n_mels=n_mels, n_lags=n_lags,
            n_classes=n_classes, max_tracks=3, backbone=backbone, causal=True,
            d_model=32, cnn_channels=16, n_temporal_layers=2, n_heads=4,
        )
        model = build_model(cfg)
 
        logmel = torch.randn(B, n_mics, T, n_mels, requires_grad=True)
        gcc = torch.randn(B, n_pairs, T, n_lags, requires_grad=True)
 
        out = model(logmel, gcc)
        print(f"position shape       : {tuple(out['position'].shape)}  (expect {(B, T, cfg.max_tracks, 3)})")
        print(f"activity_logit shape : {tuple(out['activity_logit'].shape)}  (expect {(B, T, cfg.max_tracks)})")
        print(f"class_logit shape    : {tuple(out['class_logit'].shape)}  (expect {(B, T, cfg.max_tracks, cfg.n_classes)})")
 
        out["position"].sum().backward()
        print(f"grad flows to logmel input : {logmel.grad is not None}")
        print(f"grad flows to gcc input    : {gcc.grad is not None}")
 
        # causality check: eval mode (freeze BatchNorm/Dropout randomness),
        # perturb only the LAST frame of the input, verify EARLIER frames
        # of the output are unchanged.
        model.eval()
        with torch.no_grad():
            logmel_a = torch.randn(1, n_mics, T, n_mels)
            gcc_a = torch.randn(1, n_pairs, T, n_lags)
            out_a = model(logmel_a, gcc_a)
 
            logmel_b = logmel_a.clone()
            gcc_b = gcc_a.clone()
            logmel_b[:, :, -1, :] += 5.0   # perturb only the LAST (future-most) frame
            gcc_b[:, :, -1, :] += 5.0
            out_b = model(logmel_b, gcc_b)
 
            earlier_diff = (out_a["position"][:, :-1] - out_b["position"][:, :-1]).abs().max().item()
            print(f"max diff in EARLIER frames after perturbing the LAST frame: {earlier_diff:.2e}  "
                  f"({'PASS' if earlier_diff < 1e-5 else 'FAIL'} -- should be ~0 for a causal model)")
        model.train()
        print()
 
