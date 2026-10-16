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
    # (n_mics, pair counts) and FeatureConfig (n_mels, lag counts).
    n_mics: int
    n_mels: int

    # GCC-PHAT input -- exactly ONE of the two layouts:
    #  (a) legacy, FeatureConfig.pair_mode="all": one tensor batch["gcc_phat"]
    n_pairs: int = 0
    n_lags: int = 0
    #  (b) grouped, pair_mode="grouped": batch["gcc_phat_intra"] (pairs inside one
    #      sub-array: few lags, sub-sample interpolated) and batch["gcc_phat_inter"]
    #      (pairs across sub-arrays: long baseline, many lags). Each gets its own
    #      encoder. A layout without inter pairs (single array) leaves *_inter = 0.
    n_pairs_intra: int = 0
    n_lags_intra: int = 0
    n_pairs_inter: int = 0
    n_lags_inter: int = 0

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
    gcc_freq_bins: int = 32      # bins kept per channel from the GCC-PHAT branch (legacy layout)
    # grouped layout: the INTRA lag axis is short (~100-150 interpolated lags), so
    # it is pooled less; the INTER lag axis is long (~350-700 lags), pooled like legacy.
    gcc_intra_freq_pool: int = 2
    gcc_intra_freq_bins: int = 32
    gcc_inter_freq_pool: int = 4
    gcc_inter_freq_bins: int = 32
 
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
        legacy = self.n_pairs > 0
        grouped = (self.n_pairs_intra > 0) or (self.n_pairs_inter > 0)
        if legacy == grouped:
            raise ValueError(
                "ModelConfig needs exactly one GCC layout: n_pairs/n_lags (legacy 'gcc_phat') "
                "OR n_pairs_intra/n_lags_intra and/or n_pairs_inter/n_lags_inter (grouped). "
                "Use infer_dims_from_batch(batch) to fill them."
            )
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
    dims = {"n_mics": batch["logmel"].shape[1], "n_mels": batch["logmel"].shape[3]}
    if "gcc_phat" in batch:
        dims.update(n_pairs=batch["gcc_phat"].shape[1], n_lags=batch["gcc_phat"].shape[3])
    if "gcc_phat_intra" in batch:
        dims.update(n_pairs_intra=batch["gcc_phat_intra"].shape[1],
                    n_lags_intra=batch["gcc_phat_intra"].shape[3])
    if "gcc_phat_inter" in batch:
        dims.update(n_pairs_inter=batch["gcc_phat_inter"].shape[1],
                    n_lags_inter=batch["gcc_phat_inter"].shape[3])
    return dims


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
        self.legacy_gcc = cfg.n_pairs > 0
        self.gcc_encoder = None
        self.gcc_intra_encoder = None
        self.gcc_inter_encoder = None
        if self.legacy_gcc:
            self.gcc_encoder = FeatureCNNEncoder(
                cfg.n_pairs, cfg.cnn_channels, cfg.causal, cfg.dropout,
                freq_pool=cfg.freq_pool, out_freq_bins=cfg.gcc_freq_bins)
        else:
            if cfg.n_pairs_intra > 0:
                self.gcc_intra_encoder = FeatureCNNEncoder(
                    cfg.n_pairs_intra, cfg.cnn_channels, cfg.causal, cfg.dropout,
                    freq_pool=cfg.gcc_intra_freq_pool, out_freq_bins=cfg.gcc_intra_freq_bins)
            if cfg.n_pairs_inter > 0:
                self.gcc_inter_encoder = FeatureCNNEncoder(
                    cfg.n_pairs_inter, cfg.cnn_channels, cfg.causal, cfg.dropout,
                    freq_pool=cfg.gcc_inter_freq_pool, out_freq_bins=cfg.gcc_inter_freq_bins)
        n_branches = 1 + sum(e is not None for e in
                             (self.gcc_encoder, self.gcc_intra_encoder, self.gcc_inter_encoder))
        self.input_proj = nn.Linear(n_branches * cfg.cnn_channels, cfg.d_model)
 
        if cfg.backbone == "crnn":
            self.temporal = CRNNBackbone(cfg.d_model, cfg.n_temporal_layers, cfg.causal, cfg.dropout)
        else:  # "conformer", validated in ModelConfig.__post_init__
            self.temporal = ConformerBackbone(
                cfg.d_model, cfg.n_temporal_layers, cfg.n_heads, cfg.conv_kernel_size, cfg.causal, cfg.dropout
            )
 
        self.activity_head = nn.Linear(cfg.d_model, cfg.max_tracks)
        self.position_head = nn.Linear(cfg.d_model, cfg.max_tracks * 3)
        self.class_head = nn.Linear(cfg.d_model, cfg.max_tracks * cfg.n_classes)
 
    def forward(
        self,
        logmel: torch.Tensor,
        gcc_phat: Optional[torch.Tensor] = None,
        gcc_phat_intra: Optional[torch.Tensor] = None,
        gcc_phat_inter: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        logmel         : (B, n_mics, n_frames, n_mels)
        gcc_phat       : (B, n_pairs, n_frames, n_lags)        -- legacy layout only
        gcc_phat_intra : (B, n_pairs_intra, n_frames, n_lags_i) -- grouped layout
        gcc_phat_inter : (B, n_pairs_inter, n_frames, n_lags_o) -- grouped layout
        (or simply model.forward_batch(batch)). Returns a dict with exactly the
        keys/shapes pit_loss.py's pit_loss_single_category expects for `pred`.
        """
        parts = [self.logmel_encoder(logmel)]               # (B, T, cnn_channels)
        for enc, feat, name in ((self.gcc_encoder, gcc_phat, "gcc_phat"),
                                (self.gcc_intra_encoder, gcc_phat_intra, "gcc_phat_intra"),
                                (self.gcc_inter_encoder, gcc_phat_inter, "gcc_phat_inter")):
            if enc is None:
                continue
            if feat is None:
                raise ValueError(f"this model was built with a '{name}' branch but no '{name}' tensor was passed")
            parts.append(enc(feat))
        x = torch.cat(parts, dim=-1)      # (B, T, n_branches*cnn_channels)
        x = self.input_proj(x)            # (B, T, d_model)
        x = self.temporal(x)              # (B, T, d_model)

        B, T, _ = x.shape
        activity_logit = self.activity_head(x)                                    # (B, T, max_tracks)
        position = self.position_head(x).view(B, T, self.cfg.max_tracks, 3)        # (B, T, max_tracks, 3)
        class_logit = self.class_head(x).view(B, T, self.cfg.max_tracks, self.cfg.n_classes)
 
        return {"position": position, "activity_logit": activity_logit, "class_logit": class_logit}

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convenience for either layout: model.forward_batch(batch)."""
        return self(batch["logmel"], batch.get("gcc_phat"),
                    batch.get("gcc_phat_intra"), batch.get("gcc_phat_inter"))
 
 
def build_model(cfg: ModelConfig) -> SELDModel:
    return SELDModel(cfg)
 
 
if __name__ == "__main__":
    # Self-check with synthetic tensors (no dataset needed):
    #   1. output shapes match what pit_loss.py expects
    #   2. gradients flow into every feature encoder
    #   3. CAUSALITY: with causal=True, changing a FUTURE frame's input must not
    #      change an EARLIER frame's output -- the property streaming_tracker.py
    #      depends on.
    # Runs for both GCC layouts (legacy single tensor / grouped intra+inter).
    torch.manual_seed(0)
    B, T = 2, 40
    n_mics, n_mels, n_classes = 8, 64, 2
    layouts = {
        "legacy":  dict(n_pairs=28, n_lags=129),
        "grouped": dict(n_pairs_intra=12, n_lags_intra=109, n_pairs_inter=16, n_lags_inter=351),
    }

    def make_inputs(dims, requires_grad=False):
        x = {"logmel": torch.randn(B, n_mics, T, n_mels, requires_grad=requires_grad)}
        if "n_pairs" in dims:
            x["gcc_phat"] = torch.randn(B, dims["n_pairs"], T, dims["n_lags"], requires_grad=requires_grad)
        if "n_pairs_intra" in dims:
            x["gcc_phat_intra"] = torch.randn(B, dims["n_pairs_intra"], T, dims["n_lags_intra"], requires_grad=requires_grad)
            x["gcc_phat_inter"] = torch.randn(B, dims["n_pairs_inter"], T, dims["n_lags_inter"], requires_grad=requires_grad)
        return x

    for layout, dims in layouts.items():
        for backbone in ("crnn", "conformer"):
            print(f"--- layout={layout}, backbone={backbone} ---")
            cfg = ModelConfig(
                n_mics=n_mics, n_mels=n_mels, **dims,
                n_classes=n_classes, max_tracks=3, backbone=backbone, causal=True,
                d_model=32, cnn_channels=16, n_temporal_layers=2, n_heads=4,
            )
            model = build_model(cfg)
            inp = make_inputs(dims, requires_grad=True)
            assert infer_dims_from_batch(inp) == {"n_mics": n_mics, "n_mels": n_mels, **dims}
            out = model.forward_batch(inp)
            print(f"position shape       : {tuple(out['position'].shape)}  (expect {(B, T, cfg.max_tracks, 3)})")
            print(f"activity_logit shape : {tuple(out['activity_logit'].shape)}  (expect {(B, T, cfg.max_tracks)})")
            print(f"class_logit shape    : {tuple(out['class_logit'].shape)}  (expect {(B, T, cfg.max_tracks, cfg.n_classes)})")
            out["position"].sum().backward()
            for k, v in inp.items():
                print(f"grad flows to {k:15s}: {v.grad is not None}")

            model.eval()
            with torch.no_grad():
                a = make_inputs(dims)
                out_a = model.forward_batch(a)
                b = {k: v.clone() for k, v in a.items()}
                for v in b.values():
                    v[:, :, -1, :] += 5.0       # perturb only the LAST frame
                out_b = model.forward_batch(b)
                diff = (out_a["position"][:, :-1] - out_b["position"][:, :-1]).abs().max().item()
                print(f"max diff in EARLIER frames after perturbing the LAST frame: {diff:.2e}  "
                      f"({'PASS' if diff < 1e-5 else 'FAIL'} -- should be ~0 for a causal model)")
            print()
