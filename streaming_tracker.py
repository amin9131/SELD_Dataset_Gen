"""
streaming_tracker.py
=====================
Real-time, class-selective multi-target tracker that threads a model's
PER-CHUNK detections into persistent tracks ACROSS chunks.
 
This is an INFERENCE-time component (not training-time -- for training,
see pit_loss.py, which matches predicted/label tracks independently
within one chunk). This module answers a different question: "is this
drone I'm seeing in chunk N the same drone I saw in chunk N-1?"
 
Why this can't just be "a bigger RNN"
--------------------------------------
A model with a recurrent/temporal encoder (CRNN, Conformer, ...) already
gets *implicit* temporal context within its receptive field. But it does
NOT give you:
  - a stable, explicit track ID you can log/display/hand to a downstream
    consumer ("Drone #3 has been tracked for 40s at bearing 220 deg")
  - a principled way to survive a multi-chunk gap in detection (e.g. the
    classifier drops the track for one chunk due to noise) without either
    losing the track entirely or silently stitching it to an unrelated
    detection
  - class-selective computation: skipping tracking work entirely for
    off-target classes (e.g. bird/car), not just down-weighting them
 
This module provides that explicit layer, in the same spirit as
classical online multi-object tracking (SORT-style: constant-velocity
prediction + Hungarian matching + track birth/death with hit/miss
counters), adapted to this framework's chunk-based, class-labeled,
multi-track output format (see pit_loss.py's `pred` dict format, which
this module consumes one chunk / one batch element at a time).
 
Pipeline per chunk
-------------------
  1. CLASS-GATE: reduce each predicted track slot's frame-wise output to
     at most one "detection" for this chunk, and DROP it immediately if
     its majority-voted class isn't in `TrackerConfig.target_class_ids`.
     Off-target detections never reach the matching step below --
     this is the "class-aware localization/tracking" idea: only spend
     tracking computation and track slots on classes you actually care
     about (e.g. drones only).
  2. EXTRAPOLATE every existing track's position to "now" using its
     current constant-velocity estimate (classic Kalman/SORT-style
     motion prediction, kept deliberately simple -- a full Kalman filter
     would let you also propagate position UNCERTAINTY, which this
     module does not model; see "Known limitations" below).
  3. MATCH extrapolated track positions against this chunk's detections
     via Hungarian assignment on Euclidean distance, gated by
     `max_match_distance_m` (a match farther than this is treated as no
     match at all, not just a bad match).
  4. UPDATE matched tracks (position, EMA-smoothed velocity, majority-vote
     class buffer -- so a single misclassified chunk doesn't flip the
     track's reported class), age/kill unmatched tracks past
     `max_misses_before_death`, and spawn tentative tracks for unmatched
     detections (confirmed only after `min_hits_to_confirm` consecutive
     hits, to avoid single-frame false positives becoming visible tracks).
 
Known limitations (stated plainly, not hidden)
------------------------------------------------
- Constant-velocity extrapolation only -- no acceleration model, no
  uncertainty propagation. Fine for the framework's drone motion patterns
  at chunk-to-chunk timescales (a few seconds), but a real Kalman filter
  with a proper process/measurement noise model would extrapolate more
  gracefully through longer gaps or sharper maneuvers (e.g. orbit
  trajectories -- see trajectory.py's OrbitTrajectory).
- ONE detection per predicted track slot per chunk (the chunk's last
  active frame's position). This discards intra-chunk trajectory detail
  that IS available in `pred["position"]` -- a reasonable simplification
  for chunk-to-chunk tracking, but not for sub-chunk (frame-level)
  tracking.
- No re-identification: once a track is killed (too many consecutive
  misses), a source reappearing later gets a NEW track ID, never
  automatically linked back to the old one.
"""
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
 
import numpy as np
from scipy.optimize import linear_sum_assignment
 
 
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
 
 
@dataclass
class TrackState:
    track_id: int
    position: np.ndarray            # (3,) last known/updated position, meters
    velocity: np.ndarray            # (3,) EMA-smoothed velocity estimate, m/s
    class_id: int
    class_votes: Dict[int, int] = field(default_factory=dict)  # running majority-vote buffer
    hits: int = 1                    # consecutive successful chunk-matches (resets on a miss)
    misses: int = 0                  # consecutive missed chunks
    confirmed: bool = False          # True once hits >= TrackerConfig.min_hits_to_confirm
    last_update_time_s: float = 0.0  # absolute time this track's position was last a real detection
 
    def predict(self, t_now_s: float) -> np.ndarray:
        """Constant-velocity extrapolation to absolute time `t_now_s`."""
        dt = t_now_s - self.last_update_time_s
        return self.position + self.velocity * dt
 
    def _vote_class(self, class_id: int) -> None:
        self.class_votes[class_id] = self.class_votes.get(class_id, 0) + 1
        self.class_id = max(self.class_votes, key=self.class_votes.get)
 
 
@dataclass
class TrackerConfig:
    target_class_ids: Optional[Set[int]] = None   # None = track every class; else ONLY these
                                                     # class_ids are ever tracked (class-gate)
    activity_threshold: float = 0.5     # sigmoid(activity_logit) threshold for "this frame is active"
    class_conf_threshold: float = 0.0   # optional softmax-confidence gate on class_logit (0 = disabled)
    max_match_distance_m: float = 15.0  # matching gate: farther than this = not a match at all
    min_hits_to_confirm: int = 2        # consecutive chunk-hits before a track is reported as "confirmed"
    max_misses_before_death: int = 3    # consecutive missed chunks before a track is dropped
    velocity_smoothing: float = 0.5     # EMA factor for velocity updates: 0=freeze old estimate, 1=instant
 
 
@dataclass
class _Detection:
    track_slot: int          # which predicted-track-slot index this came from (for debugging only)
    position: np.ndarray      # (3,)
    velocity_hint: np.ndarray  # (3,) finite-difference velocity estimate WITHIN this chunk
    class_id: int
    time_s: float             # absolute time of the frame `position` was taken from
 
 
class StreamingTracker:
    """
    Call `update(pred, chunk_start_time_s, frame_dt_s)` once per chunk, in
    strict time order, for ONE stream (batch size 1 -- this is an
    inference-time component; pit_loss.py handles the batched
    training-time matching within a single chunk).
 
    `pred` is the same per-CHUNK, per-BATCH-ELEMENT prediction dict
    pit_loss.py's `pit_loss_single_category` expects, WITHOUT the batch
    axis:
        pred["position"]       : (n_frames, n_pred_tracks, 3)
        pred["activity_logit"] : (n_frames, n_pred_tracks)
        pred["class_logit"]    : (n_frames, n_pred_tracks, n_classes), optional
                                  (required if TrackerConfig.target_class_ids is set)
    """
 
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.tracks: Dict[int, TrackState] = {}
        self._next_id = 0
 
    def active_tracks(self) -> List[TrackState]:
        """Currently CONFIRMED, alive tracks -- what you'd display/report."""
        return [t for t in self.tracks.values() if t.confirmed]
 
    def all_tracks(self) -> List[TrackState]:
        """Every live track, including unconfirmed/tentative ones (for debugging)."""
        return list(self.tracks.values())
 
    # -----------------------------------------------------------------
    # Step 1: reduce one chunk's frame-wise predictions to <=1 detection
    # per predicted track slot, applying the class-gate.
    # -----------------------------------------------------------------
    def _extract_chunk_detections(
        self, pred: Dict[str, np.ndarray], chunk_start_time_s: float, frame_dt_s: float
    ) -> List[_Detection]:
        position = pred["position"]              # (n_frames, n_pred_tracks, 3)
        activity_logit = pred["activity_logit"]   # (n_frames, n_pred_tracks)
        class_logit = pred.get("class_logit")     # (n_frames, n_pred_tracks, n_classes) or None
 
        n_frames, n_pred_tracks = activity_logit.shape
        active_mask = _sigmoid(activity_logit) > self.cfg.activity_threshold  # (n_frames, n_pred_tracks)
 
        detections: List[_Detection] = []
        for p in range(n_pred_tracks):
            active_frames = np.where(active_mask[:, p])[0]
            if len(active_frames) == 0:
                continue  # this slot has no detection at all this chunk
 
            # class: majority vote over active frames (optionally confidence-gated)
            class_id = -1
            if class_logit is not None:
                logits = class_logit[active_frames, p, :]           # (n_active, n_classes)
                probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True)
                conf = probs.max(axis=-1)
                pred_class = probs.argmax(axis=-1)
                if self.cfg.class_conf_threshold > 0:
                    keep = conf >= self.cfg.class_conf_threshold
                    if not keep.any():
                        continue  # nothing confident enough this chunk -> no detection
                    pred_class = pred_class[keep]
                votes = np.bincount(pred_class)
                class_id = int(votes.argmax())
 
            # CLASS-GATE: drop this detection entirely if it isn't a target class.
            # This is what keeps tracking computation "class-selective" -- an
            # off-target detection (e.g. a bird) never reaches matching.
            if self.cfg.target_class_ids is not None and class_id not in self.cfg.target_class_ids:
                continue
 
            last_frame = active_frames[-1]
            pos = position[last_frame, p, :]
 
            # within-chunk velocity hint: finite difference between the first
            # and last active frame, if there are at least two -- gives the
            # new-track birth case a non-zero initial velocity instead of 0.
            if len(active_frames) >= 2:
                first_frame = active_frames[0]
                dt = (last_frame - first_frame) * frame_dt_s
                vel_hint = (position[last_frame, p, :] - position[first_frame, p, :]) / max(dt, 1e-6)
            else:
                vel_hint = np.zeros(3)
 
            detections.append(_Detection(
                track_slot=p, position=pos, velocity_hint=vel_hint, class_id=class_id,
                time_s=chunk_start_time_s + float(last_frame) * frame_dt_s,
            ))
 
        return detections
 
    # -----------------------------------------------------------------
    # Step 2-4: extrapolate, match, update/age/spawn.
    # -----------------------------------------------------------------
    def update(
        self, pred: Dict[str, np.ndarray], chunk_start_time_s: float, frame_dt_s: float
    ) -> List[TrackState]:
        """
        Processes one chunk's predictions and returns `active_tracks()`
        afterward. `chunk_start_time_s` is this chunk's absolute start
        time (e.g. running episode/wall-clock time, incremented by
        chunk_duration_s between calls -- caller's responsibility to keep
        this monotonically increasing and consistent with `frame_dt_s`,
        the audio-frame spacing = FeatureConfig.hop_length / fs).
        """
        detections = self._extract_chunk_detections(pred, chunk_start_time_s, frame_dt_s)
 
        track_ids = list(self.tracks.keys())
 
        matched_track_ids: set = set()
        matched_det_idx: set = set()
 
        if track_ids and detections:
            cost = np.full((len(track_ids), len(detections)), fill_value=1e6)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    # extrapolate this track to THIS detection's own timestamp
                    # (a detection that ends mid-chunk is not at the chunk's end)
                    dist = float(np.linalg.norm(self.tracks[tid].predict(det.time_s) - det.position))
                    if dist <= self.cfg.max_match_distance_m:
                        cost[i, j] = dist
 
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if cost[i, j] >= 1e6:
                    continue  # gated out -- not a real match
                tid = track_ids[i]
                det = detections[j]
                self._apply_match(tid, det, frame_dt_s)
                matched_track_ids.add(tid)
                matched_det_idx.add(j)
 
        # age/kill unmatched existing tracks
        for tid in track_ids:
            if tid in matched_track_ids:
                continue
            track = self.tracks[tid]
            # NOTE: position / last_update_time_s deliberately stay at the last
            # REAL detection; predict() extrapolates from there. (Overwriting
            # position with the extrapolated value while keeping the old
            # last_update_time_s double-counted the elapsed motion on the next
            # predict().)
            track.misses += 1
            track.hits = 0
            if track.misses > self.cfg.max_misses_before_death:
                del self.tracks[tid]
 
        # spawn new tentative tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in matched_det_idx:
                continue
            self._spawn_track(det)
 
        return self.active_tracks()
 
    def _apply_match(self, tid: int, det: "_Detection", frame_dt_s: float) -> None:
        track = self.tracks[tid]
        dt = max(det.time_s - track.last_update_time_s, frame_dt_s)
        vel_measured = (det.position - track.position) / dt
        alpha = self.cfg.velocity_smoothing
        track.velocity = (1 - alpha) * track.velocity + alpha * vel_measured
        track.position = det.position
        if det.class_id >= 0:
            track._vote_class(det.class_id)
        track.hits += 1
        track.misses = 0
        track.last_update_time_s = det.time_s
        if track.hits >= self.cfg.min_hits_to_confirm:
            track.confirmed = True
 
    def _spawn_track(self, det: "_Detection") -> None:
        tid = self._next_id
        self._next_id += 1
        track = TrackState(
            track_id=tid,
            position=det.position.copy(),
            velocity=det.velocity_hint.copy(),
            class_id=det.class_id,
            hits=1,
            misses=0,
            confirmed=self.cfg.min_hits_to_confirm <= 1,
            last_update_time_s=det.time_s,
        )
        if det.class_id >= 0:
            track._vote_class(det.class_id)
        self.tracks[tid] = track
 
 
if __name__ == "__main__":
    # Self-check with synthetic per-chunk predictions (no dataset needed),
    # in the same spirit as validate_gcc_phat.py / pit_loss.py's own
    # standalone checks. Simulates:
    #   - one drone (class_id=0) moving at constant velocity across 4 chunks
    #   - one bird (class_id=1) present the whole time, which the tracker
    #     must NEVER produce a track for once target_class_ids={0}
    #   - a one-chunk dropout of the drone detection (tests miss-survival)
    n_frames, n_pred_tracks, n_classes = 10, 2, 2
    frame_dt_s = 0.01
    chunk_duration_s = n_frames * frame_dt_s
 
    cfg = TrackerConfig(target_class_ids={0}, min_hits_to_confirm=2, max_misses_before_death=2)
    tracker = StreamingTracker(cfg)
 
    drone_start_pos = np.array([10.0, 0.0, 20.0])
    drone_velocity = np.array([2.0, 0.0, 0.0])  # m/s
    bird_pos = np.array([-5.0, 5.0, 8.0])
 
    def make_chunk_pred(chunk_idx: int, drone_present: bool):
        position = np.zeros((n_frames, n_pred_tracks, 3))
        activity_logit = np.full((n_frames, n_pred_tracks), -10.0)  # inactive by default
        class_logit = np.zeros((n_frames, n_pred_tracks, n_classes))
 
        t0 = chunk_idx * chunk_duration_s
        for f in range(n_frames):
            t = t0 + f * frame_dt_s
            # slot 0: drone (if present this chunk)
            if drone_present:
                position[f, 0, :] = drone_start_pos + drone_velocity * t
                activity_logit[f, 0] = 10.0
                class_logit[f, 0, 0] = 10.0  # confidently class 0
 
            # slot 1: bird, always present, must be filtered by class-gate
            position[f, 1, :] = bird_pos
            activity_logit[f, 1] = 10.0
            class_logit[f, 1, 1] = 10.0  # confidently class 1
 
        return {"position": position, "activity_logit": activity_logit, "class_logit": class_logit}
 
    presence = [True, True, False, True, True]  # chunk 2 = dropout
    track_id_over_time = []
    for chunk_idx, present in enumerate(presence):
        pred = make_chunk_pred(chunk_idx, present)
        active = tracker.update(pred, chunk_start_time_s=chunk_idx * chunk_duration_s, frame_dt_s=frame_dt_s)
        ids = sorted(t.track_id for t in active)
        track_id_over_time.append(ids)
        print(f"chunk {chunk_idx} (drone_present={present}): active track ids = {ids}, "
              f"n_tentative_total={len(tracker.all_tracks())}")
 
    print()
    print("Bird (class 1) never produced a track:",
          all(t.class_id != 1 for t in tracker.all_tracks()))
    print("Drone track ID stayed the same across the dropout at chunk 2:",
          len(set(sum(track_id_over_time, []))) <= 1)
 
