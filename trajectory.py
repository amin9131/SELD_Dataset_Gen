"""
trajectory.py
=============
Realistic, continuous 3D drone flight paths. A Trajectory spans a whole
"episode" (e.g. 15-40 seconds of continuous flight); the episode is later
sliced into consecutive fixed-length chunks for rendering (see
mixer.render_episode). Because motion is continuous across chunk
boundaries, consecutive output samples can be concatenated to reconstruct
one long, physically continuous ground-truth trajectory -- this is the
"inter-scenario motion strategy" (an episode = many time-consecutive
training clips of one real flight).
 
Every trajectory exposes:
    position_at(t) -> np.ndarray([x, y, z])   for 0 <= t <= duration_s
 
Four realistic flight-pattern presets are provided:
 
  "hover_jitter"    : near-stationary loiter (e.g. photographing a target),
                      small mean-reverting (Ornstein-Uhlenbeck) 3D drift to
                      emulate GPS-hold precision + wind-gust jitter instead
                      of a perfectly frozen point.
 
  "linear_flyby"    : constant-velocity straight-line pass (a common
                      approach/recede or reconnaissance-overflight pattern).
 
  "waypoint_mission": realistic multi-leg flight between random waypoints,
                      each leg using a trapezoidal velocity profile
                      (accelerate -> cruise at v_max -> decelerate), the
                      same shape real autopilots (PX4/ArduPilot mission
                      mode) command -- with optional hover dwell at each
                      waypoint.
 
  "orbit"           : constant-radius circular loiter around a point of
                      interest at fixed altitude -- the classic
                      surveillance/observation pattern.
"""
 
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
 
 
class Trajectory:
    duration_s: float
 
    def position_at(self, t: float) -> np.ndarray:
        raise NotImplementedError
 
    def sample(self, t_start: float, t_end: float, dt: float) -> List[dict]:
        """Convenience: sample [t_start, t_end] at fixed dt, clipped to the
        trajectory's valid range, for building ground-truth metadata."""
        times = np.arange(t_start, t_end, dt)
        out = []
        for t in times:
            t_clamped = min(max(t, 0.0), self.duration_s)
            pos = self.position_at(t_clamped)
            out.append({"t": float(t), "position_xyz_m": pos.tolist()})
        return out
 
 
class StaticTrajectory(Trajectory):
    """Not moving at all (kept for completeness / backward compatibility)."""
 
    def __init__(self, position: Sequence[float], duration_s: float):
        self.position = np.array(position, dtype=np.float64)
        self.duration_s = duration_s
 
    def position_at(self, t: float) -> np.ndarray:
        return self.position.copy()
 
 
class HoverJitterTrajectory(Trajectory):
    """
    Near-stationary loiter with small mean-reverting random drift
    (Ornstein-Uhlenbeck process on velocity, integrated to position),
    which looks like realistic GPS-hold wander rather than a frozen point
    or an unrealistic pure random walk.
    """
 
    def __init__(
        self,
        center: Sequence[float],
        duration_s: float,
        jitter_std_m: float = 0.5,
        time_constant_s: float = 1.5,
        fs_internal: float = 20.0,
        rng: np.random.Generator = None,
    ):
        self.center = np.array(center, dtype=np.float64)
        self.duration_s = duration_s
        rng = rng or np.random.default_rng()
 
        n = max(2, int(np.ceil(duration_s * fs_internal)) + 1)
        dt = 1.0 / fs_internal
        theta = 1.0 / time_constant_s
        sigma = jitter_std_m * np.sqrt(2 * theta)
 
        pos = np.zeros((n, 3))
        vel = np.zeros(3)
        p = np.zeros(3)
        for i in range(1, n):
            noise = rng.standard_normal(3) * sigma * np.sqrt(dt)
            vel = vel - theta * p * dt + noise
            p = p + vel * dt
            pos[i] = p
 
        self._times = np.arange(n) * dt
        self._positions = self.center + pos
 
    def position_at(self, t: float) -> np.ndarray:
        return np.array([
            np.interp(t, self._times, self._positions[:, k]) for k in range(3)
        ])
 
 
class LinearFlybyTrajectory(Trajectory):
    """Constant-velocity straight-line flight for the whole episode."""
 
    def __init__(self, start_pos: Sequence[float], velocity_mps: Sequence[float], duration_s: float):
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.velocity = np.array(velocity_mps, dtype=np.float64)
        self.duration_s = duration_s
 
    def position_at(self, t: float) -> np.ndarray:
        return self.start_pos + self.velocity * t
 
 
def _trapezoidal_arclength(t: float, distance: float, v_max: float, a_max: float) -> float:
    """1D position along a segment of total length `distance`, given a
    trapezoidal (accelerate / cruise / decelerate) velocity profile."""
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc ** 2
 
    if 2 * d_acc >= distance:
        # triangular profile: never reaches v_max
        t_acc = np.sqrt(distance / a_max)
        total_time = 2 * t_acc
        if t <= t_acc:
            return 0.5 * a_max * t ** 2
        t2 = total_time - t
        return distance - 0.5 * a_max * max(t2, 0.0) ** 2
    else:
        d_cruise = distance - 2 * d_acc
        t_cruise = d_cruise / v_max
        total_time = 2 * t_acc + t_cruise
        if t <= t_acc:
            return 0.5 * a_max * t ** 2
        elif t <= t_acc + t_cruise:
            return d_acc + v_max * (t - t_acc)
        else:
            t2 = total_time - t
            return distance - 0.5 * a_max * max(t2, 0.0) ** 2
 
 
def _segment_duration(distance: float, v_max: float, a_max: float) -> float:
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc ** 2
    if 2 * d_acc >= distance:
        return 2 * np.sqrt(distance / a_max)
    d_cruise = distance - 2 * d_acc
    return 2 * t_acc + d_cruise / v_max
 
 
class WaypointMissionTrajectory(Trajectory):
    """
    Realistic multi-leg flight: straight legs between random waypoints,
    each flown with a trapezoidal velocity profile (bounded speed and
    acceleration, as a real flight controller would fly a mission), with
    an optional hover dwell at each waypoint.
    """
 
    def __init__(
        self,
        waypoints: List[Sequence[float]],
        v_max_mps: float,
        a_max_mps2: float,
        hover_dwell_s: List[float] = None,
    ):
        self.waypoints = [np.array(w, dtype=np.float64) for w in waypoints]
        self.v_max = v_max_mps
        self.a_max = a_max_mps2
        n_legs = len(self.waypoints) - 1
        self.hover_dwell_s = hover_dwell_s or [0.0] * (n_legs + 1)
 
        self._leg_durations = []
        self._leg_distances = []
        self._leg_dirs = []
        t_cursor = self.hover_dwell_s[0]
        self._leg_start_times = []
        for i in range(n_legs):
            a, b = self.waypoints[i], self.waypoints[i + 1]
            d = float(np.linalg.norm(b - a))
            direction = (b - a) / d if d > 1e-9 else np.zeros(3)
            leg_dur = _segment_duration(d, self.v_max, self.a_max)
            self._leg_start_times.append(t_cursor)
            self._leg_distances.append(d)
            self._leg_dirs.append(direction)
            self._leg_durations.append(leg_dur)
            t_cursor += leg_dur + self.hover_dwell_s[i + 1]
 
        self.duration_s = t_cursor
 
    def position_at(self, t: float) -> np.ndarray:
        t = min(max(t, 0.0), self.duration_s)
        # before first leg starts: hovering at first waypoint
        if t <= self._leg_start_times[0]:
            return self.waypoints[0].copy()
 
        for i, leg_start in enumerate(self._leg_start_times):
            leg_end = leg_start + self._leg_durations[i]
            if t <= leg_end:
                local_t = t - leg_start
                arc = _trapezoidal_arclength(local_t, self._leg_distances[i], self.v_max, self.a_max)
                return self.waypoints[i] + self._leg_dirs[i] * arc
            # dwell after this leg, before next leg starts
            next_start = (self._leg_start_times[i + 1] if i + 1 < len(self._leg_start_times)
                          else self.duration_s)
            if t <= next_start:
                return self.waypoints[i + 1].copy()
 
        return self.waypoints[-1].copy()
 
 
class OrbitTrajectory(Trajectory):
    """
    Constant-radius circular loiter around a center point at fixed
    altitude -- the classic surveillance/observation flight pattern.
    """
 
    def __init__(
        self,
        center_xy: Sequence[float],
        altitude_m: float,
        radius_m: float,
        tangential_speed_mps: float,
        duration_s: float,
        start_angle_deg: float = 0.0,
        clockwise: bool = False,
    ):
        self.center_xy = np.array(center_xy, dtype=np.float64)
        self.altitude = altitude_m
        self.radius = radius_m
        self.omega = tangential_speed_mps / max(radius_m, 0.1)
        if clockwise:
            self.omega = -self.omega
        self.start_angle = np.deg2rad(start_angle_deg)
        self.duration_s = duration_s
 
    def position_at(self, t: float) -> np.ndarray:
        angle = self.start_angle + self.omega * t
        x = self.center_xy[0] + self.radius * np.cos(angle)
        y = self.center_xy[1] + self.radius * np.sin(angle)
        return np.array([x, y, self.altitude])
 
 
# ---------------------------------------------------------------------------
# Random sampling of a trajectory from config-defined ranges
# ---------------------------------------------------------------------------
 
@dataclass
class MotionConfig:
    """Physically-plausible parameter ranges for small/medium UAV flight."""
    pattern_weights: dict = None  # {"hover_jitter": w, "linear_flyby": w, "waypoint_mission": w, "orbit": w}
    v_max_range_mps: tuple = (3.0, 18.0)
    a_max_range_mps2: tuple = (1.5, 4.0)
    hover_jitter_std_range_m: tuple = (0.2, 1.2)
    orbit_radius_range_m: tuple = (5.0, 30.0)
    n_waypoints_range: tuple = (2, 4)
    waypoint_hover_dwell_range_s: tuple = (0.0, 2.0)
 
    def __post_init__(self):
        if self.pattern_weights is None:
            self.pattern_weights = {
                "hover_jitter": 0.2,
                "linear_flyby": 0.25,
                "waypoint_mission": 0.35,
                "orbit": 0.2,
            }
 
 
def sample_trajectory(
    motion_cfg: MotionConfig,
    duration_s: float,
    region_sampler,   # callable: rng -> np.ndarray([x,y,z]), e.g. PositionRange.sample
    rng: np.random.Generator,
) -> Trajectory:
    """
    Picks one motion pattern (weighted random choice) and instantiates it
    with parameters drawn from `motion_cfg`'s ranges, anchored inside the
    region given by `region_sampler`.
    """
    patterns = list(motion_cfg.pattern_weights.keys())
    weights = np.array([motion_cfg.pattern_weights[p] for p in patterns], dtype=np.float64)
    weights /= weights.sum()
    pattern = rng.choice(patterns, p=weights)
 
    anchor = region_sampler(rng)
 
    if pattern == "hover_jitter":
        jitter_std = rng.uniform(*motion_cfg.hover_jitter_std_range_m)
        return HoverJitterTrajectory(anchor, duration_s, jitter_std_m=jitter_std, rng=rng)
 
    if pattern == "linear_flyby":
        v_max = rng.uniform(*motion_cfg.v_max_range_mps)
        direction = rng.normal(size=3)
        direction[2] *= 0.15  # mostly horizontal flight, gentle climb/descent
        direction /= np.linalg.norm(direction)
        return LinearFlybyTrajectory(anchor, direction * v_max, duration_s)
 
    if pattern == "orbit":
        radius = rng.uniform(*motion_cfg.orbit_radius_range_m)
        v_max = rng.uniform(*motion_cfg.v_max_range_mps)
        return OrbitTrajectory(
            center_xy=anchor[:2], altitude_m=anchor[2], radius_m=radius,
            tangential_speed_mps=v_max, duration_s=duration_s,
            start_angle_deg=float(rng.uniform(0, 360)),
            clockwise=bool(rng.integers(0, 2)),
        )
 
    # waypoint_mission
    n_wp = int(rng.integers(motion_cfg.n_waypoints_range[0], motion_cfg.n_waypoints_range[1] + 1))
    waypoints = [anchor] + [region_sampler(rng) for _ in range(n_wp - 1)]
    v_max = rng.uniform(*motion_cfg.v_max_range_mps)
    a_max = rng.uniform(*motion_cfg.a_max_range_mps2)
    dwell = [rng.uniform(*motion_cfg.waypoint_hover_dwell_range_s) for _ in range(n_wp)]
    traj = WaypointMissionTrajectory(waypoints, v_max, a_max, hover_dwell_s=dwell)
 
    # stretch/loop to exactly fill duration_s if the mission finishes early
    # (common: short missions inside a long episode) by holding position at
    # the final waypoint for the remainder -- StaticTrajectory-style tail
    # is already the natural behavior of position_at() past self.duration_s
    # in WaypointMissionTrajectory (clamped), so we just report the
    # requested episode duration here:
    traj.duration_s = max(traj.duration_s, duration_s)
    return traj
 
