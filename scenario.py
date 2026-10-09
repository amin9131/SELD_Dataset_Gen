"""
scenario.py
===========
Samples one random "episode": a continuous block of simulated flight time
(e.g. 15-30 s). Instead of a fixed number of sources present for the WHOLE
episode, each source is a **SourceEvent**: one appearance with its own
random onset time, duration, and continuous flight trajectory spanning
just that appearance. Multiple events (same or different class) may
overlap in time (simultaneous sources) or not (sequential appearances) --
both the NUMBER of simultaneously-active sources and their timing are
therefore variable across the episode, similar to how real-world
polyphonic sound-event datasets (e.g. DCASE SELD) are constructed.
 
An episode is rendered once (mixer.render_episode) as one continuous
multichannel signal, then sliced into consecutive fixed-length chunks --
each chunk's metadata only lists the events that actually overlap that
chunk's time window, so the number of "active" sources per chunk varies
naturally with entries/exits, while consecutive chunks remain literal
contiguous slices of one continuous recording (the "inter-scenario motion
strategy").
"""
 
from dataclasses import dataclass, field
from typing import List
import numpy as np
 
from diffuse_noise import NoiseInjectionConfig
from trajectory import Trajectory, MotionConfig, sample_trajectory
 
 
@dataclass
class PositionRange:
    """Axis-aligned box (meters) OR a spherical-shell region around the
    origin, used to randomly anchor trajectories. Set `use_shell=True` to
    sample a radius/azimuth/elevation range instead of a box."""
    use_shell: bool = True
 
    x_range: tuple = (-15.0, 15.0)
    y_range: tuple = (-15.0, 15.0)
    z_range: tuple = (5.0, 60.0)
 
    radius_range: tuple = (5.0, 60.0)
    azimuth_range_deg: tuple = (0.0, 360.0)
    elevation_range_deg: tuple = (5.0, 80.0)
 
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if not self.use_shell:
            return np.array([
                rng.uniform(*self.x_range),
                rng.uniform(*self.y_range),
                rng.uniform(*self.z_range),
            ])
        r = rng.uniform(*self.radius_range)
        az = np.deg2rad(rng.uniform(*self.azimuth_range_deg))
        el = np.deg2rad(rng.uniform(*self.elevation_range_deg))
        return np.array([
            r * np.cos(el) * np.cos(az),
            r * np.cos(el) * np.sin(az),
            r * np.sin(el),
        ])
 
 
@dataclass
class EventConfig:
    """Controls how many times a source category appears during one
    episode, and how long each appearance lasts."""
    n_events_range: tuple = (1, 3)              # inclusive: total appearances in the episode
    event_duration_s_range: tuple = (4.0, 12.0)  # how long each appearance stays audible
    min_event_duration_s: float = 1.0            # hard floor (also clamped to fit inside the episode)
 
 
@dataclass
class SourceEvent:
    kind: str                # "drone" or "known_unwanted"
    start_s: float           # absolute onset time within the episode
    end_s: float             # absolute offset time within the episode (start_s + duration)
    trajectory: Trajectory    # spans exactly (end_s - start_s) seconds, local time (t=0 at start_s)
 
 
@dataclass
class Episode:
    episode_duration_s: float
    chunk_duration_s: float
    drones: List[SourceEvent] = field(default_factory=list)
    interferers: List[SourceEvent] = field(default_factory=list)
    wind_cfg: NoiseInjectionConfig = field(default_factory=NoiseInjectionConfig)
    ambient_cfg: NoiseInjectionConfig = field(default_factory=NoiseInjectionConfig)
    temperature_c: float = 20.0
    relative_humidity_pct: float = 50.0
    pressure_kpa: float = 101.325
    seed: int = 0
 
 
@dataclass
class EpisodeSamplerConfig:
    episode_duration_s_range: tuple = (15.0, 30.0)
    chunk_duration_s: float = 4.0
 
    drone_event_cfg: EventConfig = field(default_factory=lambda: EventConfig(
        n_events_range=(1, 3), event_duration_s_range=(4.0, 12.0)))
    interferer_event_cfg: EventConfig = field(default_factory=lambda: EventConfig(
        n_events_range=(0, 3), event_duration_s_range=(2.0, 8.0)))
 
    drone_position_range: PositionRange = field(default_factory=PositionRange)
    interferer_position_range: PositionRange = field(default_factory=PositionRange)
 
    drone_motion_cfg: MotionConfig = field(default_factory=MotionConfig)
    interferer_motion_cfg: MotionConfig = field(default_factory=lambda: MotionConfig(
        pattern_weights={"hover_jitter": 0.5, "linear_flyby": 0.5, "waypoint_mission": 0.0, "orbit": 0.0}
    ))
 
    wind_snr_db_range: tuple = (0.0, 20.0)
    wind_mode_choices: tuple = ("diffuse", "directional")
    wind_coherent: bool = True    # spatially-coherent diffuse field (True) vs. legacy
                                    # per-mic-independent noise (False), only affects "diffuse" mode
 
    ambient_snr_db_range: tuple = (5.0, 25.0)
    ambient_mode_choices: tuple = ("diffuse", "directional")
    ambient_coherent: bool = True
 
    temperature_c_range: tuple = (0.0, 35.0)
    relative_humidity_pct_range: tuple = (20.0, 90.0)
    pressure_kpa_range: tuple = (98.0, 103.0)
 
 
def _sample_events(
    kind: str,
    event_cfg: EventConfig,
    episode_duration_s: float,
    motion_cfg: MotionConfig,
    position_range: PositionRange,
    rng: np.random.Generator,
) -> List[SourceEvent]:
    n_events = int(rng.integers(event_cfg.n_events_range[0], event_cfg.n_events_range[1] + 1))
    events = []
    for _ in range(n_events):
        max_dur = min(event_cfg.event_duration_s_range[1], episode_duration_s)
        min_dur = min(event_cfg.event_duration_s_range[0], max_dur)
        min_dur = max(min_dur, min(event_cfg.min_event_duration_s, max_dur))
        duration = float(rng.uniform(min_dur, max_dur)) if max_dur > min_dur else max_dur
 
        latest_start = max(0.0, episode_duration_s - duration)
        start_s = float(rng.uniform(0.0, latest_start)) if latest_start > 0 else 0.0
        end_s = start_s + duration
 
        trajectory = sample_trajectory(motion_cfg, duration, position_range.sample, rng)
        events.append(SourceEvent(kind=kind, start_s=start_s, end_s=end_s, trajectory=trajectory))
 
    return events
 
 
def sample_episode(cfg: EpisodeSamplerConfig, rng: np.random.Generator, seed: int) -> Episode:
    episode_duration_s = float(rng.uniform(*cfg.episode_duration_s_range))
 
    drones = _sample_events(
        "drone", cfg.drone_event_cfg, episode_duration_s,
        cfg.drone_motion_cfg, cfg.drone_position_range, rng,
    )
    interferers = _sample_events(
        "known_unwanted", cfg.interferer_event_cfg, episode_duration_s,
        cfg.interferer_motion_cfg, cfg.interferer_position_range, rng,
    )
 
    wind_cfg = NoiseInjectionConfig(
        mode=str(rng.choice(cfg.wind_mode_choices)),
        snr_db=float(rng.uniform(*cfg.wind_snr_db_range)),
        coherent=cfg.wind_coherent,
    )
    ambient_cfg = NoiseInjectionConfig(
        mode=str(rng.choice(cfg.ambient_mode_choices)),
        snr_db=float(rng.uniform(*cfg.ambient_snr_db_range)),
        coherent=cfg.ambient_coherent,
    )
 
    return Episode(
        episode_duration_s=episode_duration_s,
        chunk_duration_s=cfg.chunk_duration_s,
        drones=drones,
        interferers=interferers,
        wind_cfg=wind_cfg,
        ambient_cfg=ambient_cfg,
        temperature_c=float(rng.uniform(*cfg.temperature_c_range)),
        relative_humidity_pct=float(rng.uniform(*cfg.relative_humidity_pct_range)),
        pressure_kpa=float(rng.uniform(*cfg.pressure_kpa_range)),
        seed=seed,
    )
 
