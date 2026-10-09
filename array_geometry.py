"""
array_geometry.py
==================
Customizable microphone array / arrays definitions.
 
A "geometry" is a dict: {mic_id (str) -> (x, y, z) in meters}, in one
global coordinate frame. Includes builders for common layouts plus full
support for arbitrary manual point lists and multiple distributed
sub-arrays (as in the reference paper, which used two 4-mic arrays
offset from the origin).
"""
 
from typing import Dict, List, Sequence, Tuple
import numpy as np
 
MicPositions = Dict[str, np.ndarray]
 
 
def _to_array(points: Dict[str, Sequence[float]]) -> MicPositions:
    return {k: np.array(v, dtype=np.float64) for k, v in points.items()}
 
 
def custom_points(points: Dict[str, Sequence[float]]) -> MicPositions:
    """Fully manual microphone placement: {mic_name: (x, y, z)} in meters."""
    return _to_array(points)
 
 
def linear_array(
    n_mics: int,
    spacing_m: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "x",
    prefix: str = "mic",
) -> MicPositions:
    axis_vec = {"x": np.array([1.0, 0, 0]),
                "y": np.array([0, 1.0, 0]),
                "z": np.array([0, 0, 1.0])}[axis]
    origin = np.array(origin, dtype=np.float64)
    offsets = (np.arange(n_mics) - (n_mics - 1) / 2.0) * spacing_m
    return {f"{prefix}{i+1}": origin + off * axis_vec for i, off in enumerate(offsets)}
 
 
def circular_array(
    n_mics: int,
    radius_m: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    plane: str = "xy",
    start_angle_deg: float = 0.0,
    prefix: str = "mic",
) -> MicPositions:
    origin = np.array(origin, dtype=np.float64)
    angles = np.deg2rad(start_angle_deg + np.linspace(0, 360, n_mics, endpoint=False))
    positions = {}
    for i, a in enumerate(angles):
        c, s = radius_m * np.cos(a), radius_m * np.sin(a)
        if plane == "xy":
            offset = np.array([c, s, 0.0])
        elif plane == "xz":
            offset = np.array([c, 0.0, s])
        elif plane == "yz":
            offset = np.array([0.0, c, s])
        else:
            raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
        positions[f"{prefix}{i+1}"] = origin + offset
    return positions
 
 
def tetrahedral_array(
    radius_m: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    prefix: str = "mic",
) -> MicPositions:
    """
    4-mic arrangement analogous to Figure 1 of the reference paper:
    2 mics for in-plane AOA, 2 mics for elevation.
    """
    origin = np.array(origin, dtype=np.float64)
    base = {
        f"{prefix}1": np.array([radius_m, 0.0, 0.0]),
        f"{prefix}2": np.array([-radius_m, 0.0, 0.0]),
        f"{prefix}3": np.array([0.0, radius_m, 0.0]),
        f"{prefix}4": np.array([0.0, radius_m, radius_m]),
    }
    return {k: origin + v for k, v in base.items()}
 
 
def distributed_sub_arrays(
    sub_array_builder,
    sub_array_kwargs: dict,
    centers: List[Sequence[float]],
    array_prefixes: List[str] = None,
) -> MicPositions:
    """
    Build several identical sub-arrays and merge them into one global
    geometry dict. Generalizes the paper's "two distributed arrays" idea
    to any number of sub-arrays at any locations.
    """
    if array_prefixes is None:
        array_prefixes = [f"arr{chr(65 + i)}" for i in range(len(centers))]
    all_positions: MicPositions = {}
    for center, prefix in zip(centers, array_prefixes):
        sub = sub_array_builder(origin=center, prefix=f"{prefix}_", **sub_array_kwargs)
        all_positions.update(sub)
    return all_positions
 
 
def as_matrix(mic_positions: MicPositions) -> Tuple[List[str], np.ndarray]:
    names = list(mic_positions.keys())
    matrix = np.stack([mic_positions[n] for n in names], axis=0)
    return names, matrix
 
 
def array_centroid(mic_positions: MicPositions) -> np.ndarray:
    _, mat = as_matrix(mic_positions)
    return mat.mean(axis=0)
