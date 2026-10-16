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


def infer_mic_groups(mic_names: List[str]) -> List[str]:
    """
    Sub-array membership of every microphone, in the same order as
    `mic_names`. distributed_sub_arrays() names mics "<array>_<mic>"
    (e.g. "arrA_mic1"), so the group is the text before the first "_".
    Names without "_" (linear/circular/tetrahedral single arrays) all
    fall into one group, "all". Used to split microphone pairs into
    INTRA-array pairs (compact, alias-free over the audio band, give an
    unambiguous coarse DOA) and INTER-array pairs (long baseline, give
    precise but potentially ambiguous TDOA -- the range/parallax cue).
    """
    return [n.split("_")[0] if "_" in n else "all" for n in mic_names]


def split_pairs_by_group(mic_names: List[str], groups: List[str] = None, inter_pairs="all"):
    """
    Returns (intra_pairs, inter_pairs_list): lists of (i, j) index pairs.

    inter_pairs selects WHICH cross-array pairs are kept:
      "all" : every mic of one sub-array with every mic of another
      "ref" : 1 pair per sub-array pair (k = 1)
      k     : k pairs per sub-array pair -- the i-th mic of one array with the
              i-th mic of the other, i < k.
    Between two compact sub-arrays all cross pairs measure (almost) the same
    long baseline, so "ref" / small k loses little information while cutting
    the number of GCC-PHAT pairs (and the inter-branch input channels) a lot:
    3 sub-arrays x 4 mics -> 48 cross pairs ("all") vs 3 ("ref") vs 6 (k=2).
    """
    groups = groups or infer_mic_groups(mic_names)
    intra = []
    for a in range(len(mic_names)):
        for b in range(a + 1, len(mic_names)):
            if groups[a] == groups[b]:
                intra.append((a, b))

    if inter_pairs == "all":
        inter = [(a, b) for a in range(len(mic_names)) for b in range(a + 1, len(mic_names))
                 if groups[a] != groups[b]]
        return intra, inter

    k = 1 if inter_pairs == "ref" else int(inter_pairs)
    if k < 1:
        raise ValueError(f"inter_pairs must be 'all', 'ref' or an int >= 1, got {inter_pairs!r}")
    order = list(dict.fromkeys(groups))                    # groups in order of first appearance
    members = {g: [i for i, x in enumerate(groups) if x == g] for g in order}
    inter = []
    for x in range(len(order)):
        for y in range(x + 1, len(order)):
            A, Bm = members[order[x]], members[order[y]]
            for i in range(min(k, len(A), len(Bm))):
                inter.append((A[i], Bm[i]))
    return intra, inter


# ---------------------------------------------------------------------------
# Sub-array placement helpers + config entry point
# ---------------------------------------------------------------------------

def regular_polygon_centers(n: int = 3, side_m: float = 10.0,
                            z_offsets: Sequence[float] = None,
                            start_angle_deg: float = 90.0) -> List[List[float]]:
    """Centers of n sub-arrays on a regular polygon centred on the origin
    (n=3: equilateral triangle) with the given SIDE length. `z_offsets`
    lifts individual arrays (e.g. [0, 0, 3] = one array on a 3 m mast, which
    gives the height (z) parallax a flat layout cannot)."""
    z_offsets = list(z_offsets) if z_offsets is not None else [0.0] * n
    if len(z_offsets) != n:
        raise ValueError("z_offsets must have one entry per sub-array")
    radius = side_m / (2.0 * np.sin(np.pi / n))
    out = []
    for k in range(n):
        a = np.deg2rad(start_angle_deg + 360.0 * k / n)
        out.append([float(radius * np.cos(a)), float(radius * np.sin(a)), float(z_offsets[k])])
    return out


def resolve_centers(centers):
    """`centers` in a config is either an explicit list of [x, y, z], or
    {type: regular_polygon, n: 3, side_m: 10, z_offsets: [0, 0, 3]}."""
    if isinstance(centers, dict):
        if centers.get("type") != "regular_polygon":
            raise ValueError(f"unknown centers spec: {centers}")
        return regular_polygon_centers(centers.get("n", 3), centers["side_m"],
                                       centers.get("z_offsets"), centers.get("start_angle_deg", 90.0))
    return centers


def from_config(array_cfg: dict) -> MicPositions:
    """Builds the microphone geometry from the YAML `array:` block (single
    source of truth, used by dataset_generator.py and array_analysis.py)."""
    kind = array_cfg["type"]
    if kind == "custom_points":
        return custom_points(array_cfg["points"])
    if kind == "linear":
        return linear_array(**array_cfg["params"])
    if kind == "circular":
        return circular_array(**array_cfg["params"])
    if kind == "tetrahedral":
        return tetrahedral_array(**array_cfg["params"])
    if kind == "distributed_sub_arrays":
        builder = {"linear": linear_array, "circular": circular_array,
                   "tetrahedral": tetrahedral_array}[array_cfg["sub_array_type"]]
        return distributed_sub_arrays(
            sub_array_builder=builder,
            sub_array_kwargs=array_cfg["sub_array_params"],
            centers=resolve_centers(array_cfg["centers"]),
            array_prefixes=array_cfg.get("array_prefixes"),
        )
    raise ValueError(f"Unknown array type: {kind}")
