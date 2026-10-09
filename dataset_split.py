"""
dataset_split.py
=================
Automatic, leakage-free train/val/test splitting.
 
The split happens at the level of individual AUDIO FILES (never
episodes), stratified per source class (the same class taxonomy
AudioPool derives from subfolders -- see audio_pool.py). Each class's
files are independently shuffled (with a fixed, reproducible seed) and
divided proportionally across splits, so train/val/test each get a
representative share of every class, and NO audio file -- and therefore
no specific recording's characteristic noise floor, room tone, or
recording-chain artifact -- is ever shared across splits.
 
This matters because a model can otherwise learn to recognize a
*specific recording* (its exact background hiss, mic self-noise, or
compression artifacts) rather than the general acoustic phenomenon it's
supposed to represent. That kind of session-level leakage inflates
validation/test metrics without the model actually generalizing -- a
well-known failure mode in audio ML that plain random splitting at the
*episode* level would NOT catch, since the same underlying files could
still end up reused across splits.
"""
 
from typing import Dict, List
import random
 
from audio_pool import AudioPool
 
 
def _split_file_list(files: List[str], ratios: Dict[str, float], rng: random.Random) -> Dict[str, List[str]]:
    """
    Largest-remainder proportional allocation of `files` across the named
    splits in `ratios` (which need not sum to exactly 1.0 -- they are
    normalized here). Deterministic given `rng`'s state.
    """
    files = files[:]
    rng.shuffle(files)
    n = len(files)
 
    names = list(ratios.keys())
    total = sum(ratios.values())
    fracs = [ratios[name] / total for name in names]
 
    raw = [f * n for f in fracs]
    counts = [int(r) for r in raw]  # floor
    remainder = n - sum(counts)
 
    # Give the leftover files to the splits with the largest fractional
    # remainder first (standard "largest remainder" rounding) -- keeps
    # small per-class file counts from silently collapsing into one split.
    order = sorted(range(len(names)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(names)]] += 1
 
    result = {}
    idx = 0
    for name, c in zip(names, counts):
        result[name] = files[idx: idx + c]
        idx += c
    return result
 
 
def split_pool_by_class(
    pool: AudioPool,
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, AudioPool]:
    """
    Splits `pool` into new AudioPool instances (one per split name in
    `ratios`), stratified so every class present in `pool` contributes
    its proportional share of files to every split -- as long as it has
    enough files to do so; classes with very few files may end up
    entirely in one split (reported via `describe_split`, so this is
    never silent).
    """
    rng = random.Random(seed)
 
    by_class: Dict[str, List[str]] = {}
    for f in pool.files:
        by_class.setdefault(pool._class_name_for(f), []).append(f)
 
    per_split_files: Dict[str, List[str]] = {name: [] for name in ratios}
    for class_name in sorted(by_class):
        allocation = _split_file_list(by_class[class_name], ratios, rng)
        for name, split_files in allocation.items():
            per_split_files[name].extend(split_files)
 
    return {
        name: AudioPool.from_file_list(files, pool.folder, pool.target_fs, pool.default_class_name)
        for name, files in per_split_files.items()
    }
 
 
def describe_split(pool: AudioPool, split_pools: Dict[str, AudioPool]) -> dict:
    """
    Human/JSON-readable summary of exactly which files (grouped by class)
    went to which split -- written to split_manifest.json for
    transparency and so a leakage audit is a one-line grep away, plus a
    warning list for any class that ended up entirely absent from one or
    more splits (too few source files to divide up).
    """
    by_class_total: Dict[str, int] = {}
    for f in pool.files:
        by_class_total[pool._class_name_for(f)] = by_class_total.get(pool._class_name_for(f), 0) + 1
 
    summary = {
        "folder": pool.folder,
        "total_files": len(pool.files),
        "files_per_class": by_class_total,
        "splits": {},
        "warnings": [],
    }
 
    per_split_class_counts: Dict[str, Dict[str, int]] = {}
    for name, sp in split_pools.items():
        files_by_class: Dict[str, List[str]] = {}
        for f in sp.files:
            files_by_class.setdefault(sp._class_name_for(f), []).append(f)
        summary["splits"][name] = {
            "n_files": len(sp.files),
            "files_by_class": files_by_class,
        }
        per_split_class_counts[name] = {c: len(v) for c, v in files_by_class.items()}
 
    for class_name in by_class_total:
        missing = [name for name in split_pools if per_split_class_counts[name].get(class_name, 0) == 0]
        if missing:
            summary["warnings"].append(
                f"class '{class_name}' in '{pool.folder}' has 0 files in split(s): {missing} "
                f"(only {by_class_total[class_name]} file(s) total for this class -- "
                f"add more recordings if you need every split to cover it)"
            )
 
    return summary
 
 
def assert_no_overlap(split_pools: Dict[str, AudioPool]) -> None:
    """Hard safety check: raises if any file appears in more than one
    split (should be structurally impossible given how splitting works,
    but this is cheap insurance against future refactors introducing a
    bug here)."""
    seen: Dict[str, str] = {}
    for split_name, pool in split_pools.items():
        for f in pool.files:
            if f in seen:
                raise AssertionError(
                    f"Data leakage detected: '{f}' appears in both "
                    f"'{seen[f]}' and '{split_name}' splits."
                )
            seen[f] = split_name
