"""
dataset_generator.py
=====================
Main entry point. Reads a YAML config, builds the microphone array and
asset pools, then generates N random EPISODES (continuous flights),
rendering each once and slicing it into consecutive chunks:
 
  <output_dir>/episode_000000/chunk_000/audio.wav
  <output_dir>/episode_000000/chunk_000/metadata.json
  <output_dir>/episode_000000/chunk_001/audio.wav
  ...
 
Consecutive chunk_XXX within one episode_XXXXXX are contiguous in time
(chunk_001 starts exactly where chunk_000 ends), so they can be
concatenated to reconstruct one continuous recording + continuous
ground-truth trajectory for sequence/tracking training, or used
individually for single-frame localization training.
 
PARALLEL GENERATION: episodes are fully independent of one another, so
they can be rendered across multiple CPU cores. Set `n_workers` in the
config (or pass --n_workers) to > 1 to use a process pool. Every episode
seed is drawn from `master_seed` BEFORE any parallel work starts, in
strict episode-index order, so the generated dataset (and even the
console progress-line order) is IDENTICAL regardless of how many workers
are used -- parallelism only changes wall-clock time, never the output.
 
TRAIN/VAL/TEST SPLITTING: if `split.enabled: true` in the config, the
four audio-file pools (drone/wind/ambient_unknown/known_unwanted) are
each divided into train/val/test at the FILE level, stratified per
source class, BEFORE any episode is generated -- so no audio file (and
therefore no single recording's characteristic noise floor or artifact)
is ever shared across splits. See dataset_split.py for the algorithm.
Output then goes to <output_dir>/<split_name>/episode_.../... and a
split_manifest.json at the dataset root documents exactly which files
went where.
 
Usage:
    python dataset_generator.py --config example_config.yaml
    python dataset_generator.py --config example_config.yaml --n_workers 8
    python dataset_generator.py --config example_config.yaml --split train
"""
 
import argparse
import concurrent.futures
import dataclasses
import json
import os
import numpy as np
import soundfile as sf
import yaml
 
import array_geometry as ag
from audio_pool import AudioPool
from dataset_split import split_pool_by_class, describe_split, assert_no_overlap
from mixer import AssetPools, MixerConfig, render_episode
from propagation import GroundReflectionConfig
from scenario import EpisodeSamplerConfig, EventConfig, PositionRange, sample_episode
from trajectory import MotionConfig
 
 
def build_event_cfg(d: dict) -> EventConfig:
    return EventConfig(**d)
 
 
def build_array_from_config(array_cfg: dict) -> ag.MicPositions:
    return ag.from_config(array_cfg)


def build_position_range(d: dict) -> PositionRange:
    return PositionRange(**d)
 
 
def build_motion_cfg(d: dict) -> MotionConfig:
    return MotionConfig(**d)
 
 
def build_mixer_cfg(cfg: dict, fs: int, class_map: dict) -> MixerConfig:
    ground_cfg = GroundReflectionConfig(**cfg.get("ground_reflection", {"enabled": False}))
    return MixerConfig(
        fs=fs,
        apply_air_absorption=cfg.get("apply_air_absorption", True),
        ground_reflection=ground_cfg,
        moving_block_size=cfg.get("moving_block_size", 2048),
        metadata_dt_s=cfg.get("metadata_dt_s", 0.05),
        event_fade_s=cfg.get("event_fade_s", 0.05),
        label_time_reference=cfg.get("label_time_reference", "reception"),
        label_vad_db=cfg.get("label_vad_db", 30.0),
        label_vad_frame_s=cfg.get("label_vad_frame_s", 0.02),
        label_min_gap_s=cfg.get("label_min_gap_s", 0.10),
        label_min_segment_s=cfg.get("label_min_segment_s", 0.10),
        class_map=class_map,
    )
 
 
def build_sampler_cfg(ep_cfg_dict: dict) -> EpisodeSamplerConfig:
    return EpisodeSamplerConfig(
        episode_duration_s_range=tuple(ep_cfg_dict["episode_duration_s_range"]),
        chunk_duration_s=float(ep_cfg_dict["chunk_duration_s"]),
        drone_event_cfg=build_event_cfg(ep_cfg_dict["drone_events"]),
        interferer_event_cfg=build_event_cfg(ep_cfg_dict["interferer_events"]),
        drone_position_range=build_position_range(ep_cfg_dict["drone_position_range"]),
        interferer_position_range=build_position_range(ep_cfg_dict["interferer_position_range"]),
        drone_motion_cfg=build_motion_cfg(ep_cfg_dict["drone_motion"]),
        interferer_motion_cfg=build_motion_cfg(ep_cfg_dict["interferer_motion"]),
        wind_snr_db_range=tuple(ep_cfg_dict["wind_snr_db_range"]),
        wind_mode_choices=tuple(ep_cfg_dict["wind_mode_choices"]),
        wind_coherent=ep_cfg_dict.get("wind_coherent", True),
        ambient_snr_db_range=tuple(ep_cfg_dict["ambient_snr_db_range"]),
        ambient_mode_choices=tuple(ep_cfg_dict["ambient_mode_choices"]),
        ambient_coherent=ep_cfg_dict.get("ambient_coherent", True),
        temperature_c_range=tuple(ep_cfg_dict["temperature_c_range"]),
        relative_humidity_pct_range=tuple(ep_cfg_dict["relative_humidity_pct_range"]),
        pressure_kpa_range=tuple(ep_cfg_dict["pressure_kpa_range"]),
    )
 
 
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
 
 
def _guard_against_empty_pools(
    pools: AssetPools, sampler_cfg: EpisodeSamplerConfig, label: str = ""
) -> EpisodeSamplerConfig:
    """
    Defends against a pool that ended up with zero audio files -- either
    an unused known_unwanted/ folder (a normal, supported setup: the
    README/example config both leave it with just a .gitkeep by default),
    or -- in split mode -- a source class with too few files to appear in
    every split (see dataset_split.py's warnings / split_manifest.json).
 
    Without this guard, scenario.py can still sample a nonzero event count
    for a category whose pool is empty (scenario.py has no visibility into
    which pools actually have files -- that's mixer.py's job), and
    mixer.py would then crash mid-render trying to draw a clip from an
    empty pool.
 
    - drone pool empty: the drone is the primary localization TARGET, so
      an empty pool while events are still configured to be sampled is
      almost certainly a misconfiguration (missing recordings, or -- in
      split mode -- a split left with zero drone files). Raise
      immediately, before any (possibly expensive) generation starts,
      rather than producing a "dataset" with silently zero drones, or
      crashing partway through.
    - known_unwanted pool empty: interferers are optional by design
      (n_events_range legitimately starts at 0 in the example config), so
      this is handled gracefully: force that category's event count to
      (0, 0) for this run instead of crashing, with a one-line notice
      printed once (not per-episode).
    """
    prefix = f"[{label}] " if label else ""
 
    if pools.drone.is_empty():
        raise RuntimeError(
            f"{prefix}The drone audio pool is empty -- add drone recordings under "
            f"assets/drone/ (or, in split mode, check split_manifest.json: this split "
            f"may have ended up with zero drone files -- add more recordings, or lower "
            f"n_events_range for this run) before generating."
        )
 
    if pools.known_unwanted.is_empty() and tuple(sampler_cfg.interferer_event_cfg.n_events_range) != (0, 0):
        print(f"{prefix}[guard] known_unwanted pool is empty -- forcing "
              f"interferer_events.n_events_range to (0, 0) for this run "
              f"(no known_unwanted recordings available; this is expected "
              f"if you haven't added any yet).")
        sampler_cfg = dataclasses.replace(
            sampler_cfg,
            interferer_event_cfg=dataclasses.replace(
                sampler_cfg.interferer_event_cfg, n_events_range=(0, 0)
            ),
        )
 
    return sampler_cfg
 
 
# ---------------------------------------------------------------------------
# Core per-episode work: sample, render, write to disk. Used directly in the
# sequential path, and via a tiny process-pool wrapper in the parallel path
# (see _init_worker / _worker_task below) -- so both paths share exactly one
# implementation and always produce byte-identical output.
# ---------------------------------------------------------------------------
 
def _build_context(mic_positions, pools, mixer_cfg, sampler_cfg, output_dir, fs) -> dict:
    return dict(
        mic_positions=mic_positions,
        pools=pools,
        mixer_cfg=mixer_cfg,
        sampler_cfg=sampler_cfg,
        output_dir=output_dir,
        fs=fs,
        mic_names=list(mic_positions.keys()),
    )
 
 
def _render_and_write_episode(context: dict, episode_index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    episode = sample_episode(context["sampler_cfg"], rng, seed)
    chunks = render_episode(episode, context["mic_positions"], context["pools"],
                             context["mixer_cfg"], rng)
 
    mic_names = context["mic_names"]
    fs = context["fs"]
 
    episode_id = f"episode_{episode_index:06d}"
    episode_dir = os.path.join(context["output_dir"], episode_id)
    os.makedirs(episode_dir, exist_ok=True)
 
    for c_idx, chunk in enumerate(chunks):
        chunk_id = f"chunk_{c_idx:03d}"
        chunk_dir = os.path.join(episode_dir, chunk_id)
        os.makedirs(chunk_dir, exist_ok=True)
 
        multichannel = np.stack([chunk["audio"][name] for name in mic_names], axis=1)
        # Clipping protection is done once per EPISODE in mixer.render_episode
        # (a per-chunk gain would break cross-chunk continuity).
        if np.max(np.abs(multichannel)) > 1.0:
            raise RuntimeError("chunk exceeds full scale after episode-level normalization")
 
        # PCM_24: soundfile's default for .wav is PCM_16, whose ~-98 dBFS
        # quantization floor is audible to GCC-PHAT (PHAT whitening gives
        # quiet high-frequency bins the same weight as loud ones).
        sf.write(os.path.join(chunk_dir, "audio.wav"), multichannel, fs, subtype="PCM_24")
 
        meta = chunk["metadata"]
        meta["mic_channel_order"] = mic_names
        meta["mic_groups"] = ag.infer_mic_groups(mic_names)
        with open(os.path.join(chunk_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
 
    return {
        "episode_index": episode_index,
        "episode_id": episode_id,
        "duration_s": episode.episode_duration_s,
        "n_chunks": len(chunks),
        "n_drone_events": len(episode.drones),
        "n_interferer_events": len(episode.interferers),
        "wind_mode": episode.wind_cfg.mode,
        "wind_snr_db": episode.wind_cfg.snr_db,
    }
 
 
def _print_progress(idx: int, n_episodes: int, result: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}[{idx + 1}/{n_episodes}] {result['episode_id']}: "
          f"{result['duration_s']:.1f}s episode, {result['n_chunks']} chunks, "
          f"{result['n_drone_events']} drone event(s), "
          f"{result['n_interferer_events']} interferer event(s), "
          f"wind={result['wind_mode']}({result['wind_snr_db']:.1f}dB)")
 
 
# --- process-pool plumbing: each worker process builds its context ONCE
#     (via the pool's initializer), then handles many (episode_index, seed)
#     tasks without re-sending the (fairly large) config/pools each time ---
_worker_context: dict = {}
 
 
def _init_worker(context: dict) -> None:
    global _worker_context
    _worker_context = context
 
 
def _worker_task(args) -> dict:
    episode_index, seed = args
    return _render_and_write_episode(_worker_context, episode_index, seed)
 
 
def run_generation(
    pools: AssetPools,
    mic_positions: ag.MicPositions,
    mixer_cfg: MixerConfig,
    sampler_cfg: EpisodeSamplerConfig,
    output_dir: str,
    fs: int,
    n_episodes: int,
    n_workers: int,
    master_seed: int,
    label: str = "",
) -> int:
    """
    Generates `n_episodes` episodes into `output_dir`, sequentially or in
    parallel depending on `n_workers`. Returns the total number of chunks
    written. `label` is just a console-progress prefix (e.g. the split
    name); it has no effect on the generated data.
    """
    os.makedirs(output_dir, exist_ok=True)
 
    master_rng = np.random.default_rng(master_seed)
    # All per-episode seeds are drawn up front, in strict index order,
    # before any parallel work begins -- this is what makes the output
    # independent of n_workers (see module docstring).
    seeds = [int(master_rng.integers(0, 2**31 - 1)) for _ in range(n_episodes)]
 
    context = _build_context(mic_positions, pools, mixer_cfg, sampler_cfg, output_dir, fs)
 
    total_chunks_written = 0
    prefix = f"[{label}] " if label else ""
 
    if n_workers <= 1:
        print(f"{prefix}[run] sequential (n_workers=1)")
        for idx, seed in enumerate(seeds):
            result = _render_and_write_episode(context, idx, seed)
            total_chunks_written += result["n_chunks"]
            _print_progress(idx, n_episodes, result, label)
    else:
        print(f"{prefix}[run] parallel, {n_workers} worker processes")
        tasks = list(enumerate(seeds))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(context,)
        ) as executor:
            # executor.map preserves task submission order in its results,
            # so progress lines print in the same episode-index order as
            # the sequential path, even though workers finish out of order.
            for idx, result in enumerate(executor.map(_worker_task, tasks)):
                total_chunks_written += result["n_chunks"]
                _print_progress(idx, n_episodes, result, label)
 
    print(f"{prefix}Done. {n_episodes} episodes, {total_chunks_written} chunks written to '{output_dir}'.")
    return total_chunks_written
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_episodes", type=int, default=None,
                         help="override n_episodes from the config file "
                              "(applies to every split being generated, if splitting is enabled)")
    parser.add_argument("--n_workers", type=int, default=None,
                         help="override n_workers from the config file "
                              "(1 = sequential, single process; "
                              "0 = use all available CPU cores)")
    parser.add_argument("--split", type=str, default=None,
                         help="only generate this one split (e.g. 'train'); "
                              "default generates every split listed in split.ratios, "
                              "one after another. Ignored if split.enabled is false.")
    args = parser.parse_args()
 
    cfg = load_config(args.config)
 
    mic_positions = build_array_from_config(cfg["array"])
    print(f"[array] {len(mic_positions)} microphones: {list(mic_positions.keys())}")
 
    fs = cfg["fs"]
    assets_cfg = cfg["assets"]
 
    full_pools = AssetPools(
        drone=AudioPool(assets_cfg["drone_dir"], fs, default_class_name="drone"),
        wind=AudioPool(assets_cfg["wind_dir"], fs, default_class_name="wind", allow_empty=True),
        ambient_unknown=AudioPool(assets_cfg["ambient_unknown_dir"], fs,
                                   default_class_name="ambient_unknown", allow_empty=True),
        known_unwanted=AudioPool(assets_cfg["known_unwanted_dir"], fs,
                                  default_class_name="known_unwanted", allow_empty=True),
    )
 
    # --- global, dataset-wide class_id map, computed from the FULL (unsplit)
    #     pools so ids stay stable and consistent across every split ---
    class_names = sorted(set(full_pools.drone.list_available_classes())
                          | set(full_pools.known_unwanted.list_available_classes()))
    class_map = {name: idx for idx, name in enumerate(class_names)}
    print(f"[classes] {len(class_map)} source classes: {class_map}")
 
    mixer_cfg = build_mixer_cfg(cfg, fs, class_map)
    sampler_cfg = build_sampler_cfg(cfg["episode"])
 
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "class_map.json"), "w") as f:
        json.dump({"class_name_to_id": class_map,
                   "id_to_class_name": {v: k for k, v in class_map.items()},
                   "num_classes": len(class_map)}, f, indent=2)
 
    master_seed = cfg.get("master_seed", 0)
    n_workers = args.n_workers if args.n_workers is not None else cfg.get("n_workers", 1)
    if n_workers is None or n_workers <= 0:
        n_workers = os.cpu_count() or 1
 
    split_cfg = cfg.get("split", {"enabled": False})
 
    if not split_cfg.get("enabled", False):
        # --- original, non-split behavior: one pool, one output_dir ---
        n_episodes = args.n_episodes or cfg["n_episodes"]
        sampler_cfg = _guard_against_empty_pools(full_pools, sampler_cfg)
        run_generation(full_pools, mic_positions, mixer_cfg, sampler_cfg,
                       output_dir, fs, n_episodes, n_workers, master_seed)
        return
 
    # --- split mode: partition every pool by file (stratified per class),
    #     then generate each requested split into its own subfolder ---
    ratios = split_cfg["ratios"]
    split_seed = split_cfg.get("split_seed", master_seed)
    split_n_episodes = split_cfg["n_episodes"]  # dict: {split_name: count}
 
    split_pools_per_category = {
        "drone": split_pool_by_class(full_pools.drone, ratios, split_seed),
        "wind": split_pool_by_class(full_pools.wind, ratios, split_seed),
        "ambient_unknown": split_pool_by_class(full_pools.ambient_unknown, ratios, split_seed),
        "known_unwanted": split_pool_by_class(full_pools.known_unwanted, ratios, split_seed),
    }
 
    for category, per_split in split_pools_per_category.items():
        assert_no_overlap(per_split)
 
    manifest = {}
    all_warnings = []
    for category, pool in [("drone", full_pools.drone), ("wind", full_pools.wind),
                            ("ambient_unknown", full_pools.ambient_unknown),
                            ("known_unwanted", full_pools.known_unwanted)]:
        summary = describe_split(pool, split_pools_per_category[category])
        manifest[category] = summary
        all_warnings.extend(summary["warnings"])
 
    with open(os.path.join(output_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
 
    print(f"[split] wrote split_manifest.json ({len(ratios)} splits: {list(ratios.keys())})")
    for w in all_warnings:
        print(f"[split] WARNING: {w}")
 
    split_names = [args.split] if args.split else list(ratios.keys())
    for split_name in split_names:
        if split_name not in ratios:
            raise ValueError(f"--split '{split_name}' is not one of the configured splits: {list(ratios)}")
 
        split_pools = AssetPools(
            drone=split_pools_per_category["drone"][split_name],
            wind=split_pools_per_category["wind"][split_name],
            ambient_unknown=split_pools_per_category["ambient_unknown"][split_name],
            known_unwanted=split_pools_per_category["known_unwanted"][split_name],
        )
 
        n_episodes = args.n_episodes or split_n_episodes[split_name]
        split_output_dir = os.path.join(output_dir, split_name)
        # a distinct, deterministic seed per split, derived from split_seed
        # (order-stable: based on position in the configured ratios, not
        # dict hash) so re-runs are reproducible regardless of --split
        split_master_seed = split_seed + list(ratios.keys()).index(split_name) + 1
 
        split_sampler_cfg = _guard_against_empty_pools(split_pools, sampler_cfg, label=split_name)
 
        run_generation(split_pools, mic_positions, mixer_cfg, split_sampler_cfg,
                       split_output_dir, fs, n_episodes, n_workers,
                       split_master_seed, label=split_name)
 
 
if __name__ == "__main__":
    main()
 
