Outdoor Acoustic Drone-Localization Dataset Generator

A modular Python framework for simulating multi-source, free-field (outdoor) acoustic scenes -- with realistic continuous drone flight -- for training/evaluating drone acoustic-localization algorithms (TDOA/GCC-PHAT, beamforming, deep-learning position/DOA estimators, multi-target tracking, etc.).

Every source signal is a real recorded audio file you provide -- the framework does not synthesize drone/wind/noise signals; it only handles the spatial physics (propagation, motion, mixing, array geometry).

What it models
Layer	Included
Level 1 (always on)	exact per-microphone propagation delay, spherical (1/r) spreading loss
Level 2 (togglable)	frequency-dependent atmospheric absorption (ISO 9613-1); a single specular ground reflection (image-source) with a realistic, frequency- and angle-dependent reflection coefficient (Delany-Bazley ground impedance model)
Motion	drones (and interferers) follow continuous, physically-bounded flight trajectories -- not fixed points -- rendered via block-based overlap-add propagation that reproduces time-varying TDOA and an approximate Doppler effect
Multi-source	any number of drones + "known-unwanted" interferers, each appearing and disappearing at independent random times (variable polyphony -- see "Variable number of active sources" below), not a fixed count present for the whole episode
Background noise	wind and unidentified ambient noise, each independently "diffuse" (incoherent per mic) or "directional" (far-field plane wave from a random/fixed bearing), each with its own target SNR
Realistic flight motion

Instead of one static position per sample, each drone follows a continuous trajectory spanning a whole "episode" (e.g. 15-30 s of simulated flight), picked from four physically-motivated patterns (trajectory.py):

Pattern	Behavior
hover_jitter	near-stationary loiter with small mean-reverting (Ornstein-Uhlenbeck) drift -- realistic GPS-hold wander, not a frozen point
linear_flyby	constant-velocity straight-line pass (approach/recede/overflight)
waypoint_mission	multi-leg flight between random waypoints, each leg flown with a trapezoidal velocity profile (accelerate -> cruise at v_max -> decelerate) -- the same profile shape real autopilots (PX4/ArduPilot mission mode) fly, with optional hover dwell at waypoints
orbit	constant-radius circular loiter around a point of interest at fixed altitude -- the classic surveillance/observation pattern

Speed and acceleration are drawn from configurable, physically-plausible ranges (default: 3-18 m/s cruise, 1.5-4 m/s² acceleration).

Inter-scenario motion strategy: episodes -> chunks

An episode (one continuous simulated flight, e.g. 20 s) is rendered once, as one continuous multichannel recording, and is then sliced into consecutive fixed-length chunks (e.g. 4 s each). Each chunk is written out as one training sample, but because chunks are literal contiguous slices of one continuously-rendered episode:

audio is bit-for-bit continuous across chunk_000 -> chunk_001 -> ... (verified: boundary sample-to-sample jump is the same order of magnitude as an ordinary in-chunk sample-to-sample step)
ground-truth drone position is continuous across the same boundaries

This means the exact same generated data supports both single-frame localization training (use any one chunk on its own) and sequence/tracking training (concatenate consecutive chunks from the same episode to reconstruct one long, physically continuous flight).

Variable number of active sources (random entries/exits)

Sources are not present for the whole episode. Each drone or interferer is a SourceEvent (scenario.py): one appearance with its own random onset time, duration, and trajectory confined to just that appearance. An episode gets a random NUMBER of such appearances (n_events_range), each placed at a random, independent time within the episode (event_duration_s_range controls how long each stays audible). This means, across an episode:

there can be stretches with zero active drones (silence/background only), stretches with one, and stretches where two or more events overlap in time (simultaneous sources, same or different class)
a chunk's metadata.json only lists the events that actually overlap that chunk's time window (n_active_drones / n_active_known_unwanted), so the active-source count naturally varies from chunk to chunk, just like real-world polyphonic recordings (similar in spirit to how DCASE SELD data is constructed)
each event fades in/out over a short (event_fade_s, default 50 ms) raised-cosine ramp so entries/exits don't produce an audible click
background noise (wind/ambient) is unaffected by this -- it still runs continuously for the whole episode, since it's not a localizable point source

Configure this per source category under episode.drone_events / episode.interferer_events in the YAML (n_events_range, event_duration_s_range, min_event_duration_s).

Realistic ground reflection (Delany-Bazley model)

When ground_reflection.enabled: true, the single-bounce ground echo (image-source method: a mirrored virtual source below the ground plane) is no longer scaled by one constant real number. By default (model: delany_bazley) it uses the Delany-Bazley empirical ground impedance model, giving a complex, frequency- and angle-dependent reflection coefficient:

R(f) = (zeta(f)*cos(theta) - 1) / (zeta(f)*cos(theta) + 1)

where zeta(f) is the ground's normalized acoustic impedance (a function of frequency and the ground's flow resistivity) and theta is the angle of incidence from the ground normal (computed exactly from the source and microphone heights and the reflected path length -- steeper bounces reflect differently than grazing ones). This reproduces real, well-known outdoor ground-effect behavior that a constant scalar cannot: harder ground (asphalt, concrete) reflects strongly and almost flatly across the whole band, while soft porous ground (grass, forest floor, fresh snow) absorbs progressively more at higher frequencies and shows the characteristic reflectivity dip near grazing incidence.

Pick a ground surface with ground_type (presets in ground_impedance.GROUND_TYPE_PRESETS: snow_fresh, forest_floor, soil_soft, grass, sand, soil_compacted, gravel, asphalt, concrete, rigid), or give an exact flow_resistivity_kpa_sm2 to override the preset. The old constant-coefficient behavior is still available via model: constant + reflection_coefficient if you want the simpler legacy model.

Reference: Delany, M.E. and Bazley, E.N. (1970), "Acoustical properties of fibrous absorbent materials", Applied Acoustics 3(2), 105-116 -- the standard starting point for outdoor ground-effect modeling (see also Attenborough/Embleton's outdoor sound propagation work and ISO 9613-2).

Spatially-coherent diffuse noise (wind / ambient)

By default, mode: diffuse wind/ambient noise is no longer generated as fully independent random noise at each microphone. It's synthesized as a physically realistic spatially-coherent diffuse field, following the standard spherically isotropic diffuse-field coherence law used throughout array-processing and beamforming literature:

Gamma_ij(f) = sinc(2 f d_ij / c)

where d_ij is the distance between microphones i and j. Nearby microphones end up seeing strongly correlated noise (especially at low frequencies), while far-apart microphones/sub-arrays see progressively less correlated noise -- exactly like real outdoor wind/ambient noise, and unlike fully independent per-mic noise, which is an artificially easy case for spatial filters (MVDR, MUSIC, GCC-PHAT, ...) to suppress. Independent-per-mic noise makes a beamformer look better than it would on real recordings, since it never has to work against any real spatial noise structure.

Implementation: M mutually-independent noise realizations are drawn from your real wind/ambient audio files (one per microphone -- still no synthetic signal generation anywhere), then mixed per STFT frequency bin with a matrix whose Gram matrix reproduces the target coherence for this exact array geometry (the Habets-Cohen-Gannot method), then scaled to the configured SNR per channel. This was validated end-to-end: measured coherence from generated audio matches the theoretical curve to within a few percent across frequency, for both closely- and widely-spaced microphone pairs.

Set wind_coherent: false / ambient_coherent: false per category to fall back to the older, simpler independent-per-mic model (kept only for ablation/comparison studies -- it does not reflect real outdoor noise).

Reference: Cook, R.K., Waterhouse, R.V., Berendt, R.D., Edelman, S. and Thompson, M.C. (1955), "Measurement of Correlation Coefficients in Reverberant Sound Fields," JASA 27(6); Habets, E.A.P., Cohen, I. and Gannot, S. (2008), "Generating nonstationary multisensor signals under a spatial coherence constraint," JASA 124(5), 2911-2917.

Folder layout expected for audio assets
assets/
    drone/              <- target class: real drone motor/propeller recordings
    wind/               <- wind noise recordings
    ambient_unknown/    <- unidentified environmental background noise
    known_unwanted/     <- other real, localizable-but-non-target sources
                           (birds, road noise, voices, etc.)

Supported formats: .wav .flac .ogg .aiff .aif (mono or multi-channel; multi-channel files are downmixed to mono on load). Files are automatically resampled to the project's target fs. Each drone/interferer draws ONE continuous clip covering the WHOLE episode duration (looped if the file is shorter), so the audio content itself, not just the position, is continuous across chunks.

Before you add files: the framework assumes these clips are "dry" (close-mic'd / on-board / source-level) recordings -- see "Dry source" assumption under Known simplifications below before picking recordings that were themselves captured from a distant, stationary microphone while the drone was flying.

Source-class labels (for joint localization + detection)

drone/ and known_unwanted/ can optionally be split into subfolders, one per source category -- e.g.:

assets/drone/quadcopter/*.wav
assets/drone/fixedwing/*.wav
assets/known_unwanted/bird/*.wav
assets/known_unwanted/car/*.wav

A clip's class name is taken from its immediate subfolder; files placed directly in drone/ or known_unwanted/ (no subfolder) simply get the default class name ("drone" / "known_unwanted"), so a flat layout keeps working unchanged. At startup, dataset_generator.py scans both pools, builds one global, dataset-wide class_id map (so ids are consistent across every episode, not re-discovered per episode), and writes it to output/class_map.json. Every chunk's metadata.json then carries, for each active source, category / class_name / class_id -- both at the source level and duplicated inside every per-frame trajectory_samples entry, alongside position_xyz_m and tdoa_s -- so each time step gives you position + TDOA + class together, ready for joint SELD-style (localization + detection) training.

Files
File	Purpose
atmosphere.py	speed of sound, ISO 9613-1 air-absorption coefficient
array_geometry.py	build custom/linear/circular/tetrahedral/multi-sub-array mic geometries
audio_pool.py	load real audio files, resample, fit to duration (crop/loop), RMS normalize, derive class_name from subfolder
trajectory.py	realistic flight-path models (hover jitter, linear flyby, waypoint mission, orbit)
propagation.py	frequency-domain point-source propagation for a FIXED position: delay + spreading + air absorption + optional ground reflection
ground_impedance.py	Delany-Bazley ground impedance model: realistic frequency- and angle-dependent reflection coefficient for outdoor ground surfaces
moving_propagation.py	block-based overlap-add propagation for a MOVING source (reuses propagation.py per time-block)
diffuse_noise.py	diffuse (spatially-coherent by default) / directional background-noise injection with SNR control
diffuse_field.py	spatially-coherent diffuse noise field synthesis (spherically isotropic coherence law + Habets-Cohen-Gannot mixing-matrix method)
scenario.py	random EPISODE sampling: for each category, a random NUMBER of source appearances ("events"), each with its own onset/duration/trajectory/SNR
mixer.py	renders a full episode once (each event only added during its own active window, with onset/offset fades), slices it into consecutive training chunks, and attaches class_id/class_name alongside position/TDOA in the ground-truth metadata
dataset_generator.py	main CLI: reads YAML config, builds the global class_map, batch-generates episodes/chunks (optionally in parallel across CPU cores, optionally split into train/val/test)
dataset_split.py	automatic, leakage-free train/val/test splitting -- divides audio files (not episodes) per source class
validate_gcc_phat.py	sanity checks: static-source AND moving-source TDOA correctness against exact geometry
example_config.yaml	example configuration (paper-style dual tetrahedral array + realistic motion)

(Training-preparation modules -- feature extraction, label alignment, normalization, PyTorch Dataset, PIT loss -- are documented separately below in "Training-preparation pipeline (added on top of the generator)".)

Beyond the generator and training-prep layers there are two more files:

File	Purpose
model.py	SELDModel: log-mel + GCC-PHAT CNN encoders -> causal CRNN/Conformer backbone -> activity / position / class heads (single head set, max_tracks slots)
streaming_tracker.py	inference-time, class-selective multi-target tracker threading per-chunk detections into persistent track IDs across chunks
Quick start
Put your real recordings into the four assets/* folders above.
Edit example_config.yaml (or copy it) to set your array geometry, motion pattern mixture, position ranges, noise SNR ranges, episode/chunk durations, and how many episodes to generate.
Sanity-check the physics:
bash
   python validate_gcc_phat.py --config example_config.yaml

This runs both a static-source check and a moving-source check; both should report a TDOA error within a few samples. 4. Generate the dataset:

bash
   python dataset_generator.py --config example_config.yaml --n_episodes 200

For large datasets, add --n_workers 0 to use every CPU core (or --n_workers N for exactly N processes) -- see "Parallel generation" below.

Parallel generation

Episodes are fully independent of one another, so they can be rendered across multiple CPU cores. Set n_workers in the config, or pass --n_workers on the command line:

n_workers: 1 (default) -- sequential, single process. Easiest to debug (a full Python traceback if something goes wrong); use this for small test runs.
n_workers: 0 -- use every available CPU core automatically.
n_workers: N -- use exactly N worker processes.

The generated dataset is guaranteed byte-for-byte identical no matter what you choose here -- every episode's random seed is drawn from master_seed up front, in strict episode-index order, before any parallel work starts, so which worker happens to render which episode never affects the result (only wall-clock time changes). This was verified directly: the same config run with n_workers: 1 and n_workers: 3 produces identical audio.wav/metadata.json files for every episode/chunk.

Each worker process builds its own AudioPool handles once (via a pool initializer) and then renders many episodes without re-sending the config for each one, so the per-task overhead stays small even for short episodes.

Automatic train/val/test splitting (leakage-free)

Set split.enabled: true to have the framework automatically divide your dataset into train/val/test sets at the audio-FILE level, before any episode is generated:

yaml
split:
  enabled: true
  ratios: {train: 0.7, val: 0.15, test: 0.15}
  split_seed: 123
  n_episodes: {train: 140, val: 30, test: 30}

Why file-level, not episode-level: if the same physical recording (e.g. one specific drone .wav) could appear in both a training episode and a test episode, a model can learn to recognize that recording's noise floor, mic self-noise, or compression artifacts rather than the general acoustic phenomenon -- classic session-level leakage in audio ML, which quietly inflates val/test metrics without the model actually generalizing. Splitting whole episodes would NOT prevent this, since nothing would stop the same underlying file from being redrawn in episodes on both sides of the split. So instead, every file in drone/, wind/, ambient_unknown/, and known_unwanted/ is assigned to exactly ONE split up front; each split's episodes then only ever draw clips from their own file subset. Verified directly: cross-referencing every source_file actually used in the generated train/, val/, and test/ metadata confirms zero overlap.

The split is stratified per source class (the same subfolder-derived classes used for class_id labels), so train/val/test each get a proportional share of every class rather than, say, all fixedwing recordings accidentally landing only in train. If a class has too few files to appear in every split, a warning is printed and recorded in split_manifest.json -- it's never silent.

Output layout becomes <output_dir>/<split_name>/episode_.../... instead of going directly into <output_dir>, and n_episodes at the top of the config is ignored in favor of the per-split counts under split.n_episodes. A split_manifest.json at the dataset root records exactly which files (grouped by class) went to which split, for a one-file leakage audit.

Use --split train (or val/test) to generate just one split in a given run -- e.g. to spread the three splits across separate machines or invocations. This was verified to produce byte-identical output to generating all three splits together in one run.

Output format
output/
    class_map.json        # {class_name: class_id} for the WHOLE dataset (consistent everywhere)
    episode_000000/
        chunk_000/
            audio.wav        # multichannel WAV, one channel per mic
            metadata.json     # full ground truth for this 4 s window
        chunk_001/
            ...               # continues exactly where chunk_000 left off
    episode_000001/
        ...

Each metadata.json contains:

episode_time_range_s: absolute [t0, t1] of this chunk within its episode
mic_positions_m, medium (temperature/humidity/pressure/speed of sound)
class_map: the same dataset-wide {class_name: class_id} map, duplicated here so each file is self-contained (no need to load a sibling file)
n_active_drones / n_active_known_unwanted: how many events actually overlap THIS chunk's time window (varies chunk to chunk -- see "Variable number of active sources" above)
drones: a list with only the drone events active in this chunk (can be empty). Each entry has category ("drone"), class_name, class_id, source_file, onset_s/offset_s (the event's full absolute onset/ offset time within the episode, even if only part of it falls in this chunk), and trajectory_samples -- a time series (default 20 Hz, configurable via metadata_dt_s), covering just the part of the event that overlaps this chunk, of {t, position_xyz_m, tdoa_s, class_name, class_id}. Every sampled instant gives you position + TDOA + class together, ready for direct-supervision joint localization+detection (SELD-style) training targets.
known_unwanted: same structure as drones (only events overlapping this chunk, with onset_s/offset_s, class_name/class_id, trajectory_samples with {t, position_xyz_m, tdoa_s, class_name, class_id} -- position + TDOA + class, exactly like drones), useful both as a hard-negative / distractor class for the detection head, and as a fully-labeled non-target class if you want one model that localizes AND classifies every point-source class it hears (not just "drone vs. not").
wind / ambient noise mode and realized SNR (these run continuously for the whole episode, independent of drone/interferer entries and exits)
Customizing the microphone array

Four ways to define array: in the YAML config:

custom_points: fully manual {name: [x,y,z]} list
linear: uniform linear array (n_mics, spacing, axis)
circular: uniform ring (n_mics, radius, plane)
tetrahedral: the paper's 4-mic AOA+elevation cluster
distributed_sub_arrays: place any number of the above sub-arrays at any set of centers (generalizes the paper's "two distributed 4-mic arrays" to N arrays anywhere)
Customizing motion

episode.drone_motion and episode.interferer_motion each take a pattern_weights dict (probabilities need not sum to 1, they're renormalized) plus ranges for max speed, max acceleration, hover jitter amplitude, orbit radius, waypoint count, and waypoint dwell time. Set any pattern's weight to 0 to disable it (the example config gives interferers a simpler hover/flyby-only mixture, and disables their orbit/waypoint_mission patterns).

Known simplifications / next steps
Ground reflection is still a SINGLE specular bounce (one image source; no multiple reflections between the ground and other surfaces, and no other reflecting surfaces like buildings or trees). Within that single bounce, the reflection coefficient IS now frequency- and angle-dependent (Delany-Bazley model) rather than a constant scalar -- see "Realistic ground reflection" above.
No wind-direction-dependent effective sound-speed model yet (isotropic v_sound is used).
Doppler is only APPROXIMATE: it emerges naturally from the block-based time-varying delay (smoothly changing per ~21 ms block), which is accurate for realistic drone speeds but is not an exact continuous-time Doppler model.
"Dry source" assumption (important): the framework assumes every drone/ and known_unwanted/ clip is a clean, source-level recording -- as if captured very close to the source (or on-board it), with no meaningful relative motion between the source and the microphone that originally recorded the clip. All delay, spreading, air absorption, and Doppler in the OUTPUT come only from the trajectory this framework simulates. If your uploaded clip is itself a recording of a drone already flying past a distant, stationary microphone, that clip already contains its OWN (unknown, unlogged) Doppler shift and amplitude fluctuation from that original recording session. Feeding it in as a "source" and then applying a second, synthetic flight path on top does not produce an invalid signal, but it does mean the final audio reflects two superimposed motions -- the original uncontrolled one baked into the file, plus the synthetic one this framework logs as ground truth -- so the exact ground-truth position/TDOA labels no longer perfectly explain 100% of what's audible in the mix. Practical guidance: prefer on-board / close-mic'd recordings when possible (near-zero relative velocity to the recording mic, so this issue mostly disappears); if only distant fly-by recordings are available, prefer trimming to the closest-approach moment (near-zero radial velocity there); otherwise treat the residual mismatch as an acceptable source of extra real-world variability rather than a correctness bug.
Diffuse noise now HAS spatial coherence structure by default (spherically isotropic model -- see "Spatially-coherent diffuse noise" above), but the model is still a simplification: it assumes a uniform, all-directions noise field, whereas real outdoor wind/ambient noise can have a somewhat dominant arrival pattern (e.g. more energy from the upwind side) that a pure isotropic model doesn't capture -- mode: directional partially covers that case, but a hybrid (mostly-diffuse-with-a-dominant-lobe) model would be more realistic still.
GCC-PHAT (used only in validate_gcc_phat.py, not in the generator itself) is fundamentally ambiguous for near-periodic/tonal test signals; this is a known property of the algorithm, not a pipeline bug -- real drone recordings have enough broadband turbulence/motor noise structure that this is not an issue in practice.
Training-preparation pipeline (added on top of the generator)

Five modules turn the generator's raw output (audio.wav + metadata.json per chunk) into batched, normalized, PIT-loss-ready PyTorch tensors. None of them modify the generator itself (mixer.py, dataset_generator.py, etc. are unchanged) -- they are a separate consumer layer.

File	Purpose
feature_extraction.py	Array-agnostic features from one shared STFT: log-mel spectrogram per mic channel, and frame-wise GCC-PHAT per mic pair (same PHAT technique as validate_gcc_phat.py, including its n_fft // 2 lag-window cap).
label_alignment.py	Resamples the sparse, per-event trajectory_samples in metadata.json onto the exact audio-frame grid (linear interpolation of position; TDOA is recomputed from the interpolated position via propagation.true_tdoa_seconds, not interpolated directly), converts chunk-relative to absolute episode time, and reshapes multi-source labels into a fixed, padded (n_frames, max_tracks, ...) tensor with an activity mask.
feature_normalization.py	Per-feature-dimension z-score standardization (per (mic, mel_bin) for log-mel, per (pair, lag) for GCC-PHAT), fit with a single-pass Welford accumulator so it scales to large datasets. Must be fit on the train split only and reused on val/test (same leakage principle dataset_split.py applies to raw audio files, applied here to feature statistics).
torch_dataset.py	SELDDataset / build_dataloader: wraps the two modules above into a torch.utils.data.Dataset with a shape-checking collate function (raises rather than silently mis-batching mismatched chunks). First module in the project depending on torch.
pit_loss.py	Permutation-Invariant Training loss (Hungarian matching via scipy.optimize.linear_sum_assignment) for the multi-track labels label_alignment.py produces -- one permutation per (batch element, category), pooled over the whole chunk, matching how a track slot corresponds to one whole SourceEvent. Includes a standalone self-check (python pit_loss.py) verifying the loss is differentiable and genuinely permutation-invariant.

Typical end-to-end usage:

python
from feature_extraction import FeatureConfig
from feature_normalization import FeatureNormalizer
from torch_dataset import SELDDataset, build_dataloader
from pit_loss import pit_loss_multi_category, target_from_batch

norm = FeatureNormalizer().fit("output/train")   # fit on TRAIN ONLY
norm.save("output/train/feature_norm_stats.json")

ds = SELDDataset("output/train", feat_cfg=FeatureConfig(), max_tracks=3, normalizer=norm)
loader = build_dataloader(ds, batch_size=8, shuffle=True)

# model.py's SELDModel has ONE set of heads, so merge both label categories
# into a single track pool (max_tracks = 2 * 3 = 6); the class head tells drone
# slots from known_unwanted slots.
from model import ModelConfig, build_model, infer_dims_from_batch
from pit_loss import pit_loss_single_category, merge_category_targets

batch = next(iter(loader))
model = build_model(ModelConfig(**infer_dims_from_batch(batch), n_classes=2, max_tracks=6))

for batch in loader:
    pred = model(batch["logmel"], batch["gcc_phat"])
    loss = pit_loss_single_category(pred, merge_category_targets(batch))["loss"]
    loss.backward()

(pit_loss_multi_category is for the alternative design with a separate {category: pred} output per category.)

Known limitations of this layer (documented in each file's own docstring, summarized here):

Log-mel and GCC-PHAT are computed on the physically MIXED multichannel signal -- they cannot be un-mixed into one clean feature per source when events overlap in time. PIT fixes the label/matching side of variable polyphony, not the underlying source-separation problem the model itself has to learn.
label_alignment.py raises (rather than silently truncating) if a chunk has more simultaneously-active events than max_tracks -- set max_tracks with your config's n_events_range values in mind.
SELDDataset does not cache computed features to disk; for very large datasets, I/O + on-the-fly STFT/mel/GCC-PHAT computation may become the training bottleneck (not addressed here).