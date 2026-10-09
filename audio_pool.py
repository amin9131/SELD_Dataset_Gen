"""
audio_pool.py
=============
Loads REAL recorded audio files from category folders on disk. No signal
is synthetically generated here -- every source clip used in a scenario
comes from a user-supplied .wav/.flac/.ogg file.
 
Expected folder layout (paths are configurable in the YAML config):
 
    assets/
        drone/              -> target signal: drone motor/propeller recordings
        wind/               -> wind noise recordings (diffuse/ambient noise class)
        ambient_unknown/    -> unidentified environmental background noise
        known_unwanted/     -> identifiable but non-target sound sources
                               (birds, cars, voices, other real point sources
                                that are NOT the drone we want to localize)
 
Each category is a pool of interchangeable clips; a scenario draws one
(or more) random clips from the relevant pool(s).
"""
 
import os
import glob
import random
from dataclasses import dataclass
from typing import Optional
from math import gcd
 
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
 
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif")
 
 
@dataclass
class AudioClip:
    signal: np.ndarray      # mono float64, shape (n_samples,)
    fs: int                 # sample rate AFTER resampling to target fs
    source_path: str        # original file path, for provenance in metadata
    class_name: str         # source-category label (e.g. "drone", "quadcopter",
                             # "bird", "car") -- see AudioPool for how it's derived
 
 
class AudioPool:
    """A pool of audio files in one folder, loaded on demand and resampled
    / downmixed to mono at a fixed target sample rate.
 
    Class-label taxonomy: files can optionally be organized into
    subfolders, one per source category, e.g.:
 
        assets/drone/quadcopter/*.wav
        assets/drone/fixedwing/*.wav
        assets/known_unwanted/bird/*.wav
        assets/known_unwanted/car/*.wav
 
    A file's class_name is taken from its immediate parent subfolder
    relative to the pool's root folder. Files placed directly in the root
    folder (no subfolder) fall back to `default_class_name`, so existing
    flat folder layouts keep working unchanged -- every clip just gets the
    single default class (e.g. "drone").
    """
 
    def __init__(self, folder: str, target_fs: int, default_class_name: str,
                 allow_empty: bool = False):
        self.folder = folder
        self.target_fs = target_fs
        self.default_class_name = default_class_name
        self.files = sorted(
            f for ext in AUDIO_EXTENSIONS
            for f in glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True)
        )
        if not self.files and not allow_empty:
            raise FileNotFoundError(
                f"No audio files found in '{folder}'. "
                f"Supported extensions: {AUDIO_EXTENSIONS}"
            )
 
    @classmethod
    def from_file_list(cls, files: list, folder: str, target_fs: int,
                        default_class_name: str) -> "AudioPool":
        """
        Build a pool restricted to an explicit list of files (which must
        already live under `folder`), bypassing the directory scan. This
        is how train/val/test split pools are constructed -- see
        dataset_split.py -- without re-globbing the whole folder and
        without requiring every split to be non-empty up front.
        """
        obj = cls.__new__(cls)
        obj.folder = folder
        obj.target_fs = target_fs
        obj.default_class_name = default_class_name
        obj.files = list(files)
        return obj
 
    def _class_name_for(self, path: str) -> str:
        rel = os.path.relpath(path, self.folder)
        parts = rel.split(os.sep)
        return parts[0] if len(parts) > 1 else self.default_class_name
 
    def list_available_classes(self) -> list:
        """All distinct class names present in this pool (for building a
        global, dataset-wide class_id map -- see dataset_generator.py)."""
        return sorted({self._class_name_for(f) for f in self.files})
 
    def __len__(self):
        return len(self.files)
 
    def is_empty(self) -> bool:
        return len(self.files) == 0
 
    def _load_and_resample(self, path: str) -> AudioClip:
        data, fs_in = sf.read(path, always_2d=True)
        mono = data.mean(axis=1).astype(np.float64)
        if fs_in != self.target_fs:
            g = gcd(int(fs_in), int(self.target_fs))
            mono = resample_poly(mono, self.target_fs // g, fs_in // g)
        return AudioClip(signal=mono, fs=self.target_fs, source_path=path,
                          class_name=self._class_name_for(path))
 
    def random_clip(self, rng: Optional[random.Random] = None) -> AudioClip:
        rng = rng or random
        return self._load_and_resample(rng.choice(self.files))
 
    def get_clip(self, index: int) -> AudioClip:
        return self._load_and_resample(self.files[index])
 
 
def fit_to_duration(clip: AudioClip, duration_s: float, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Return a signal of exactly `duration_s` seconds built from `clip`:
      - longer clip  -> random contiguous segment of the right length
      - shorter clip -> looped (tiled) to fill the requested duration
    """
    rng = rng or random
    n_target = int(round(duration_s * clip.fs))
    sig = clip.signal
    n = len(sig)
    if n == 0:
        return np.zeros(n_target)
    if n >= n_target:
        max_start = n - n_target
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return sig[start:start + n_target].copy()
    n_repeats = int(np.ceil(n_target / n))
    tiled = np.tile(sig, n_repeats)
    start = rng.randint(0, len(tiled) - n_target) if len(tiled) > n_target else 0
    return tiled[start:start + n_target].copy()
 
 
def rms(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return 0.0
    val = np.sqrt(np.mean(signal.astype(np.float64) ** 2))
    return val if val > 0 else 1e-12
 
 
def normalize_rms(signal: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    return signal * (target_rms / rms(signal))
 
