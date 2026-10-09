"""
validate_gcc_phat.py
=====================
Sanity checks for the whole pipeline:
 
  1. STATIC check: a single non-moving source (HoverJitterTrajectory with
     zero jitter) at a known position -- GCC-PHAT on the rendered audio
     should recover the exact geometric TDOA.
 
  2. MOVING check: a LinearFlybyTrajectory -- GCC-PHAT is run on a short
     local window near a chosen time t, and compared against the exact
     geometric TDOA at trajectory.position_at(t). This confirms the
     block-based moving-source propagation (moving_propagation.py)
     produces correctly time-varying delay, not just a static average.
 
Usage:
    python validate_gcc_phat.py --config example_config.yaml
"""
 
import argparse
import random
import numpy as np
import yaml
 
from dataset_generator import build_array_from_config
from audio_pool import AudioPool
from propagation import GroundReflectionConfig, MediumConditions, true_tdoa_seconds
from moving_propagation import propagate_moving_source_to_array
from trajectory import HoverJitterTrajectory, LinearFlybyTrajectory
from array_geometry import as_matrix
from audio_pool import AudioClip
from mixer import MixerConfig, _render_event
from scenario import SourceEvent
 
 
def gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray, fs: float, max_tau: float = None) -> float:
    """Classic GCC-PHAT TDOA estimator: tau_ab = t_a - t_b (seconds)."""
    n = sig_a.shape[0] + sig_b.shape[0]
    n_fft = 1
    while n_fft < n:
        n_fft *= 2
 
    SIG_A = np.fft.rfft(sig_a, n=n_fft)
    SIG_B = np.fft.rfft(sig_b, n=n_fft)
    R = SIG_A * np.conj(SIG_B)
    R /= np.maximum(np.abs(R), 1e-12)
 
    cc = np.fft.irfft(R, n=n_fft)
    max_shift = n_fft // 2
    if max_tau is not None:
        max_shift = min(int(max_tau * fs), max_shift)
 
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(fs)
 
 
class _FakePool:
    """Minimal stand-in for AudioPool: always returns one broadband-noise clip."""
    def __init__(self, fs: int, seconds: float = 2.0):
        noise = np.random.default_rng(1).standard_normal(int(fs * seconds))
        self._clip = AudioClip(signal=noise, fs=fs, source_path="<synthetic>", class_name="drone")
 
    def random_clip(self, rng=None):
        return self._clip
 
 
def check_event_time_alignment(fs, mic_positions, mic_a_name, mic_b_name, medium, ground_cfg) -> bool:
    """
    CHECK 3 -- regression test for the onset-time bug. Renders a flyby
    through the SAME code path mixer.render_episode uses (_render_event)
    for an event that starts at episode time 10 s, then checks TDOA at two
    different local times. If the trajectory were read at episode time
    instead of event-local time, every block would clamp to the final
    position and BOTH estimates would be identical (and wrong).
    """
    duration = 3.0
    traj = LinearFlybyTrajectory(np.array([-20.0, 5.0, 15.0]), np.array([8.0, 0.5, 0.2]), duration)
    event = SourceEvent(kind="drone", start_s=10.0, end_s=10.0 + duration, trajectory=traj)
    mixer_cfg = MixerConfig(fs=fs, apply_air_absorption=True, ground_reflection=ground_cfg)
    per_mic, _, _ = _render_event(event, _FakePool(fs), fs, mic_positions, medium, mixer_cfg, random.Random(0))
 
    ok = True
    for t_local in (0.6, 2.4):
        s0, s1 = int((t_local - 0.15) * fs), int((t_local + 0.15) * fs)
        est = gcc_phat(per_mic[mic_a_name][s0:s1], per_mic[mic_b_name][s0:s1], fs, max_tau=0.05)
        true = true_tdoa_seconds(traj.position_at(t_local), mic_positions[mic_a_name],
                                 mic_positions[mic_b_name], medium)
        err = abs(est - true) * fs
        print(f"event-local t={t_local}s: true={true*1e6:8.1f} us  est={est*1e6:8.1f} us  err={err:5.1f} samples")
        ok &= err <= 25.0
    return ok
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mic_a", default=None)
    parser.add_argument("--mic_b", default=None)
    args = parser.parse_args()
 
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
 
    fs = cfg["fs"]
    mic_positions = build_array_from_config(cfg["array"])
    mic_names, mic_mat = as_matrix(mic_positions)
    mic_a_name = args.mic_a or mic_names[0]
    mic_b_name = args.mic_b or mic_names[1]
    mic_a, mic_b = mic_positions[mic_a_name], mic_positions[mic_b_name]
 
    drone_pool = AudioPool(cfg["assets"]["drone_dir"], fs, default_class_name="drone")
    clip = drone_pool.random_clip()
    medium = MediumConditions(temperature_c=20.0, relative_humidity_pct=50.0, pressure_kpa=101.325)
    ground_cfg = GroundReflectionConfig(enabled=False)
 
    print("=" * 70)
    print("CHECK 1: static source")
    print("=" * 70)
 
    duration = 3.0
    n = int(duration * fs)
    sig = np.tile(clip.signal, int(np.ceil(n / len(clip.signal))))[:n]
 
    static_pos = [12.0, 8.0, 20.0]
    static_traj = HoverJitterTrajectory(static_pos, duration, jitter_std_m=0.0)
 
    per_mic = propagate_moving_source_to_array(
        sig, fs, static_traj, 0.0, mic_positions, medium,
        apply_air_absorption=True, ground_reflection=ground_cfg,
    )
    est_tau = gcc_phat(per_mic[mic_a_name], per_mic[mic_b_name], fs, max_tau=0.05)
    true_tau = true_tdoa_seconds(np.array(static_pos), mic_a, mic_b, medium)
    err_samples = abs(est_tau - true_tau) * fs
    print(f"Mic pair          : {mic_a_name} - {mic_b_name}")
    print(f"True TDOA         : {true_tau*1e6:.2f} us")
    print(f"GCC-PHAT estimate : {est_tau*1e6:.2f} us")
    print(f"Error             : {err_samples:.2f} samples")
    print("[PASS]" if err_samples <= 1.5 else "[FAIL]")
 
    print()
    print("=" * 70)
    print("CHECK 2: moving source (linear flyby), TDOA at t=1.5s")
    print("=" * 70)
    print("(Uses an internally-generated broadband noise probe rather than")
    print(" your drone recording: GCC-PHAT is fundamentally ambiguous for a")
    print(" purely periodic/tonal test signal -- a real drone recording's")
    print(" broadband propeller/motor turbulence noise does not have this")
    print(" issue, but a clean synthetic test tone can. This check validates")
    print(" the moving-source propagation physics independent of that.)")
    print()
 
    probe_rng = np.random.default_rng(7)
    probe_sig = probe_rng.standard_normal(n)
 
    start_pos = np.array([-20.0, 5.0, 15.0])
    velocity = np.array([8.0, 0.5, 0.2])   # m/s, realistic drone cruise speed
    moving_traj = LinearFlybyTrajectory(start_pos, velocity, duration)
 
    per_mic_moving = propagate_moving_source_to_array(
        probe_sig, fs, moving_traj, 0.0, mic_positions, medium,
        apply_air_absorption=True, ground_reflection=ground_cfg,
    )
 
    t_probe = 1.5
    window_s = 0.3  # short local window around t_probe for a locally-valid TDOA estimate
    s0 = int((t_probe - window_s / 2) * fs)
    s1 = int((t_probe + window_s / 2) * fs)
    est_tau_moving = gcc_phat(per_mic_moving[mic_a_name][s0:s1], per_mic_moving[mic_b_name][s0:s1],
                               fs, max_tau=0.05)
    true_pos_at_t = moving_traj.position_at(t_probe)
    true_tau_moving = true_tdoa_seconds(true_pos_at_t, mic_a, mic_b, medium)
    err_samples_moving = abs(est_tau_moving - true_tau_moving) * fs
 
    print(f"Position at t={t_probe}s : {true_pos_at_t.round(2).tolist()}")
    print(f"True TDOA               : {true_tau_moving*1e6:.2f} us")
    print(f"GCC-PHAT estimate       : {est_tau_moving*1e6:.2f} us")
    print(f"Error                   : {err_samples_moving:.2f} samples")
    print("[PASS]" if err_samples_moving <= 25.0 else "[FAIL]")
    print()
    print("(Tolerance is looser here: a 0.3s local window blends TDOA over")
    print(" continuous motion -- at 8 m/s the true delay itself shifts by")
    print(" several samples across that window, on top of ordinary")
    print(" finite-window GCC-PHAT estimation noise.)")
 
    print()
    print("=" * 70)
    print("CHECK 3: event with onset > 0 rendered through mixer._render_event")
    print("=" * 70)
    ok3 = check_event_time_alignment(fs, mic_positions, mic_a_name, mic_b_name, medium, ground_cfg)
    print("[PASS]" if ok3 else "[FAIL]")
 
 
if __name__ == "__main__":
    main()
 
