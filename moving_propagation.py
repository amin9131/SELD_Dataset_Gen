"""
moving_propagation.py
======================
Propagation for a MOVING point source (a drone following a Trajectory)
to a microphone array, over an arbitrarily long duration.
 
Approach: block-based overlap-add (OLA). The signal is cut into
overlapping, Hann-windowed blocks (50% overlap, which satisfies the
constant-overlap-add / COLA condition: shifted Hann windows at 50% hop
sum to ~1 everywhere in the interior of the signal). Each block is
propagated using the EXACT instantaneous source position at that
block's center time (reusing propagation.propagate_single_path_untrimmed
-- the same delay + spherical spreading + air-absorption + optional
ground-reflection model used for static sources), and the UNTRIMMED
(full delay-padded) result is added into the output accumulator at the
block's absolute start sample. Using the untrimmed result matters:
propagation delay (tens of milliseconds, for the distances used here)
can easily exceed a single block's duration, so trimming each block's
output back to the block's own length would discard most of the
delayed energy.
 
Because the same linear-phase delay filter is applied per block, and
adjacent blocks use nearly identical delays (drones move only a few
centimeters per ~20 ms hop), the OLA reconstruction is accurate to the
same standard used for slowly-time-varying-filter OLA processing, and
reduces exactly to the correct static-source result when the source
does not move at all (verified in validate_gcc_phat.py).
"""
 
from typing import Dict, Optional
import numpy as np
 
from array_geometry import MicPositions, as_matrix
from propagation import (
    MediumConditions,
    GroundReflectionConfig,
    propagate_single_path_untrimmed,
    _mirror_source_for_ground,
)
from ground_impedance import plane_wave_reflection_coefficient, resolve_flow_resistivity
from trajectory import Trajectory
 
 
def propagate_moving_source_to_array(
    signal: np.ndarray,
    fs: float,
    trajectory: Trajectory,
    t_start: float,
    mic_positions: MicPositions,
    medium: MediumConditions,
    apply_air_absorption: bool = True,
    ground_reflection: Optional[GroundReflectionConfig] = None,
    block_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """
    Propagate `signal` along `trajectory` to every microphone, returning a
    dict of received signals the same length as the input.
 
    `t_start` is the TRAJECTORY-LOCAL time (the time axis of
    `trajectory.position_at`, where t=0 is the start of the trajectory) at
    which `signal[0]` is emitted. For an event whose trajectory spans
    exactly its own clip (as in mixer.py / scenario.py) this is 0.0 -- NOT
    the event's onset in episode time. Passing episode time here makes
    every block clamp to the trajectory's final position (the source
    appears frozen) for any event with onset > 0.
    """
    ground_reflection = ground_reflection or GroundReflectionConfig(enabled=False)
    n = len(signal)
    hop = block_size // 2
    window = np.hanning(block_size)
 
    mic_names, mic_mat = as_matrix(mic_positions)
    block_starts = list(range(0, n, hop))
 
    # --- pass 1: block-center positions, per-mic distances, and the max
    #     delay we'll need to size the accumulation buffer for ---
    block_positions = []
    per_block_distance = {name: [] for name in mic_names}
    per_block_ground_distance = {name: [] for name in mic_names}
    max_delay_samples = 0
 
    for start in block_starts:
        t_center = t_start + (start + block_size / 2.0) / fs
        t_center = min(max(t_center, 0.0), trajectory.duration_s)
        pos = trajectory.position_at(t_center)
        block_positions.append(pos)
 
        for name in mic_names:
            d = float(np.linalg.norm(pos - mic_positions[name]))
            per_block_distance[name].append(d)
            max_delay_samples = max(max_delay_samples, int(np.ceil(d / medium.v_sound * fs)))
 
            if ground_reflection.enabled:
                image_pos = _mirror_source_for_ground(pos, ground_reflection.ground_z)
                dg = float(np.linalg.norm(image_pos - mic_positions[name]))
                per_block_ground_distance[name].append(dg)
                max_delay_samples = max(max_delay_samples, int(np.ceil(dg / medium.v_sound * fs)))
 
    out_len = n + max_delay_samples + block_size + 16
    accum = {name: np.zeros(out_len) for name in mic_names}
 
    # --- pass 2: propagate each windowed block with its own instantaneous
    #     distance, and add the UNTRIMMED (full delay-padded) result at
    #     the block's absolute start sample ---
    for bi, start in enumerate(block_starts):
        end = min(start + block_size, n)
        block = np.zeros(block_size)
        block[: end - start] = signal[start:end]
 
        if (end - start) == block_size:
            win = window
        else:
            # last (partial) block: use the corresponding LEADING portion
            # of the same analysis window, so COLA still holds at the
            # true end of the signal (natural fade-out), rather than
            # fabricating a differently-shaped short window.
            win = np.zeros(block_size)
            win[: end - start] = window[: end - start]
 
        windowed = block * win
 
        for name in mic_names:
            d = per_block_distance[name][bi]
            propagated = propagate_single_path_untrimmed(
                windowed, fs, d, medium, apply_air_absorption
            )
 
            if ground_reflection.enabled:
                dg = per_block_ground_distance[name][bi]
 
                if ground_reflection.model == "constant":
                    reflected = propagate_single_path_untrimmed(
                        windowed, fs, dg, medium, apply_air_absorption,
                        amplitude_scale=ground_reflection.reflection_coefficient,
                    )
                else:
                    h_source = block_positions[bi][2] - ground_reflection.ground_z
                    h_mic = mic_positions[name][2] - ground_reflection.ground_z
                    cos_theta = (h_source + h_mic) / max(dg, 1e-6)
                    sigma = resolve_flow_resistivity(
                        ground_reflection.ground_type, ground_reflection.flow_resistivity_kpa_sm2
                    )
                    reflect_fn = lambda freqs, ct=cos_theta, s=sigma: plane_wave_reflection_coefficient(freqs, ct, s)
                    reflected = propagate_single_path_untrimmed(
                        windowed, fs, dg, medium, apply_air_absorption,
                        amplitude_scale=1.0, extra_freq_response=reflect_fn,
                    )
 
                L = max(len(propagated), len(reflected))
                if len(propagated) < L:
                    propagated = np.pad(propagated, (0, L - len(propagated)))
                if len(reflected) < L:
                    reflected = np.pad(reflected, (0, L - len(reflected)))
                propagated = propagated + reflected
 
            L = len(propagated)
            accum[name][start:start + L] += propagated
 
    return {name: accum[name][:n] for name in mic_names}
 
