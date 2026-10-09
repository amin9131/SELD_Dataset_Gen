"""
propagation.py
==============
Free-field ("Level 1 + Level 2") point-source propagation model.
 
For a point source with a known signal, position, and a set of
microphone positions, this computes what each microphone receives:
 
  Level 1 (always applied):
    - exact propagation delay   tau = |p_source - p_mic| / v_sound
    - spherical spreading loss  ~ 1 / r
 
  Level 2 (optional, individually togglable):
    - frequency-dependent atmospheric absorption (ISO 9613-1)
    - a single specular ground reflection (image-source method,
      flat ground at z = ground_z, configurable reflection coefficient)
 
Everything is applied in the FREQUENCY DOMAIN via one FFT block per
source clip: this gives sample-accurate (bandlimited) fractional delay,
and lets the frequency-dependent air-absorption be a plain multiplication.
"""
 
from dataclasses import dataclass
from typing import Dict, Optional
 
import numpy as np
 
from atmosphere import speed_of_sound, air_absorption_coefficient
from ground_impedance import plane_wave_reflection_coefficient, resolve_flow_resistivity
 
 
@dataclass
class MediumConditions:
    temperature_c: float = 20.0
    relative_humidity_pct: float = 50.0
    pressure_kpa: float = 101.325
 
    @property
    def v_sound(self) -> float:
        return speed_of_sound(self.temperature_c)
 
 
@dataclass
class GroundReflectionConfig:
    enabled: bool = False
    ground_z: float = 0.0
    model: str = "delany_bazley"   # "delany_bazley" (realistic, frequency+angle dependent)
                                     # or "constant" (legacy: single real scalar, no
                                     # frequency/angle dependence)
    ground_type: str = "grass"      # preset name (see ground_impedance.GROUND_TYPE_PRESETS),
                                     # used when model="delany_bazley" and
                                     # flow_resistivity_kpa_sm2 is not given
    flow_resistivity_kpa_sm2: float = None  # explicit override (kPa*s/m^2); if set,
                                              # takes precedence over `ground_type`
    reflection_coefficient: float = 0.3      # used only when model="constant"
 
 
def propagate_single_path_untrimmed(
    signal: np.ndarray,
    fs: float,
    distance_m: float,
    medium: MediumConditions,
    apply_air_absorption: bool,
    amplitude_scale: float = 1.0,
    extra_freq_response=None,
) -> np.ndarray:
    """
    Exact frequency-domain delay + 1/r spreading + (optional) air
    absorption + (optional) an arbitrary extra frequency response. Returns
    the FULL padded-length result (length n + pad, where pad covers the
    propagation delay), WITHOUT trimming back to the input length.
 
    `extra_freq_response`, if given, is a callable `freqs_hz -> complex
    ndarray` (same shape as the internal rfft frequency bins) that is
    multiplied into the spectrum in addition to delay/spreading/air
    absorption. This is how the realistic (frequency- and angle-dependent)
    ground reflection coefficient is applied -- see
    propagate_point_source_to_array / moving_propagation.py.
 
    This full-length (untrimmed) result is required whenever the
    propagation delay can be comparable to or larger than the signal
    itself -- e.g. when propagating short, per-block chunks of a moving
    source (see moving_propagation.py): trimming to the input length in
    that case would silently discard most or all of the delayed energy,
    since it would land beyond the trimmed window.
    """
    n = len(signal)
    tau = distance_m / medium.v_sound
    pad = int(np.ceil(tau * fs)) + 8  # avoid circular wrap-around
    n_fft = n + pad
 
    spec = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    phase = np.exp(-1j * 2 * np.pi * freqs * tau)
    spreading = amplitude_scale / max(distance_m, 0.05)
 
    if apply_air_absorption:
        alpha = air_absorption_coefficient(
            freqs, medium.temperature_c, medium.relative_humidity_pct, medium.pressure_kpa
        )
        absorption = np.exp(-alpha * distance_m)
    else:
        absorption = 1.0
 
    extra = extra_freq_response(freqs) if extra_freq_response is not None else 1.0
 
    spec_out = spec * phase * spreading * absorption * extra
    return np.fft.irfft(spec_out, n=n_fft)
 
 
def propagate_single_path(
    signal: np.ndarray,
    fs: float,
    distance_m: float,
    medium: MediumConditions,
    apply_air_absorption: bool,
    amplitude_scale: float = 1.0,
    extra_freq_response=None,
) -> np.ndarray:
    """
    Same physics as `propagate_single_path_untrimmed`, but trimmed back to
    the input length -- valid whenever the propagation delay is small
    compared to the signal length (true for a single long/static-source
    clip, which is what this function is used for). For short per-block
    processing of a MOVING source, use `propagate_single_path_untrimmed`
    instead (see moving_propagation.py).
    """
    n = len(signal)
    return propagate_single_path_untrimmed(
        signal, fs, distance_m, medium, apply_air_absorption, amplitude_scale, extra_freq_response
    )[:n]
 
 
def _mirror_source_for_ground(source_pos: np.ndarray, ground_z: float) -> np.ndarray:
    mirrored = source_pos.copy()
    mirrored[2] = 2 * ground_z - source_pos[2]
    return mirrored
 
 
def propagate_point_source_to_array(
    signal: np.ndarray,
    fs: float,
    source_pos: np.ndarray,
    mic_positions: Dict[str, np.ndarray],
    medium: MediumConditions,
    apply_air_absorption: bool = True,
    ground_reflection: Optional[GroundReflectionConfig] = None,
) -> Dict[str, np.ndarray]:
    """Propagate one point-source clip to every microphone. Returns
    {mic_name: received_signal (same length as input `signal`)}."""
    ground_reflection = ground_reflection or GroundReflectionConfig(enabled=False)
    received = {}
    for mic_name, mic_pos in mic_positions.items():
        direct_distance = float(np.linalg.norm(source_pos - mic_pos))
        out = propagate_single_path(signal, fs, direct_distance, medium, apply_air_absorption)
 
        if ground_reflection.enabled:
            image_pos = _mirror_source_for_ground(source_pos, ground_reflection.ground_z)
            ground_distance = float(np.linalg.norm(image_pos - mic_pos))
 
            if ground_reflection.model == "constant":
                reflected = propagate_single_path(
                    signal, fs, ground_distance, medium, apply_air_absorption,
                    amplitude_scale=ground_reflection.reflection_coefficient,
                )
            else:
                h_source = source_pos[2] - ground_reflection.ground_z
                h_mic = mic_pos[2] - ground_reflection.ground_z
                cos_theta = (h_source + h_mic) / max(ground_distance, 1e-6)
                sigma = resolve_flow_resistivity(
                    ground_reflection.ground_type, ground_reflection.flow_resistivity_kpa_sm2
                )
                reflect_fn = lambda freqs, ct=cos_theta, s=sigma: plane_wave_reflection_coefficient(freqs, ct, s)
                reflected = propagate_single_path(
                    signal, fs, ground_distance, medium, apply_air_absorption,
                    amplitude_scale=1.0, extra_freq_response=reflect_fn,
                )
            out = out + reflected
 
        received[mic_name] = out
    return received
 
 
def true_tdoa_seconds(
    source_pos: np.ndarray,
    mic_a: np.ndarray,
    mic_b: np.ndarray,
    medium: MediumConditions,
) -> float:
    """Ground-truth TDOA (tau_ab = t_a - t_b) for a source/mic pair,
    used by validate_gcc_phat.py to sanity-check the pipeline."""
    da = np.linalg.norm(source_pos - mic_a)
    db = np.linalg.norm(source_pos - mic_b)
    return (da - db) / medium.v_sound
