"""
atmosphere.py
=============
Physical models for the propagation medium (air):
  - Speed of sound as a function of temperature
  - Frequency-dependent atmospheric absorption (ISO 9613-1 model)
 
These are the "Level 2" physics additions on top of plain free-field
delay + spherical spreading.
 
Reference: ISO 9613-1:1993 "Acoustics -- Attenuation of sound during
propagation outdoors -- Part 1: Calculation of the absorption of sound
by the atmosphere."
"""
 
import numpy as np
 
 
def speed_of_sound(temperature_c: float) -> float:
    """
    Speed of sound in air (m/s) as a function of temperature (Celsius).
    v = 331.3 + 0.606 * T
    """
    return 331.3 + 0.606 * temperature_c
 
 
def _saturation_pressure_ratio(temperature_k: float) -> float:
    """Ratio p_sat / p_r used in the ISO 9613-1 humidity calculation."""
    T01 = 273.16  # triple point temperature of water, K
    C = -6.8346 * (T01 / temperature_k) ** 1.261 + 4.6151
    return 10.0 ** C
 
 
def air_absorption_coefficient(
    freqs_hz: np.ndarray,
    temperature_c: float = 20.0,
    relative_humidity_pct: float = 50.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """
    Atmospheric absorption coefficient alpha(f) in Nepers/meter (ISO 9613-1).
    Apply as exp(-alpha * r) to get amplitude attenuation over distance r.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    freqs_hz = np.clip(freqs_hz, 1e-6, None)
 
    T = temperature_c + 273.15
    T0 = 293.15
    pa_pr = pressure_kpa / 101.325
 
    psat_pr = _saturation_pressure_ratio(T)
    h = relative_humidity_pct * psat_pr / pa_pr  # molar concentration, %
 
    Fro = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    Frn = pa_pr * (T / T0) ** (-0.5) * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0))
    )
 
    f2 = freqs_hz ** 2
    term_classical = 1.84e-11 * (1.0 / pa_pr) * (T / T0) ** 0.5
    # ISO 9613-1 eq. (5): the 0.01275 / 2239.1 K pair belongs to OXYGEN
    # (relaxation frequency frO), the 0.1068 / 3352 K pair to NITROGEN (frN).
    # Fr/(f^2+Fr^2) == 1/(Fr + f^2/Fr).
    term_rot_vib = (T / T0) ** (-2.5) * (
        0.01275 * np.exp(-2239.1 / T) * (Fro / (f2 + Fro ** 2))
        + 0.1068 * np.exp(-3352.0 / T) * (Frn / (f2 + Frn ** 2))
    )
 
    alpha_db_per_m = 8.686 * f2 * (term_classical + term_rot_vib)
    return alpha_db_per_m / 8.686  # dB/m -> Nepers/m
