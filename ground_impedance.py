"""
ground_impedance.py
====================
Frequency- and angle-dependent acoustic ground reflection, using the
Delany-Bazley (1970) empirical impedance model for a locally-reacting,
porous ground surface (grass, soil, snow, gravel, asphalt, ...).
 
This replaces a single constant "reflection coefficient" with a
physically grounded model: harder/denser ground (asphalt, concrete)
reflects strongly and almost non-dispersively across the whole band;
soft porous ground (grass, forest floor, fresh snow) absorbs more, and
does so differently at different frequencies and at different incidence
angles -- exactly the frequency-tilted, angle-dependent "muffled echo"
character a real outdoor ground bounce has, which a single real-valued
scalar cannot reproduce.
 
Reference: Delany, M.E. and Bazley, E.N. (1970), "Acoustical properties
of fibrous absorbent materials", Applied Acoustics 3(2), 105-116. This
empirical model (and its many refinements) is the standard starting
point for outdoor ground-effect modeling in the acoustics literature
(see e.g. Attenborough & Embleton's outdoor sound propagation work, and
the ground-effect term in ISO 9613-2).
"""
 
import numpy as np
 
# Typical effective flow resistivity presets (kPa*s/m^2), approximate
# values compiled from commonly-cited outdoor-acoustics ground tables.
# Larger = harder/denser ground = closer to a perfectly rigid reflector
# (the model correctly approaches |R| -> 1 as flow resistivity -> infinity).
GROUND_TYPE_PRESETS = {
    "snow_fresh": 20.0,
    "forest_floor": 40.0,
    "soil_soft": 80.0,
    "grass": 250.0,
    "sand": 300.0,
    "soil_compacted": 800.0,
    "gravel": 2000.0,
    "asphalt": 5.0e5,
    "concrete": 2.0e6,
    "rigid": 1.0e8,   # numerically "perfectly hard": |R| stays close to 1 across the whole band
}
 
 
def delany_bazley_impedance(freqs_hz: np.ndarray, flow_resistivity_kpa_sm2: float) -> np.ndarray:
    """
    Normalized surface impedance zeta = Zc / Z0 (complex, dimensionless)
    of a porous ground layer, via the Delany-Bazley empirical model.
 
    Valid (as an engineering approximation) roughly over 0.01 < X < 1.0,
    where X = rho0*f/sigma -- comfortably covering typical ground flow
    resistivities and audio-band frequencies used in this framework.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    freqs_hz = np.clip(freqs_hz, 1e-6, None)
 
    rho0 = 1.2  # kg/m^3, air density (matches the level of approximation
                # already used elsewhere in this framework for this effect)
    sigma = flow_resistivity_kpa_sm2 * 1000.0  # kPa*s/m^2 -> Pa*s/m^2 (mks rayls/m)
 
    X = rho0 * freqs_hz / sigma
    zeta = 1.0 + 0.0571 * X ** (-0.754) - 1j * 0.087 * X ** (-0.732)
    return zeta
 
 
def plane_wave_reflection_coefficient(
    freqs_hz: np.ndarray,
    cos_theta: float,
    flow_resistivity_kpa_sm2: float,
) -> np.ndarray:
    """
    Complex plane-wave reflection coefficient R(f) for a locally-reacting
    impedance ground, at incidence angle theta measured from the ground
    NORMAL (cos_theta = 1 is straight-down/normal incidence, cos_theta
    near 0 is grazing incidence):
 
        R(f) = (zeta(f) * cos_theta - 1) / (zeta(f) * cos_theta + 1)
 
    Returns a complex array (same shape as freqs_hz): |R(f)| gives the
    frequency-dependent amplitude loss on reflection, and the phase of
    R(f) gives the (generally non-trivial, frequency-dependent) phase
    shift the ground bounce introduces to each frequency component --
    both effects a single constant real coefficient cannot reproduce.
    """
    zeta = delany_bazley_impedance(freqs_hz, flow_resistivity_kpa_sm2)
    cos_theta = float(np.clip(cos_theta, 1e-3, 1.0))  # guard grazing/degenerate angles
    return (zeta * cos_theta - 1.0) / (zeta * cos_theta + 1.0)
 
 
def resolve_flow_resistivity(ground_type: str, flow_resistivity_kpa_sm2: float = None) -> float:
    """
    Returns the flow resistivity to use: an explicit override if given,
    otherwise the preset for `ground_type`.
    """
    if flow_resistivity_kpa_sm2 is not None:
        return float(flow_resistivity_kpa_sm2)
    if ground_type not in GROUND_TYPE_PRESETS:
        raise ValueError(
            f"Unknown ground_type '{ground_type}'. "
            f"Available presets: {sorted(GROUND_TYPE_PRESETS)}"
        )
    return GROUND_TYPE_PRESETS[ground_type]
 
