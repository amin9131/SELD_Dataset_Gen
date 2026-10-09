"""
diffuse_field.py
=================
Spatially-coherent diffuse noise field synthesis.
 
Real outdoor ambient/wind noise arriving at a microphone array is NOT
composed of statistically independent samples at each microphone -- it
has genuine spatial coherence structure: nearby microphones pick up
strongly correlated noise (it's the same physical wavefronts, arriving
from many directions at once), while far-apart microphones see
progressively less correlated noise. The standard reference model for
this is the "spherically isotropic diffuse field" coherence law (Cook,
Waterhouse, Berendt, Edelman & Thompson, 1955, JASA 27(6); a staple of
array-processing and beamforming literature, e.g. Bitzer & Simmer's
beamforming chapter in "Microphone Arrays"):
 
    Gamma_ij(f) = sinc(2 f d_ij / c)          [sinc(x) = sin(pi x)/(pi x)]
 
where d_ij is the distance between microphones i and j and c is the
speed of sound. This matters directly for evaluating array-processing
algorithms (MVDR, MUSIC, GCC-PHAT, ...): a spatial filter's ability to
suppress noise while preserving a coherent target signal depends
critically on how coherent the noise itself is across the array.
Independent-per-microphone noise (coherence = 0 for every pair, at every
frequency) is an artificially easy special case that can make a
beamformer look better than it would on real recordings, where nearby
microphone pairs actually see substantially correlated wind/ambient noise
at low frequencies.
 
Synthesis method: the standard "mixing matrix" approach of Habets, Cohen
& Gannot (2008), "Generating nonstationary multisensor signals under a
spatial coherence constraint," J. Acoust. Soc. Am. 124(5), 2911-2917.
Independent per-microphone noise realizations are drawn from REAL
recorded audio (fit_to_duration'd clips from the wind/ambient pools --
this framework never synthesizes noise from scratch), then mixed per
STFT frequency bin with a matrix whose Gram matrix reproduces the target
coherence, and inverse-STFT'd back to the time domain. The result keeps
the spectral "texture" of the real recordings while imposing the correct
physical spatial-coherence structure for this specific microphone array.
"""
 
from typing import Dict, List
import numpy as np
from scipy.signal import stft, istft
 
 
def spherical_diffuse_coherence(freqs_hz: np.ndarray, distance_m: float, v_sound: float) -> np.ndarray:
    """
    Theoretical coherence Gamma(f) between two microphones separated by
    `distance_m`, in a spherically isotropic diffuse noise field.
    """
    x = 2.0 * np.asarray(freqs_hz, dtype=np.float64) * distance_m / v_sound
    return np.sinc(x)  # np.sinc(x) = sin(pi*x)/(pi*x) -- matches sin(u)/u with u = pi*x
 
 
def _coherence_matrices(freqs_hz: np.ndarray, mic_mat: np.ndarray, v_sound: float) -> np.ndarray:
    """Returns Gamma, shape (n_freqs, M, M): the target coherence matrix
    at every STFT frequency bin, for this specific microphone geometry."""
    M = mic_mat.shape[0]
    dist = np.linalg.norm(mic_mat[:, None, :] - mic_mat[None, :, :], axis=-1)  # (M, M)
 
    n_freqs = len(freqs_hz)
    Gamma = np.empty((n_freqs, M, M))
    for i in range(M):
        for j in range(M):
            Gamma[:, i, j] = spherical_diffuse_coherence(freqs_hz, dist[i, j], v_sound)
    return Gamma
 
 
def _mixing_matrices(Gamma: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Per-frequency-bin mixing matrices C(k) such that C(k) @ C(k).T
    reproduces Gamma(k) (Cholesky decomposition of the regularized
    coherence matrix; falls back to eigenvalue clipping on the rare bins
    where Cholesky fails, e.g. near sinc zero-crossings that make Gamma(k)
    numerically singular).
    """
    n_freqs, M, _ = Gamma.shape
    C = np.zeros_like(Gamma)
    I = np.eye(M)
    for k in range(n_freqs):
        G = Gamma[k] + reg * I
        try:
            C[k] = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(G)
            w = np.clip(w, 1e-10, None)
            C[k] = V @ np.diag(np.sqrt(w))
    return C
 
 
def generate_diffuse_field(
    mic_names: List[str],
    mic_mat: np.ndarray,
    independent_signals: np.ndarray,
    fs: float,
    v_sound: float,
    nperseg: int = 1024,
) -> Dict[str, np.ndarray]:
    """
    Mixes M independent per-microphone noise realizations (real-audio
    derived) into M spatially coherent output signals matching the
    spherically isotropic diffuse-field coherence law for this specific
    microphone geometry.
 
    Parameters
    ----------
    independent_signals : np.ndarray, shape (M, n_samples)
        One mutually-independent noise realization per microphone (e.g.
        M different random clips/segments drawn from the same noise
        pool), all the same length.
    """
    M, n_samples = independent_signals.shape
    _, _, Zxx = stft(independent_signals, fs=fs, nperseg=nperseg, axis=-1)
    # Zxx shape: (M, n_freqs, n_frames)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
 
    Gamma = _coherence_matrices(freqs, mic_mat, v_sound)  # (n_freqs, M, M)
    C = _mixing_matrices(Gamma)                            # (n_freqs, M, M)
 
    n_freqs = Zxx.shape[1]
    mixed = np.zeros_like(Zxx)
    for k in range(n_freqs):
        mixed[:, k, :] = C[k] @ Zxx[:, k, :]
 
    _, out = istft(mixed, fs=fs, nperseg=nperseg)
 
    if out.shape[1] < n_samples:
        out = np.pad(out, ((0, 0), (0, n_samples - out.shape[1])))
    else:
        out = out[:, :n_samples]
 
    return {name: out[i] for i, name in enumerate(mic_names)}
