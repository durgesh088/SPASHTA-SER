"""
Signal Preprocessing Pipeline for Hindi SER:
  1. Kalman Filter — Noise reduction
  2. NLMS with FIR — Echo cancellation
  3. Wiener Filter — Signal smoothing
"""

import numpy as np
from scipy.signal import stft, istft

SAMPLE_RATE = 16000


try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ── Kalman Filter ──
if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _kalman_fast(signal, process_noise, measurement_noise):
        n = len(signal)
        filtered = np.zeros(n)
        x_est = 0.0
        p_est = 1.0
        for i in range(n):
            x_pred = x_est
            p_pred = p_est + process_noise
            k = p_pred / (p_pred + measurement_noise)
            x_est = x_pred + k * (signal[i] - x_pred)
            p_est = (1 - k) * p_pred
            filtered[i] = x_est
        return filtered
else:
    def _kalman_fast(signal, process_noise, measurement_noise):
        n = len(signal)
        filtered = np.zeros(n)
        x_est = 0.0
        p_est = 1.0
        for i in range(n):
            x_pred = x_est
            p_pred = p_est + process_noise
            k = p_pred / (p_pred + measurement_noise)
            x_est = x_pred + k * (signal[i] - x_pred)
            p_est = (1 - k) * p_pred
            filtered[i] = x_est
        return filtered

def kalman_filter(signal, process_noise=1e-3, measurement_noise=1e-5):
    """Apply 1D Kalman filter for noise reduction (optimized fast version)."""
    return _kalman_fast(signal, process_noise, measurement_noise)


# ── NLMS Filter ──
if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _nlms_fast(signal, reference, filter_order, mu, eps):
        n = len(signal)
        w = np.zeros(filter_order)
        output = np.zeros(n)
        for i in range(filter_order, n):
            y_est = 0.0
            norm = 0.0
            for j in range(filter_order):
                val = reference[i - 1 - j]
                y_est += w[j] * val
                norm += val * val
            
            error = signal[i] - y_est
            norm += eps
            
            for j in range(filter_order):
                w[j] += (mu / norm) * error * reference[i - 1 - j]
            
            output[i] = error
        return output
else:
    def _nlms_fast(signal, reference, filter_order, mu, eps):
        n = len(signal)
        w = np.zeros(filter_order)
        output = np.zeros(n)
        for i in range(filter_order, n):
            x = reference[i - filter_order:i][::-1]
            y_est = np.dot(w, x)
            error = signal[i] - y_est
            norm = np.dot(x, x) + eps
            w = w + (mu / norm) * error * x
            output[i] = error
        return output

def nlms_filter(signal, filter_order=32, mu=1e-5, eps=1e-10):
    """Apply Normalized Least Mean Square (NLMS) adaptive filter (optimized fast version)."""
    n = len(signal)
    delay = filter_order
    reference = np.zeros(n)
    reference[delay:] = signal[:-delay]
    
    output = _nlms_fast(signal, reference, filter_order, mu, eps)
    output[:filter_order] = signal[:filter_order]
    return output


def wiener_filter(signal, sr=SAMPLE_RATE, noise_frames=2,
                  n_fft=512, hop_length=256):
    """
    Apply Wiener filter in the spectral domain for signal smoothing.

    Estimates the noise power spectral density from the first few
    (assumed silent / low-energy) STFT frames, then applies the
    Wiener gain  G(f) = max(1 − N(f)/S(f), floor)  to suppress
    noise while preserving speech.

    Parameters
    ----------
    signal : np.ndarray
        Input audio waveform (1-D).
    sr : int
        Sample rate.
    noise_frames : int
        Number of leading STFT frames used to estimate noise PSD.
    n_fft : int
        FFT window size.
    hop_length : int
        STFT hop length.

    Returns
    -------
    np.ndarray
        Wiener-filtered signal (same length as input).
    """
    # Forward STFT
    f, t, Zxx = stft(signal, fs=sr, nperseg=n_fft,
                     noverlap=n_fft - hop_length)

    power_spec = np.abs(Zxx) ** 2

    # Estimate noise power from leading frames
    noise_power = np.mean(power_spec[:, :noise_frames], axis=1,
                          keepdims=True)

    # Wiener gain (floored at 0.1 to avoid spectral subtraction distortion)
    gain = np.maximum(1.0 - noise_power / (power_spec + 1e-10), 0.1)

    # Apply gain and invert
    Zxx_filtered = Zxx * gain
    _, filtered = istft(Zxx_filtered, fs=sr, nperseg=n_fft,
                        noverlap=n_fft - hop_length)

    # Match original length
    if len(filtered) < len(signal):
        filtered = np.pad(filtered, (0, len(signal) - len(filtered)))
    else:
        filtered = filtered[:len(signal)]

    return filtered


# ──────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────

def preprocess_audio(signal, sr=SAMPLE_RATE):
    """
    Full three-stage preprocessing cascade.

    1. Kalman filter  → noise reduction
    2. NLMS + FIR     → echo cancellation
    3. Wiener filter  → spectral smoothing

    Parameters
    ----------
    signal : np.ndarray
        Raw audio waveform.
    sr : int
        Sample rate.

    Returns
    -------
    np.ndarray
        Preprocessed, peak-normalised audio signal.
    """
    signal = kalman_filter(signal, process_noise=1e-3, measurement_noise=1e-5)
    signal = nlms_filter(signal, filter_order=32, mu=1e-5)
    signal = wiener_filter(signal, sr=sr, noise_frames=2)

    # Peak-normalise
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return signal

