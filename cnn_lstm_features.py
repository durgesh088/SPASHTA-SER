"""
523-Dimensional Feature Extraction for Hindi SER — 16 feature families.

Dimension breakdown (523 total)
───────────────────────────────
 1. MFCC (13) + Delta (13) + Delta² (13)  × 5 stats  = 195
 2. Mel Spectrogram (40)                   × 2 stats  =  80
 3. Spectral Contrast (7)                  × 2 stats  =  14
 4. Pitch / F0  (6 scalar stats)                      =   6
 5. Chroma (12)                            × 5 stats  =  60
 6. Zero Crossing Rate (1)                 × 5 stats  =   5
 7. LPCC (13)                              × 3 stats  =  39
 8. Tonnetz (6)                            × 3 stats  =  18
 9. Formant freq (4) + bw (4)              × 2 stats  =  16
10. Jitter  (5 measures)                               =   5
11. Shimmer (6 measures)                               =   6
12. Entropy: spectral (4 stats) + Shannon (1)          =   5
13. Duration                                           =   1
14. HNR (4 stats)                                      =   4
15. PLP (13)                               × 5 stats  =  65
16. RMS Energy (1)                         × 4 stats  =   4
                                                      ─────
                                                        523
"""

import numpy as np
import librosa
from scipy.fft import fft, ifft

SAMPLE_RATE = 16000
DURATION = 4  # seconds


# ════════════════════════════════════════════════════════════
#  Helper utilities
# ════════════════════════════════════════════════════════════

def _safe_stats(arr, stats=("mean", "std")):
    """Compute summary statistics over a 1-D array, returning zeros
    for empty / all-NaN inputs."""
    if arr is None or len(arr) == 0:
        return np.zeros(len(stats))
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.zeros(len(stats))
    result = []
    for s in stats:
        if s == "mean":
            result.append(np.mean(arr))
        elif s == "std":
            result.append(np.std(arr))
        elif s == "max":
            result.append(np.max(arr))
        elif s == "min":
            result.append(np.min(arr))
        elif s == "median":
            result.append(np.median(arr))
    return np.array(result)


def _pool_matrix(matrix, stats=("mean", "std")):
    """Pool a 2-D matrix (features × time) into a 1-D vector by
    computing row-wise statistics."""
    if matrix is None or matrix.size == 0:
        # Can't determine number of rows; return zeros for safety
        return np.array([])
    results = []
    for s in stats:
        if s == "mean":
            results.append(np.mean(matrix, axis=1))
        elif s == "std":
            results.append(np.std(matrix, axis=1))
        elif s == "max":
            results.append(np.max(matrix, axis=1))
        elif s == "min":
            results.append(np.min(matrix, axis=1))
        elif s == "median":
            results.append(np.median(matrix, axis=1))
    return np.hstack(results)


# ════════════════════════════════════════════════════════════
#  Individual feature extractors
# ════════════════════════════════════════════════════════════

# 1. MFCC + Deltas ──────────────────────────────────────────
def _extract_mfcc(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return mfcc, delta, delta2


# 2. Mel Spectrogram ────────────────────────────────────────
def _extract_mel(audio, sr, n_mels=40):
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)


# 3. Spectral Contrast ─────────────────────────────────────
def _extract_contrast(audio, sr):
    return librosa.feature.spectral_contrast(y=audio, sr=sr)


# 4. Pitch (F0) ─────────────────────────────────────────────
def _extract_pitch(audio, sr):
    """Returns (6-d scalar stats, full f0 contour for jitter/shimmer)."""
    try:
        f0 = librosa.yin(
            audio,
            fmin=60,
            fmax=500,
            sr=sr,
            hop_length=1024
        )
        f0_valid = f0[f0 > 0]
    except Exception:
        f0 = None
        f0_valid = np.array([])

    if len(f0_valid) == 0:
        return np.zeros(6), f0 if f0 is not None else np.array([])
    stats = np.array([
        np.mean(f0_valid),
        np.std(f0_valid),
        np.max(f0_valid),
        np.min(f0_valid),
        np.max(f0_valid) - np.min(f0_valid),   # range
        np.median(f0_valid),
    ])
    return stats, f0


# 5. Chroma ─────────────────────────────────────────────────
def _extract_chroma(audio, sr):
    return librosa.feature.chroma_stft(y=audio, sr=sr)


# 6. Zero Crossing Rate ────────────────────────────────────
def _extract_zcr(audio):
    return librosa.feature.zero_crossing_rate(audio)


# 7. LPCC ───────────────────────────────────────────────────
def _extract_lpcc(audio, sr, order=13, n_lpcc=13,
                  frame_length=512, hop_length=512):
    """Frame-level Linear Prediction Cepstral Coefficients."""
    frames = librosa.util.frame(audio, frame_length=frame_length,
                                hop_length=hop_length)
    n_frames = frames.shape[1]
    lpcc_mat = np.zeros((n_lpcc, n_frames))

    win = np.hanning(frame_length)
    for i in range(n_frames):
        frame = frames[:, i] * win
        if np.sum(np.abs(frame)) < 1e-10:
            continue
        try:
            a = librosa.lpc(frame + 1e-10 * np.random.randn(len(frame)),
                            order=order)
            cep = np.zeros(n_lpcc)
            cep[0] = np.log(np.sum(frame ** 2) / len(frame) + 1e-10)
            for n in range(1, min(n_lpcc, order + 1)):
                cep[n] = -a[n] if n < len(a) else 0.0
                for k in range(1, n):
                    ak = a[n - k] if (n - k) < len(a) else 0.0
                    cep[n] -= (k / n) * cep[k] * ak
            lpcc_mat[:, i] = cep
        except Exception:
            pass
    return lpcc_mat


# 8. Tonnetz ────────────────────────────────────────────────
def _extract_tonnetz(audio, sr):
    try:
        return librosa.feature.tonnetz(y=audio, sr=sr)
    except Exception:
        return np.zeros((6, 1))


# 9. Formant (F1-F4 frequency + bandwidth) ─────────────────
def _extract_formants(audio, sr, n_formants=4,
                      frame_length=512, hop_length=512):
    frames = librosa.util.frame(audio, frame_length=frame_length,
                                hop_length=hop_length)
    n_fr = frames.shape[1]
    freqs_mat = np.zeros((n_formants, n_fr))
    bws_mat = np.zeros((n_formants, n_fr))
    lpc_order = 2 * n_formants + 2
    win = np.hanning(frame_length)

    for i in range(n_fr):
        frame = frames[:, i] * win
        if np.sum(np.abs(frame)) < 1e-10:
            continue
        try:
            # Pre-emphasis
            frame_pe = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
            a = librosa.lpc(frame_pe + 1e-10 * np.random.randn(len(frame_pe)),
                            order=lpc_order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) > 0]
            if len(roots) == 0:
                continue

            angles = np.arctan2(np.imag(roots), np.real(roots))
            f = angles * (sr / (2 * np.pi))
            b = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + 1e-10)

            idx = np.argsort(f)
            f, b = f[idx], b[idx]
            valid = (f > 90) & (f < 5500) & (b < 500)
            f, b = f[valid], b[valid]

            for j in range(min(n_formants, len(f))):
                freqs_mat[j, i] = f[j]
                bws_mat[j, i] = b[j]
        except Exception:
            pass

    return freqs_mat, bws_mat


# 10. Jitter (5 measures) ──────────────────────────────────
def _extract_jitter(f0):
    f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0v) < 3:
        return np.zeros(5)

    T = 1.0 / (f0v + 1e-10)
    n = len(T)
    mu = np.mean(T)
    d = np.abs(np.diff(T))

    # local
    j_local = np.mean(d) / (mu + 1e-10)
    # absolute (seconds)
    j_abs = np.mean(d)
    # RAP (3-point)
    rap = [np.abs(T[i] - np.mean(T[i - 1:i + 2])) for i in range(1, n - 1)]
    j_rap = np.mean(rap) / (mu + 1e-10) if rap else 0.0
    # PPQ5
    if n >= 5:
        ppq = [np.abs(T[i] - np.mean(T[i - 2:i + 3])) for i in range(2, n - 2)]
        j_ppq5 = np.mean(ppq) / (mu + 1e-10)
    else:
        j_ppq5 = 0.0
    # DDP
    ddp = [np.abs(2 * T[i] - T[i - 1] - T[i + 1]) for i in range(1, n - 1)]
    j_ddp = np.mean(ddp) / (mu + 1e-10) if ddp else 0.0

    return np.array([j_local, j_abs, j_rap, j_ppq5, j_ddp])


# 11. Shimmer (6 measures) ─────────────────────────────────
def _extract_shimmer(audio, f0, sr):
    hop = 512
    f0v_mask = ~np.isnan(f0) if f0 is not None else np.array([])
    if not np.any(f0v_mask):
        return np.zeros(6)

    amps = []
    for i in range(len(f0)):
        if not np.isnan(f0[i]):
            s = i * hop
            e = min(s + hop, len(audio))
            if s < len(audio):
                amps.append(np.max(np.abs(audio[s:e])))
    A = np.array(amps)
    if len(A) < 3:
        return np.zeros(6)

    n = len(A)
    mu = np.mean(A)
    d = np.abs(np.diff(A))

    s_local = np.mean(d) / (mu + 1e-10)

    db_d = []
    for i in range(n - 1):
        if A[i] > 0 and A[i + 1] > 0:
            db_d.append(np.abs(20 * np.log10(A[i + 1] / A[i])))
    s_db = np.mean(db_d) if db_d else 0.0

    # APQ-k helper
    def apq(k):
        half = k // 2
        if n < k:
            return 0.0
        vals = [np.abs(A[i] - np.mean(A[i - half:i + half + 1]))
                for i in range(half, n - half)]
        return np.mean(vals) / (mu + 1e-10) if vals else 0.0

    s_apq3 = apq(3)
    s_apq5 = apq(5)
    s_apq11 = apq(11)
    s_dda = s_apq3 * 3

    return np.array([s_local, s_db, s_apq3, s_apq5, s_apq11, s_dda])


# 12. Entropy (spectral + Shannon) ─────────────────────────
def _extract_entropy(audio, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(audio, n_fft=n_fft,
                            hop_length=hop_length)) ** 2
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    sp_ent = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)

    # Shannon entropy of amplitude histogram
    hist, _ = np.histogram(audio, bins=256, density=True)
    hist = hist[hist > 0]
    sh_ent = -np.sum(hist * np.log2(hist + 1e-10))

    return sp_ent, sh_ent


# 14. HNR ───────────────────────────────────────────────────
def _extract_hnr(audio, sr):
    try:
        # Compute short-time autocorrelation
        fl, hl = 2048, 512
        frames = librosa.util.frame(audio, frame_length=fl, hop_length=hl)
        n_frames = frames.shape[1]
        hnr = np.zeros(n_frames)
        min_lag, max_lag = int(sr / 500), int(sr / 50)
        
        for i in range(n_frames):
            frame = frames[:, i]
            r = np.correlate(frame, frame, mode='full')[fl-1:]
            peak_idx = np.argmax(r[min_lag:max_lag]) + min_lag
            r_peak = r[peak_idx]
            r_zero = r[0]
            if r_zero > 1e-10 and r_peak < r_zero:
                hnr[i] = 10 * np.log10(r_peak / (r_zero - r_peak + 1e-10) + 1e-10)
        return hnr
    except Exception:
        return np.array([0.0])


# 15. PLP ───────────────────────────────────────────────────
def _extract_plp(audio, sr, n_plp=13, frame_length=512, hop_length=512):
    """Simplified Perceptual Linear Prediction (Hermansky 1990)."""
    frames = librosa.util.frame(audio, frame_length=frame_length,
                                hop_length=hop_length)
    n_fr = frames.shape[1]
    plp_mat = np.zeros((n_plp, n_fr))
    win = np.hanning(frame_length)

    for i in range(n_fr):
        frame = frames[:, i] * win
        if np.sum(np.abs(frame)) < 1e-10:
            continue
        try:
            # Power spectrum
            power = np.abs(np.fft.rfft(frame)) ** 2

            # Bark-scale warping
            n_bark = 20
            freqs = np.linspace(0, sr / 2, len(power))
            bark = 6 * np.arcsinh(freqs / 600)
            centres = np.linspace(1, np.max(bark), n_bark)
            bw = 1.5
            filt = np.array([np.maximum(0, 1 - np.abs(bark - c) / bw)
                             for c in centres])
            bark_spec = filt @ power

            # Equal-loudness pre-emphasis
            hz_c = 600 * np.sinh(centres / 6)
            f2 = hz_c ** 2
            eq = (f2 / (f2 + 1.6e5)) * ((f2 + 1.44e6) / (f2 + 9.61e6))
            bark_spec *= eq

            # Cube-root compression
            bark_spec = np.power(np.maximum(bark_spec, 0), 1.0 / 3.0)

            # Autocorrelation → LPC → cepstral
            ac = np.real(ifft(np.abs(fft(bark_spec, n=64)) ** 2)
                         )[:n_plp + 1]
            if ac[0] <= 0:
                continue

            # Levinson-Durbin
            a = np.zeros(n_plp + 1)
            a[0] = 1.0
            err = ac[0]
            for m in range(1, n_plp + 1):
                refl = ac[m]
                for j in range(1, m):
                    refl -= a[j] * ac[m - j]
                refl /= (err + 1e-10)
                a_new = a.copy()
                a_new[m] = refl
                for j in range(1, m):
                    a_new[j] = a[j] - refl * a[m - j]
                a = a_new
                err *= (1 - refl ** 2)

            cep = np.zeros(n_plp)
            cep[0] = np.log(err + 1e-10)
            for n in range(1, n_plp):
                cep[n] = a[n]
                for k in range(1, n):
                    cep[n] += (k / n) * cep[k] * a[n - k]
            plp_mat[:, i] = cep
        except Exception:
            pass

    return plp_mat


# 16. RMS Energy ────────────────────────────────────────────
def _extract_energy(audio, sr):
    return librosa.feature.rms(y=audio)


# ════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════

def extract_features_cnn_lstm(audio, sr, pool=True):
    """
    Extract the full 523-dimensional feature vector (when pooled).

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (will be padded / cropped to DURATION seconds).
    sr : int
        Sample rate.
    pool : bool
        If True  → returns 1-D vector of length 523.
        If False → returns dict of raw 2-D matrices plus scalar arrays
                   (needed by SPASHTA perturbation loop).

    Returns
    -------
    np.ndarray or dict
    """
    # Pad / crop to fixed length
    target_len = sr * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # ── Extract everything ──
    mfcc, mfcc_d, mfcc_d2 = _extract_mfcc(audio, sr)
    mel = _extract_mel(audio, sr)
    contrast = _extract_contrast(audio, sr)
    pitch_stats, f0 = _extract_pitch(audio, sr)
    chroma = _extract_chroma(audio, sr)
    zcr = _extract_zcr(audio)
    lpcc = _extract_lpcc(audio, sr)
    tonnetz = _extract_tonnetz(audio, sr)
    fmt_freq, fmt_bw = _extract_formants(audio, sr)
    jitter = _extract_jitter(f0)
    shimmer = _extract_shimmer(audio, f0, sr)
    sp_ent, sh_ent = _extract_entropy(audio, sr)
    hnr = _extract_hnr(audio, sr)
    plp = _extract_plp(audio, sr)
    energy = _extract_energy(audio, sr)
    duration_feat = np.array([len(audio) / sr])

    if not pool:
        return {
            # 2-D temporal matrices (features × frames)
            "mfcc": mfcc,
            "mfcc_delta": mfcc_d,
            "mfcc_delta2": mfcc_d2,
            "mel": mel,
            "contrast": contrast,
            "chroma": chroma,
            "zcr": zcr,
            "lpcc": lpcc,
            "tonnetz": tonnetz,
            "formant_freqs": fmt_freq,
            "formant_bws": fmt_bw,
            "spectral_entropy": sp_ent.reshape(1, -1),
            "hnr": hnr.reshape(1, -1) if hnr.ndim == 1 else hnr,
            "plp": plp,
            "energy": energy,
            # Scalar / utterance-level features
            "pitch_stats": pitch_stats,
            "jitter": jitter,
            "shimmer": shimmer,
            "duration": duration_feat,
            "shannon_entropy": np.array([sh_ent]),
        }

    # ── Pool into 523-D vector ──
    vec = np.hstack([
        # 1  MFCC + deltas   13×5×3 = 195
        _pool_matrix(mfcc,     ("mean", "std", "max", "min", "median")),
        _pool_matrix(mfcc_d,   ("mean", "std", "max", "min", "median")),
        _pool_matrix(mfcc_d2,  ("mean", "std", "max", "min", "median")),

        # 2  Mel spectrogram  40×2 = 80
        _pool_matrix(mel, ("mean", "std")),

        # 3  Spectral contrast 7×2 = 14
        _pool_matrix(contrast, ("mean", "std")),

        # 4  Pitch (F0)       6
        pitch_stats,

        # 5  Chroma           12×5 = 60
        _pool_matrix(chroma, ("mean", "std", "max", "min", "median")),

        # 6  ZCR              1×5 = 5
        _pool_matrix(zcr, ("mean", "std", "max", "min", "median")),

        # 7  LPCC             13×3 = 39
        _pool_matrix(lpcc, ("mean", "std", "max")),

        # 8  Tonnetz          6×3 = 18
        _pool_matrix(tonnetz, ("mean", "std", "max")),

        # 9  Formant          (4+4)×2 = 16
        _pool_matrix(fmt_freq, ("mean", "std")),
        _pool_matrix(fmt_bw,   ("mean", "std")),

        # 10 Jitter           5
        jitter,

        # 11 Shimmer          6
        shimmer,

        # 12 Entropy          sp(4) + Sh(1) = 5
        _safe_stats(sp_ent, ("mean", "std", "max", "median")),
        np.array([sh_ent]),

        # 13 Duration         1
        duration_feat,

        # 14 HNR              4
        _safe_stats(hnr, ("mean", "std", "max", "min")),

        # 15 PLP              13×5 = 65
        _pool_matrix(plp, ("mean", "std", "max", "min", "median")),

        # 16 RMS Energy       1×4 = 4
        _pool_matrix(energy, ("mean", "std", "max", "min")),
    ])

    # Replace any NaN / Inf with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec


# ════════════════════════════════════════════════════════════
#  Standalone Feature Extraction CLI
# ════════════════════════════════════════════════════════════

def _match_time_dim(arr, target_len=125):
    """Ensure 2-D temporal features match the target sequence length."""
    if arr is None or arr.size == 0:
        return np.zeros((1, target_len))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    F, T_sub = arr.shape
    if T_sub == target_len:
        return arr
    elif T_sub < target_len:
        pad_width = target_len - T_sub
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="edge")
    else:
        return arr[:, :target_len]


def extract_sequential_features(audio, sr, target_len=None):
    """
    Extract raw 2-D frame-level features aligned to a fixed sequence length.
    Replicates utterance-level features across time frames to yield (161, target_len) shape.
    """
    min_samples = sr * 4
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    
    target_len_samples = len(audio)
    if target_len is None:
        target_len = int(target_len_samples // 512)
        if target_len <= 0:
            target_len = 1

    # Extract raw components
    mfcc, mfcc_d, mfcc_d2 = _extract_mfcc(audio, sr)
    mel = _extract_mel(audio, sr)
    contrast = _extract_contrast(audio, sr)
    pitch_stats, f0 = _extract_pitch(audio, sr)
    chroma = _extract_chroma(audio, sr)
    zcr = _extract_zcr(audio)
    lpcc = _extract_lpcc(audio, sr, hop_length=512)
    tonnetz = _extract_tonnetz(audio, sr)
    fmt_freq, fmt_bw = _extract_formants(audio, sr, hop_length=512)
    jitter = _extract_jitter(f0)
    shimmer = _extract_shimmer(audio, f0, sr)
    sp_ent, sh_ent = _extract_entropy(audio, sr)
    hnr = _extract_hnr(audio, sr)
    plp = _extract_plp(audio, sr, hop_length=512)
    energy = _extract_energy(audio, sr)
    duration_feat = np.array([len(audio) / sr])

    # 19 utterance-level features
    u_feats = np.hstack([pitch_stats, jitter, shimmer, duration_feat, np.array([sh_ent])])
    u_feats_seq = np.tile(u_feats.reshape(-1, 1), (1, target_len))

    # Align 2D frame-level features (142 dimensions total)
    all_features_list = [
        _match_time_dim(mfcc, target_len),
        _match_time_dim(mfcc_d, target_len),
        _match_time_dim(mfcc_d2, target_len),
        _match_time_dim(mel, target_len),
        _match_time_dim(contrast, target_len),
        _match_time_dim(chroma, target_len),
        _match_time_dim(zcr, target_len),
        _match_time_dim(lpcc, target_len),
        _match_time_dim(tonnetz, target_len),
        _match_time_dim(fmt_freq, target_len),
        _match_time_dim(fmt_bw, target_len),
        _match_time_dim(sp_ent, target_len),
        _match_time_dim(hnr, target_len),
        _match_time_dim(plp, target_len),
        _match_time_dim(energy, target_len),
        u_feats_seq
    ]

    final_matrix = np.vstack(all_features_list)
    final_matrix = np.nan_to_num(final_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return final_matrix.astype(np.float32)


def augment_audio(audio, sr):
    """Randomly apply one of: pitch shift, stretch, or noise."""
    import random
    method = random.choice([None, "pitch", "stretch", "noise"])

    if method == "pitch":
        steps = random.choice([-2, -1, 1, 2])
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)

    elif method == "stretch":
        rate = random.uniform(0.85, 1.15)
        orig_len = len(audio)
        audio = librosa.effects.time_stretch(y=audio, rate=rate)
        if len(audio) < orig_len:
            audio = np.pad(audio, (0, orig_len - len(audio)))
        else:
            audio = audio[:orig_len]

    elif method == "noise":
        amp = 0.005 * np.random.uniform() * np.amax(np.abs(audio) + 1e-10)
        audio = audio + amp * np.random.normal(size=audio.shape)

    return audio


def process_single_file_seq(fpath, emotion, augment):
    """Processes a single file and returns sequential features."""
    try:
        # Load audio
        audio, sr = librosa.load(fpath, sr=SAMPLE_RATE)

        # Peak-normalise
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Extract sequential features
        feat = extract_sequential_features(audio, sr)
        res = [(feat, emotion)]

        # Augmented sequential features
        if augment:
            aug = augment_audio(audio.copy(), sr)
            feat_aug = extract_sequential_features(aug, sr)
            res.append((feat_aug, emotion))

        return res
    except Exception as e:
        import os
        print(f"\n[ERROR] Error processing {os.path.basename(fpath)}: {e}")
        return []


if __name__ == "__main__":
    import os
    import joblib
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split

    DATASET_DIR = "hindi_dataset"
    if os.path.exists("hindi_dataset"):
        DATASET_DIR = "hindi_dataset"
    elif os.path.exists("/content/hindi_dataset"):
        DATASET_DIR = "/content/hindi_dataset"
    elif os.path.exists("/content/drive/MyDrive/hindi_dataset"):
        DATASET_DIR = "/content/drive/MyDrive/hindi_dataset"

    OUTPUT_PKL = "cnn_lstm_seq.pkl"
    AUGMENT = True

    print("=" * 60)
    print(f"  SEQUENTIAL FEATURE EXTRACTION (161x125) -> {OUTPUT_PKL}")
    print(f"  Using Dataset Directory: {DATASET_DIR}")
    print("=" * 60)

    # Walk dataset recursively
    VALID_EMOTIONS = {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sarcastic', 'surprise'}
    emotion_to_files = {emo: [] for emo in VALID_EMOTIONS}

    for root, dirs, files in os.walk(DATASET_DIR):
        dir_name = os.path.basename(root).lower()
        if dir_name in VALID_EMOTIONS:
            for f in files:
                if f.lower().endswith(".wav"):
                     emotion_to_files[dir_name].append(os.path.join(root, f))

    all_tasks = []
    for emotion in sorted(emotion_to_files.keys()):
        for fpath in emotion_to_files[emotion]:
            all_tasks.append((fpath, emotion))

    if len(all_tasks) == 0:
        print(f"[ERROR] No .wav files found in '{DATASET_DIR}'. Extraction failed.")
    else:
        # Perform 80/20 train/validation split BEFORE feature extraction and augmentation
        fpaths = [t[0] for t in all_tasks]
        emotions = [t[1] for t in all_tasks]

        train_paths, val_paths, train_y, val_y = train_test_split(
            fpaths, emotions, test_size=0.2, random_state=42, stratify=emotions
        )

        print(f"[INFO] Train files (original): {len(train_paths)}")
        print(f"[INFO] Val files: {len(val_paths)}")

        # Parallel extraction
        from joblib import Parallel, delayed
        
        # 1. Process Train Set (with augmentation)
        print(f"\n[INFO] Processing Training Set (Parallel with Augmentation)...")
        train_tasks = list(zip(train_paths, train_y))
        raw_train = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_single_file_seq)(fpath, emotion, AUGMENT)
            for fpath, emotion in train_tasks
        )

        X_train, y_train = [], []
        for res_list in raw_train:
            for feat, emotion in res_list:
                X_train.append(feat)
                y_train.append(emotion)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 2. Process Validation Set (NO augmentation)
        print(f"\n[INFO] Processing Validation Set (Parallel, NO Augmentation)...")
        val_tasks = list(zip(val_paths, val_y))
        raw_val = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_single_file_seq)(fpath, emotion, False)
            for fpath, emotion in val_tasks
        )

        X_val, y_val = [], []
        for res_list in raw_val:
            for feat, emotion in res_list:
                X_val.append(feat)
                y_val.append(emotion)

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        print(f"\n[OK] Extracted Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"[OK] Extracted Val shape: X={X_val.shape}, y={y_val.shape}")

        print(f"[OK] Saving split sequential dataset to '{OUTPUT_PKL}' without compression...")
        joblib.dump((X_train, y_train, X_val, y_val), OUTPUT_PKL, compress=0)

        print("[INFO] Verifying saved file...")
        X_tr_c, y_tr_c, X_va_c, y_va_c = joblib.load(OUTPUT_PKL)
        assert X_tr_c.shape == X_train.shape
        assert X_va_c.shape == X_val.shape
        print("[OK] Finished feature extraction successfully!")
