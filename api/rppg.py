"""
rppg.py — Remote Photoplethysmography signal processing pipeline.

Implements the POS (Plane-Orthogonal-to-Skin) algorithm from Wang et al. 2017,
followed by band-pass filtering, Welch PSD estimation and quality gating.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend
from typing import TypedDict


# ─── Types ───────────────────────────────────────────────────────────────────

class RppgResult(TypedDict):
    hr_bpm: float
    rr_bpm: float
    confidence: str          # "High" | "Medium" | "Low"
    snr_db: float
    quality_ok: bool
    hr_updates: list[float]  # running estimates at 2-s intervals
    pulse_waveform: list[float]
    error_message: str | None


# ─── Constants ────────────────────────────────────────────────────────────────

FS = 30.0            # target sample rate (Hz)
HR_LO = 0.7         # 42 bpm
HR_HI = 3.0         # 180 bpm
RR_LO = 0.1         # 6 breaths/min
RR_HI = 0.5         # 30 breaths/min
POS_WINDOW_SEC = 1.6 # POS sliding window length
SNR_THRESHOLD_DB = 0.8   # min acceptable bandwidth-normalized SNR in dB
MOTION_THRESHOLD = 25.0  # px/s
BRIGHTNESS_MIN = 25.0    # out of 255
BRIGHTNESS_MAX = 230.0
HR_MIN_VALID = 45.0      # bpm — below this show warning
HR_MAX_VALID = 140.0     # bpm — above this show warning


# ─── Utilities ────────────────────────────────────────────────────────────────

def butter_bandpass(lo: float, hi: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return b, a


def bandpass(sig: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    b, a = butter_bandpass(lo, hi, fs)
    return filtfilt(b, a, sig)


def compute_snr(pulse: np.ndarray, fs: float, hr_hz: float) -> tuple[float, float]:
    """
    Peak SNR (bandwidth-normalized) and peak prominence ratio.
    tol is set to 0.16 Hz (~10 bpm) to account for natural HRV and RSA.
    Returns (snr_db, prominence).
    """
    nperseg = min(len(pulse), int(fs * 10))
    freqs, psd = welch(pulse, fs=fs, nperseg=nperseg)
    band = (freqs >= HR_LO) & (freqs <= HR_HI)
    if not np.any(band) or hr_hz <= 0:
        return 0.0, 0.0

    tol = 0.16  # Hz tolerance around peak
    signal_mask = (
        ((freqs >= hr_hz - tol) & (freqs <= hr_hz + tol)) |
        ((freqs >= 2 * hr_hz - tol) & (freqs <= 2 * hr_hz + tol))
    )
    noise_mask = band & ~signal_mask

    signal_power = float(np.sum(psd[signal_mask & band]))
    noise_power = float(np.sum(psd[noise_mask]))

    # Peak prominence over median noise in band
    peak_val = float(np.max(psd[signal_mask & band])) if np.any(signal_mask & band) else 0.0
    median_noise = float(np.median(psd[band])) if np.any(band) else 1e-6
    prominence = peak_val / max(median_noise, 1e-9)

    sig_bins = max(1, int(np.sum(signal_mask & band)))
    noise_bins = max(1, int(np.sum(noise_mask)))

    if noise_power <= 0 or signal_power <= 0:
        return 0.0, prominence

    # Ratio of average power spectral density (bandwidth-normalized)
    density_ratio = (signal_power / sig_bins) / (noise_power / noise_bins)
    snr_db = float(10 * np.log10(max(density_ratio, 1e-6)))

    return snr_db, prominence


def parabolic_peak_refine(freqs: np.ndarray, psd: np.ndarray, idx: int) -> float:
    """Sub-bin frequency refinement via parabolic interpolation."""
    if idx <= 0 or idx >= len(psd) - 1:
        return float(freqs[idx])
    alpha = psd[idx - 1]
    beta = psd[idx]
    gamma = psd[idx + 1]
    denom = alpha - 2 * beta + gamma
    if denom == 0:
        return float(freqs[idx])
    offset = 0.5 * (alpha - gamma) / denom
    df = freqs[1] - freqs[0]
    return float(freqs[idx] + offset * df)


# ─── POS Algorithm ───────────────────────────────────────────────────────────

def pos_algorithm(rgb: np.ndarray, fs: float) -> np.ndarray:
    """
    POS (Plane-Orthogonal-to-Skin) method — Wang et al. 2017.
    rgb: (N, 3) array of mean R, G, B per frame (0–255 or 0–1 scale).
    Returns the combined pulse signal (length N).
    """
    N = len(rgb)
    W = max(2, int(round(POS_WINDOW_SEC * fs)))  # window length
    H = np.zeros(N)
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)  # POS projection matrix

    for i in range(0, N - W + 1):
        seg = rgb[i: i + W].astype(float)           # (W, 3)
        mu = seg.mean(axis=0, keepdims=True)         # (1, 3)
        mu[mu == 0] = 1e-6
        Cn = seg / mu                                # normalized (W, 3)
        S = (P @ Cn.T)                               # (2, W)
        std1 = np.std(S[0])
        std2 = np.std(S[1])
        if std2 < 1e-9:
            std2 = 1e-9
        h = S[0] + (std1 / std2) * S[1]             # (W,)
        H[i: i + W] += h - h.mean()

    return H


# ─── Resample ────────────────────────────────────────────────────────────────

def resample_to_uniform(rgb: np.ndarray, timestamps: np.ndarray,
                        fs: float = FS) -> np.ndarray:
    """Linearly interpolate RGB traces to a uniform grid at `fs` Hz."""
    t0, t1 = timestamps[0], timestamps[-1]
    t_uniform = np.arange(t0, t1, 1.0 / fs)
    rgb_uniform = np.zeros((len(t_uniform), 3))
    for c in range(3):
        rgb_uniform[:, c] = np.interp(t_uniform, timestamps, rgb[:, c])
    return rgb_uniform


# ─── HR estimation ───────────────────────────────────────────────────────────

def estimate_hr(pulse: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Compute Welch PSD and return (hr_hz, hr_bpm) for the dominant peak.
    """
    nperseg = min(len(pulse), int(fs * 10))
    freqs, psd = welch(pulse, fs=fs, nperseg=nperseg)
    band = (freqs >= HR_LO) & (freqs <= HR_HI)
    if not np.any(band):
        return 0.0, 0.0
    psd_band = psd.copy()
    psd_band[~band] = 0
    idx = int(np.argmax(psd_band))
    hr_hz = parabolic_peak_refine(freqs, psd_band, idx)
    hr_bpm = hr_hz * 60.0
    return hr_hz, hr_bpm


# ─── RR estimation ───────────────────────────────────────────────────────────

def estimate_rr(pulse: np.ndarray, fs: float) -> float:
    """Estimate breathing rate from the low-frequency modulation of the pulse."""
    if len(pulse) < int(fs * 15):
        return 0.0
    try:
        rr_sig = bandpass(pulse, RR_LO, RR_HI, fs)
        nperseg = min(len(rr_sig), int(fs * 20))
        freqs, psd = welch(rr_sig, fs=fs, nperseg=nperseg)
        band = (freqs >= RR_LO) & (freqs <= RR_HI)
        if not np.any(band):
            return 0.0
        psd_b = psd.copy()
        psd_b[~band] = 0
        idx = int(np.argmax(psd_b))
        rr_hz = parabolic_peak_refine(freqs, psd_b, idx)
        return round(rr_hz * 60.0, 1)
    except Exception:
        return 0.0


# ─── Quality check ───────────────────────────────────────────────────────────

def quality_checks(
    rgb: np.ndarray,
    motion_pxps: float,
    snr_db: float,
    prominence: float,
    hr_bpm: float,
) -> tuple[bool, str | None]:
    """Returns (ok, error_message)."""
    mean_brightness = float(rgb.mean())
    if mean_brightness < BRIGHTNESS_MIN:
        return False, "Image too dark — move closer to a steady light."
    if mean_brightness > BRIGHTNESS_MAX:
        return False, "Image overexposed — reduce brightness or move back."
    if motion_pxps > MOTION_THRESHOLD:
        return False, "Too much motion detected — please hold still."
    # Quality gate: pass if either bandwidth-normalized SNR is acceptable OR peak prominence stands out from noise
    if snr_db < SNR_THRESHOLD_DB and prominence < 1.6:
        return False, "Signal quality too low — try better lighting and stay still."
    return True, None


def confidence_label(snr_db: float, prominence: float = 0.0) -> str:
    if snr_db >= 5.0 or prominence >= 4.0:
        return "High"
    if snr_db >= 2.0 or prominence >= 2.2:
        return "Medium"
    return "Low"


# ─── Running HR updates ──────────────────────────────────────────────────────

def running_hr_estimates(pulse: np.ndarray, fs: float,
                         update_interval_s: float = 2.0,
                         min_data_s: float = 8.0) -> list[float]:
    """
    Return HR estimates at every `update_interval_s` seconds,
    starting once `min_data_s` of data has accumulated.
    """
    step = int(round(update_interval_s * fs))
    min_samples = int(round(min_data_s * fs))
    estimates = []
    for end in range(min_samples, len(pulse) + 1, step):
        seg = pulse[:end]
        try:
            _, bpm = estimate_hr(seg, fs)
            estimates.append(round(bpm, 1))
        except Exception:
            estimates.append(0.0)
    return estimates


# ─── Main entry point ─────────────────────────────────────────────────────────

def process_rppg(
    rgb_traces: list[list[float]],  # list of [R, G, B] per frame
    timestamps: list[float],        # seconds since start
    motion_pxps: float = 0.0,
) -> RppgResult:
    """
    Full rPPG pipeline.
    rgb_traces: raw [R, G, B] mean values per frame (0–255).
    timestamps: monotonically increasing time in seconds.
    motion_pxps: mean face-box motion in pixels/second.
    """
    rgb = np.array(rgb_traces, dtype=float)
    ts = np.array(timestamps, dtype=float)

    if len(rgb) < 60:  # need at least 2 s of data
        return RppgResult(
            hr_bpm=0, rr_bpm=0, confidence="Low", snr_db=0,
            quality_ok=False, hr_updates=[], pulse_waveform=[],
            error_message="Not enough data — scan was too short.",
        )

    # Resample to uniform grid
    rgb_u = resample_to_uniform(rgb, ts, FS)

    # POS signal
    pulse_raw = pos_algorithm(rgb_u, FS)

    # Detrend + band-pass
    pulse_dt = detrend(pulse_raw)
    min_len_for_filter = 13  # minimum samples needed by filtfilt with order 4
    if len(pulse_dt) < min_len_for_filter:
        return RppgResult(
            hr_bpm=0, rr_bpm=0, confidence="Low", snr_db=0,
            quality_ok=False, hr_updates=[], pulse_waveform=[],
            error_message="Not enough frames for filtering.",
        )

    pulse = bandpass(pulse_dt, HR_LO, HR_HI, FS)

    # Estimate HR using the last 18-20 s (or full pulse if shorter) for higher frequency resolution
    analysis_len = min(len(pulse), int(FS * 20))
    tail = pulse[-analysis_len:]
    hr_hz, hr_bpm = estimate_hr(tail, FS)

    # Running estimates at 2-second intervals
    hr_updates = running_hr_estimates(pulse, FS)

    # Apply median filtering across running updates over final 15s (PRD Step 6)
    valid_updates = [bpm for bpm in hr_updates if HR_MIN_VALID <= bpm <= HR_MAX_VALID]
    if len(valid_updates) >= 3:
        median_bpm = float(np.median(valid_updates[-5:]))
        if HR_MIN_VALID <= median_bpm <= HR_MAX_VALID and abs(median_bpm - hr_bpm) < 18:
            hr_bpm = median_bpm
            hr_hz = hr_bpm / 60.0

    # SNR and peak prominence
    snr_db, prominence = compute_snr(tail, FS, hr_hz) if hr_hz > 0 else (0.0, 0.0)

    # Quality gate
    quality_ok, error_msg = quality_checks(rgb_u, motion_pxps, snr_db, prominence, hr_bpm)

    # RR
    rr_bpm = estimate_rr(pulse, FS) if quality_ok else 0.0

    # Running estimates
    hr_updates = running_hr_estimates(pulse, FS)

    # Normalize waveform for display (-1 to 1)
    p_std = pulse.std()
    if p_std > 0:
        pulse_norm = (pulse / (3 * p_std)).clip(-1, 1).tolist()
    else:
        pulse_norm = pulse.tolist()

    # Warn on out-of-range HR even if signal quality is ok
    if quality_ok and (hr_bpm < HR_MIN_VALID or hr_bpm > HR_MAX_VALID):
        error_msg = f"Unusual reading ({hr_bpm:.0f} bpm). Re-check with a proper device."

    return RppgResult(
        hr_bpm=round(hr_bpm, 1),
        rr_bpm=rr_bpm,
        confidence=confidence_label(snr_db, prominence),
        snr_db=round(snr_db, 2),
        quality_ok=quality_ok,
        hr_updates=hr_updates,
        pulse_waveform=pulse_norm,
        error_message=error_msg,
    )
