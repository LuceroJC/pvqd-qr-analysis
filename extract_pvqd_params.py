#!/usr/bin/env python3
"""
PVQD Sustained Vowel Acoustic Parameter Extraction

Extracts 14 acoustic parameters from sustained vowel recordings for QR factorization
analysis of perceptual voice quality predictors.

Parameters computed:
  Group A (Basic): F0 mean, jitter local %, shimmer local %, HNR, CPPS
  Group B (AVQI): shimmer local dB, slope, tilt
  Group C (ABI): GNE, Hno-6000 Hz, HNR-D, H1-H2, PSD
  Group D (Spectral): Alpha Ratio

Algorithms adapted from PhonaLab API:
  - /api/services/audio_utils.py (preprocessing)
  - /api/services/analysis/core.py (basic parameters)
  - /api/services/avqi.py (AVQI/ABI parameters)
  - /api/services/spectral.py (spectral measures)

Author: Jorge C. Lucero
Date: 2026-04-02
"""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call


# =============================================================================
# PREPROCESSING FUNCTIONS
# Adapted from /api/services/audio_utils.py
# =============================================================================

def trim_silence(sound, threshold_db=-30.0):
    """
    Trim leading/trailing silence from audio.

    Adapted from: /api/services/audio_utils.py:trim_silence (lines 5-14)

    Algorithm:
    - Convert to intensity contour (min pitch 50 Hz)
    - Find first/last frames above threshold_db
    - Add 0.05s padding at boundaries
    - Extract the voiced portion

    Args:
        sound: parselmouth.Sound object
        threshold_db: Intensity threshold in dB (default -30)

    Returns:
        Trimmed parselmouth.Sound object
    """
    try:
        intensity = sound.to_intensity(minimum_pitch=50)
        times = intensity.xs()
        values = intensity.values[0]

        # Find first frame above threshold
        start_idx = 0
        for i, v in enumerate(values):
            if v > threshold_db:
                start_idx = i
                break

        # Find last frame above threshold
        end_idx = len(values) - 1
        for i in range(len(values) - 1, -1, -1):
            if values[i] > threshold_db:
                end_idx = i
                break

        # Add padding and extract
        start_t = max(0, times[start_idx] - 0.05)
        end_t = min(sound.duration, times[end_idx] + 0.05)

        return call(sound, "Extract part", start_t, end_t, "rectangular", 1.0, "no")
    except Exception as e:
        print(f"  Warning: trim_silence failed ({e}), using original sound")
        return sound


def concatenate_voiced_segments(sound):
    """
    Extract and concatenate voiced segments, removing internal silences.

    Adapted from: /api/services/audio_utils.py:concatenate_voiced_segments (lines 64-102)

    Algorithm:
    - Create TextGrid with silence/sounding labels
    - Parameters: min pitch 50 Hz, min silence 0.1s, threshold -25 dB
    - Extract all "sounding" intervals
    - Concatenate them together

    Args:
        sound: parselmouth.Sound object

    Returns:
        parselmouth.Sound with only voiced segments
    """
    try:
        tg = call(
            sound,
            "To TextGrid (silences)",
            50,      # min pitch
            0.1,     # min silence duration
            -25,     # silence threshold (dB)
            0.1,     # min sounding duration
            0.1,     # padding
            "silence",
            "sounding"
        )

        parts = []
        n_int = call(tg, "Get number of intervals", 1)

        for i in range(1, n_int + 1):
            label = call(tg, "Get label of interval", 1, i)
            if label == "sounding":
                start = call(tg, "Get start time of interval", 1, i)
                end = call(tg, "Get end time of interval", 1, i)
                part = call(sound, "Extract part", start, end, "rectangular", 1.0, "no")
                parts.append(part)

        if not parts:
            return sound  # fallback if no voiced segments found

        voiced = parts[0]
        for p in parts[1:]:
            voiced = call([voiced, p], "Concatenate")

        return voiced

    except Exception as e:
        print(f"  Warning: concatenate_voiced_segments failed ({e}), using original sound")
        return sound
    finally:
        gc.collect()


def preprocess_audio(wav_path):
    """
    Load and preprocess a sustained vowel WAV file.

    Steps:
    1. Load WAV at native sample rate (no resampling)
    2. Trim leading/trailing silence
    3. Concatenate voiced segments (remove internal gaps)

    Args:
        wav_path: Path to WAV file

    Returns:
        tuple: (preprocessed Sound, sample_rate, voiced_duration, flag_short)
    """
    # Load at native sample rate
    sound = parselmouth.Sound(wav_path)
    sample_rate = int(sound.sampling_frequency)

    # Preprocess
    sound = trim_silence(sound)
    # sound = concatenate_voiced_segments(sound)

    voiced_duration = sound.duration
    flag_short = voiced_duration < 0.5 # 1.0

    return sound, sample_rate, voiced_duration, flag_short


# =============================================================================
# GROUP A: BASIC PARAMETERS
# Adapted from /api/services/analysis/core.py
# =============================================================================

def compute_f0_mean(sound):
    """
    Compute mean fundamental frequency (F0).

    Adapted from: /api/services/analysis/core.py:analyze_f0 (lines 134-159)

    Algorithm:
    - Convert to Pitch object (Praat default algorithm)
    - Extract voiced F0 values (>0)
    - Apply 2.5σ outlier removal
    - Return mean of filtered values

    Returns:
        float: Mean F0 in Hz, or NaN if calculation fails
    """
    try:
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        f0 = f0[f0 > 0]  # Only voiced frames

        if len(f0) == 0:
            return np.nan

        # Outlier removal (2.5 SD)
        m, s = np.mean(f0), np.std(f0)
        if s > 0:
            f0_filtered = f0[(f0 > m - 2.5 * s) & (f0 < m + 2.5 * s)]
            if len(f0_filtered) < len(f0) * 0.5:
                f0_filtered = f0  # Keep original if too many removed
        else:
            f0_filtered = f0

        return float(np.mean(f0_filtered))
    except Exception as e:
        print(f"  Warning: F0 calculation failed ({e})")
        return np.nan


def compute_jitter_local_pct(sound):
    """
    Compute jitter (local) as percentage.

    Adapted from: /api/services/analysis/core.py:analyze_jitter (lines 162-174)

    Algorithm:
    - Create PointProcess (periodic, cc) with 75-600 Hz range
    - Get jitter (local) with period floor 0.0001s, ceiling 0.02s, factor 1.3
    - Convert to percentage

    Returns:
        float: Jitter local %, or NaN if calculation fails
    """
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        if call(pp, "Get number of points") < 3:
            return np.nan

        jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

        if jitter_local is None or np.isnan(jitter_local):
            return np.nan

        return float(jitter_local * 100)
    except Exception as e:
        print(f"  Warning: Jitter calculation failed ({e})")
        return np.nan


def compute_shimmer_local_pct(sound):
    """
    Compute shimmer (local) as percentage.

    Adapted from: /api/services/analysis/core.py:analyze_shimmer (lines 177-189)

    Algorithm:
    - Create PointProcess (periodic, cc) with 75-600 Hz range
    - Get shimmer (local) with period floor 0.0001s, ceiling 0.02s, factors 1.3/1.6
    - Convert to percentage

    Returns:
        float: Shimmer local %, or NaN if calculation fails
    """
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        if call(pp, "Get number of points") < 3:
            return np.nan

        shimmer_local = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        if shimmer_local is None or np.isnan(shimmer_local):
            return np.nan

        return float(shimmer_local * 100)
    except Exception as e:
        print(f"  Warning: Shimmer calculation failed ({e})")
        return np.nan


def compute_hnr(sound):
    """
    Compute Harmonics-to-Noise Ratio.

    Adapted from: /api/services/analysis/core.py:analyze_hnr (lines 192-203)

    Algorithm:
    - Convert to Harmonicity object (Praat default)
    - Extract values != -200 (Praat's undefined marker)
    - Return mean

    Returns:
        float: HNR in dB, or NaN if calculation fails
    """
    try:
        h = sound.to_harmonicity()
        vals = h.values[h.values != -200]

        if len(vals) == 0:
            return np.nan

        return float(np.mean(vals))
    except Exception as e:
        print(f"  Warning: HNR calculation failed ({e})")
        return np.nan


def compute_cpps(sound):
    """
    Compute smoothed Cepstral Peak Prominence (CPPS).

    Adapted from: /api/services/analysis/core.py:analyze_cpp_efficient (lines 206-247)
    and /api/services/avqi.py:analyze_avqi_dual (lines 77-81)

    Algorithm:
    - Convert to PowerCepstrogram (pitch floor 60 Hz, time step 0.002s, max freq 5000 Hz, 50 bins)
    - Get CPPS with Praat smoothing parameters:
      - subtract_tilt_before_smoothing: no
      - time_averaging_window: 0.01s
      - quefrency_averaging_window: 0.001s
      - peak_search_range: 60-330 Hz
      - tolerance: 0.05
      - interpolation: Parabolic
      - tilt_line_quefrency_range: 0.001-0 (auto)
      - tilt_line_fit: Straight
      - fit_method: Robust

    Returns:
        float: CPPS in dB, or NaN if calculation fails
    """
    try:
        if sound.duration < 0.5:
            print(f"  Warning: Duration too short for CPPS ({sound.duration:.3f}s)")
            return np.nan

        pcg = call(sound, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        cpps = call(pcg, "Get CPPS", "no", 0.01, 0.001, 60.0, 330.0, 0.05,
                   "Parabolic", 0.001, 0.0, "Straight", "Robust")

        if cpps is None or np.isnan(cpps):
            return np.nan

        return float(cpps)
    except Exception as e:
        print(f"  Warning: CPPS calculation failed ({e})")
        return np.nan


# =============================================================================
# GROUP B: AVQI PARAMETERS
# Adapted from /api/services/avqi.py
# =============================================================================

def compute_shimmer_local_db(sound):
    """
    Compute shimmer (local, dB).

    Adapted from: /api/services/avqi.py:analyze_avqi_dual (lines 88-91)
    and extract_voice_report_params (lines 10-23)

    Algorithm:
    - Create PointProcess (periodic, cc) with 50-400 Hz range
    - Get shimmer (local_dB) with period floor 0.0001s, ceiling 0.02s, factors 1.3/1.6

    Returns:
        float: Shimmer local dB, or NaN if calculation fails
    """
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 50, 400)
        shimmer_db = call([sound, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        if shimmer_db is None or np.isnan(shimmer_db):
            return np.nan

        return float(shimmer_db)
    except Exception as e:
        print(f"  Warning: Shimmer dB calculation failed ({e})")
        return np.nan


def compute_slope(sound):
    """
    Compute LTAS slope (general slope of the spectrum).

    Adapted from: /api/services/avqi.py:analyze_avqi_dual (lines 94-95)

    Algorithm:
    - Convert to LTAS with 1 Hz bandwidth
    - Get slope between 0-1000 Hz and 1000-10000 Hz bands using energy

    Returns:
        float: Slope in dB, or NaN if calculation fails
    """
    try:
        ltas = call(sound, "To Ltas", 1)
        slope = call(ltas, "Get slope", 0, 1000, 1000, 10000, "energy")

        if slope is None or np.isnan(slope):
            return np.nan

        return float(slope)
    except Exception as e:
        print(f"  Warning: Slope calculation failed ({e})")
        return np.nan


def compute_tilt(sound):
    """
    Compute LTAS tilt (slope of regression trend line through spectrum).

    Adapted from: /api/services/avqi.py:analyze_avqi_dual (lines 96-97)

    Algorithm:
    - Convert to LTAS with 1 Hz bandwidth
    - Compute trend line through LTAS (1-10000 Hz)
    - Get slope of trend line between 0-1000 Hz and 1000-10000 Hz bands

    Returns:
        float: Tilt in dB, or NaN if calculation fails
    """
    try:
        ltas = call(sound, "To Ltas", 1)
        trend_line = call(ltas, "Compute trend line", 1, 10000)
        tilt = call(trend_line, "Get slope", 0, 1000, 1000, 10000, "energy")

        if tilt is None or np.isnan(tilt):
            return np.nan

        return float(tilt)
    except Exception as e:
        print(f"  Warning: Tilt calculation failed ({e})")
        return np.nan


# =============================================================================
# GROUP C: ABI PARAMETERS
# Adapted from /api/services/avqi.py
# =============================================================================

def compute_gne(sound):
    """
    Compute Glottal-to-Noise Excitation ratio (maximum at 4500 Hz).

    Adapted from: /api/services/avqi.py:analyze_abi_dual (lines 119-121)

    Algorithm:
    - Convert to Harmonicity (GNE method)
    - Parameters: 500 Hz low, 4500 Hz high, 1000 Hz step, 80 Hz bandwidth
    - Return maximum value

    Returns:
        float: GNE max, or NaN if calculation fails
    """
    try:
        hg = call(sound, "To Harmonicity (gne)", 500, 4500, 1000, 80)
        gne = call(hg, "Get maximum")

        if gne is None or np.isnan(gne):
            return np.nan

        return float(gne)
    except Exception as e:
        print(f"  Warning: GNE calculation failed ({e})")
        return np.nan


def compute_abi_windowed_params(sound):
    """
    Compute ABI windowed parameters: Hno-6000 Hz (hfno), HNR-D (hnrd), H1-H2.

    Adapted from: /api/services/avqi.py:calc_abi_windowed (lines 25-69)

    Algorithm (100ms windows, max 50 windows):

    For each window:
    1. hfno (Hno-6000 Hz):
       - Create spectrum -> LTAS (1-to-1)
       - Normalize LTAS (shift so min=0)
       - Compute energy ratio: mean(0-6000 Hz) / mean(6000-10000 Hz)

    2. hnrd (HNR-D, Dejonckere variant):
       - Create PowerCepstrogram -> PowerCepstrum slice
       - Subtract tilt, find quefrency peak -> F0 estimate
       - Create LTAS (6.11 Hz bandwidth)
       - Find harmonic peaks in 500-1500 Hz range (at multiples of F0)
       - Find valleys between harmonics
       - HNR-D = mean(peak amplitudes) - mean(valley amplitudes)

    3. h1h2 (H1-H2):
       - From LTAS, find amplitude of first harmonic (F0 ± 20 Hz)
       - Find amplitude of second harmonic (2*F0 ± 20 Hz)
       - H1-H2 = amplitude(H1) - amplitude(H2)

    Returns:
        dict: {'hfno': float, 'hnrd': float, 'h1h2': float} or NaN values
    """
    try:
        start = call(sound, "Get start time")
        dur = call(sound, "Get total duration")
        n_win = min(int(dur / 0.1), 50)

        if n_win < 1:
            return {'hfno': np.nan, 'hnrd': np.nan, 'h1h2': np.nan}

        hfno_vals = []
        hnrd_vals = []
        h1h2_vals = []

        for n in range(1, n_win + 1):
            try:
                # Extract 100ms window
                sw = start + (n - 1) * 0.1
                ew = start + n * 0.1
                win = call(sound, "Extract part", sw, ew, "rectangular", 1, "yes")

                # Compute spectrum and LTAS
                spec = call(win, "To Spectrum", "yes")
                ltas = call(spec, "To Ltas (1-to-1)")

                # Normalize LTAS
                lmin = call(ltas, "Get minimum", 0, 10000, "None")
                if lmin:
                    if lmin < 0:
                        call(ltas, "Formula", f"self + {abs(lmin)}")
                    else:
                        call(ltas, "Formula", f"self - {lmin}")

                # hfno: energy ratio 0-6000 / 6000-10000 Hz
                e06 = call(ltas, "Get mean", 0, 6000, "energy")
                e610 = call(ltas, "Get mean", 6000, 10000, "energy")
                if e610 and e610 > 0:
                    hfno_vals.append(e06 / e610)

                # Get F0 from cepstrum for harmonic analysis
                cg = call(win, "To PowerCepstrogram", 60, 0.002, 5000, 50)
                cp = call(cg, "To PowerCepstrum (slice)", sw + 0.05)
                call(cp, "Subtract tilt", 0.001, 0, "Straight", "Robust")
                q = call(cp, "Get quefrency of peak", 60, 400, "Parabolic")

                if not q or q <= 0:
                    continue

                f0 = 1.0 / q

                # LTAS with finer resolution for harmonic analysis
                ltas6 = call(spec, "To Ltas", 6.11)

                # Refine F0 estimate
                pf = call(ltas6, "Get frequency of maximum", f0 - 20, f0 + 20, "None") or f0

                # hnrd: harmonic peaks vs valleys in 500-1500 Hz
                fh = int(np.ceil(500 / pf))  # First harmonic in range
                lh = int(np.floor(1500 / pf))  # Last harmonic in range

                if fh <= lh:
                    # Get harmonic peaks
                    pks = []
                    for p in range(fh, lh + 1):
                        pk = call(ltas, "Get maximum", p * pf - 20, p * pf + 20, "None")
                        if pk and not np.isnan(pk):
                            pks.append(pk)

                    # Get valleys between harmonics
                    vls = []
                    for v in range(fh, int(np.floor(1500 / f0))):
                        vl = v * pf + 20
                        vr = (v + 1) * pf - 20
                        if vl < vr:
                            val = call(ltas, "Get minimum", vl, vr, "None")
                            if val and not np.isnan(val):
                                vls.append(val)

                    if pks and vls:
                        hnrd_vals.append(np.mean(pks) - np.mean(vls))

                # h1h2: amplitude difference between H1 and H2
                h1 = call(ltas6, "Get maximum", pf - 20, pf + 20, "None")
                h2 = call(ltas6, "Get maximum", 2 * pf - 20, 2 * pf + 20, "None")
                if h1 and h2 and not np.isnan(h1) and not np.isnan(h2):
                    h1h2_vals.append(h1 - h2)

            except Exception:
                continue

        return {
            'hfno': float(np.mean(hfno_vals)) if hfno_vals else np.nan,
            'hnrd': float(np.mean(hnrd_vals)) if hnrd_vals else np.nan,
            'h1h2': float(np.mean(h1h2_vals)) if h1h2_vals else np.nan
        }

    except Exception as e:
        print(f"  Warning: ABI windowed calculation failed ({e})")
        return {'hfno': np.nan, 'hnrd': np.nan, 'h1h2': np.nan}


def compute_psd(sound):
    """
    Compute Period Standard Deviation (PSD) from Voice Report.

    Adapted from: /api/services/avqi.py:extract_voice_report_params (lines 10-23)

    Algorithm:
    - Create Pitch object (75-600 Hz range)
    - Create PointProcess (cc method)
    - Generate Voice Report
    - Extract "Standard deviation of period" value

    Returns:
        float: PSD in seconds, or NaN if calculation fails
    """
    try:
        pitch = call(sound, "To Pitch", 0, 70, 600)
        pp = call([sound, pitch], "To PointProcess (cc)")
        rpt = call([sound, pitch, pp], "Voice report", 0, 0, 70, 600, 1.3, 1.6, 0.03, 0.45)

        # Extract PSD from report text
        label = "Standard deviation of period: "
        if label not in rpt:
            return np.nan

        after_label = rpt[rpt.find(label) + len(label):]
        value_str = after_label.split('\n')[0].split()[0]

        if 'undefined' in value_str.lower():
            return np.nan

        return float(value_str)

    except Exception as e:
        print(f"  Warning: PSD calculation failed ({e})")
        return np.nan


# =============================================================================
# GROUP D: SPECTRAL PARAMETERS
# Adapted from /api/services/spectral.py
# =============================================================================

def compute_alpha_ratio(sound):
    """
    Compute Alpha Ratio (spectral tilt measure).

    Adapted from: /api/services/spectral.py:calculate_alpha_ratio (lines 49-63)

    Formula: 10 × log10(Energy_0-1000Hz / Energy_1000-5000Hz)

    Algorithm:
    - Convert to Spectrum
    - Get band energy difference between 0-1000 Hz and 1000-5000 Hz

    Returns:
        float: Alpha Ratio in dB, or NaN if calculation fails
    """
    try:
        spectrum = call(sound, "To Spectrum", "yes")
        alpha_ratio = call(spectrum, "Get band energy difference", 0, 1000, 1000, 5000)

        if alpha_ratio is None or np.isnan(alpha_ratio):
            return np.nan

        return float(alpha_ratio)
    except Exception as e:
        print(f"  Warning: Alpha Ratio calculation failed ({e})")
        return np.nan

# =============================================================================
# GROUP E: NON LINEAR MEASURES
# =============================================================================

def compute_dfa(sound):
    """
    Compute Detrended Fluctuation Analysis (DFA).

    Algorithm:
    - Extract amplitude envelope
    - Integrate signal
    - Compute fluctuation function across window sizes
    - Estimate scaling exponent (alpha)

    Returns:
        float: DFA alpha (≈0.5 random, ≈1 structured)
    """
    try:
        signal = sound.values[0]

        if len(signal) < 1000:
            return np.nan

        # Remove mean
        signal = signal - np.mean(signal)

        # Integrate signal
        y = np.cumsum(signal)

        N = len(y)
        scales = np.logspace(2, np.log10(N // 4), num=10).astype(int)

        flucts = []

        for s in scales:
            if s < 10:
                continue

            n_segments = N // s
            if n_segments < 2:
                continue

            rms = []

            for i in range(n_segments):
                segment = y[i * s:(i + 1) * s]
                t = np.arange(len(segment))

                # Linear detrending
                coeffs = np.polyfit(t, segment, 1)
                trend = np.polyval(coeffs, t)

                rms.append(np.sqrt(np.mean((segment - trend) ** 2)))

            if rms:
                flucts.append(np.mean(rms))

        if len(flucts) < 2:
            return np.nan

        # Log-log slope = DFA alpha
        log_scales = np.log(scales[:len(flucts)])
        log_flucts = np.log(flucts)

        alpha, _ = np.polyfit(log_scales, log_flucts, 1)

        return float(alpha)

    except Exception as e:
        print(f"  Warning: DFA calculation failed ({e})")
        return np.nan
    
def compute_rpde(sound, m=3, tau=1, epsilon_factor=0.1, max_period=500):
    """
    Compute full RPDE using phase-space embedding and recurrence analysis.

    Parameters:
        m: embedding dimension (typical: 3–5)
        tau: delay (in samples, usually 1–3)
        epsilon_factor: radius threshold as fraction of signal std
        max_period: maximum recurrence time (in samples)

    Returns:
        float: RPDE in [0, 1]
    """
    try:
        x = sound.values[0]
        N = len(x)

        # Basic sanity check
        if N < (m + 1) * tau + 100:
            return np.nan

        # Normalize signal
        x = x - np.mean(x)
        std = np.std(x)
        if std == 0:
            return np.nan

        x = x / std

        # Build embedding
        M = N - (m - 1) * tau
        embedded = np.zeros((M, m))

        for i in range(m):
            embedded[:, i] = x[i * tau:i * tau + M]

        # Distance threshold
        epsilon = epsilon_factor * np.std(embedded)

        recurrence_times = []

        # For each point, find recurrence
        for i in range(M - max_period):
            xi = embedded[i]

            for j in range(i + 1, min(i + max_period, M)):
                dist = np.linalg.norm(xi - embedded[j])

                if dist < epsilon:
                    recurrence_times.append(j - i)
                    break  # first recurrence only

        if len(recurrence_times) < 20:
            return np.nan

        recurrence_times = np.array(recurrence_times)

        # Histogram (probability distribution)
        hist, _ = np.histogram(recurrence_times,
                               bins=50,
                               range=(1, max_period),
                               density=True)

        hist = hist[hist > 0]

        if len(hist) < 2:
            return np.nan

        # Shannon entropy
        entropy = -np.sum(hist * np.log(hist))

        # Normalize
        max_entropy = np.log(len(hist))
        if max_entropy == 0:
            return np.nan

        rpde = entropy / max_entropy

        return float(rpde)

    except Exception as e:
        print(f"  Warning: Full RPDE calculation failed ({e})")
        return np.nan
        
# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def extract_all_parameters(wav_path):
    """
    Extract all 14 acoustic parameters from a sustained vowel WAV file.

    Args:
        wav_path: Path to WAV file

    Returns:
        dict: All parameters plus metadata (sample_rate, voiced_duration, flag_short)
    """
    filename = os.path.basename(wav_path)
    print(f"Processing: {filename}")

    try:
        # Preprocess
        sound, sample_rate, voiced_duration, flag_short = preprocess_audio(wav_path)

        if flag_short:
            print(f"  Note: Short recording ({voiced_duration:.2f}s < 0.5s)")

        # Initialize results with metadata
        results = {
            'filename': filename,
            'sample_rate': sample_rate,
            'voiced_duration_sec': round(voiced_duration, 3),
            'flag_short': flag_short,
        }

        # Group A: Basic parameters
        results['f0_mean'] = compute_f0_mean(sound)
        results['jitter_local_pct'] = compute_jitter_local_pct(sound)
        results['shimmer_local_pct'] = compute_shimmer_local_pct(sound)
        results['hnr'] = compute_hnr(sound)
        results['cpps'] = compute_cpps(sound)

        # Group B: AVQI parameters
        results['shimmer_local_dB'] = compute_shimmer_local_db(sound)
        results['slope'] = compute_slope(sound)
        results['tilt'] = compute_tilt(sound)

        # Group C: ABI parameters
        results['gne'] = compute_gne(sound)
        abi_windowed = compute_abi_windowed_params(sound)
        results['hno_6000'] = abi_windowed['hfno']
        results['hnr_d'] = abi_windowed['hnrd']
        results['h1_h2'] = abi_windowed['h1h2']
        results['psd'] = compute_psd(sound)

        # Group D: Spectral parameters
        results['alpha_ratio'] = compute_alpha_ratio(sound)

        # # Group E: Nonlinear parameters
        # results['rpde'] = compute_rpde(sound)
        # results['dfa'] = compute_dfa(sound)

        return results

    except Exception as e:
        print(f"  ERROR: Failed to process {filename}: {e}")
        return {
            'filename': filename,
            'sample_rate': np.nan,
            'voiced_duration_sec': np.nan,
            'flag_short': True,
            'f0_mean': np.nan,
            'jitter_local_pct': np.nan,
            'shimmer_local_pct': np.nan,
            'hnr': np.nan,
            'cpps': np.nan,
            'shimmer_local_dB': np.nan,
            'slope': np.nan,
            'tilt': np.nan,
            'gne': np.nan,
            'hno_6000': np.nan,
            'hnr_d': np.nan,
            'h1_h2': np.nan,
            'psd': np.nan,
            'alpha_ratio': np.nan,
            # 'rpde': np.nan,
            # 'dfa': np.nan,
        }
    finally:
        gc.collect()


# =============================================================================
# BATCH PROCESSING AND CLI
# =============================================================================

def find_wav_files(input_dir, pattern=None):
    """
    Find WAV files in directory matching optional pattern.

    Args:
        input_dir: Directory to search
        pattern: Optional suffix pattern (e.g., '_vowel_a.wav')

    Returns:
        Sorted list of full file paths
    """
    wav_files = []
    for f in os.listdir(input_dir):
        if f.lower().endswith('.wav'):
            if pattern is None or f.endswith(pattern):
                wav_files.append(os.path.join(input_dir, f))
    return sorted(wav_files)


def process_directory(input_dir, output_path, pattern=None):
    """
    Process all WAV files in a directory and write results to CSV.

    Args:
        input_dir: Directory containing WAV files
        output_path: Output CSV file path
        pattern: Optional filename suffix pattern (e.g., '_vowel_a.wav')
    """
    wav_files = find_wav_files(input_dir, pattern)

    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} WAV files in {input_dir}")
    print(f"Output will be saved to: {output_path}")
    print("-" * 60)

    # CSV columns
    fieldnames = [
        'filename', 'sample_rate', 'voiced_duration_sec', 'flag_short',
        'f0_mean', 'jitter_local_pct', 'shimmer_local_pct', 'shimmer_local_dB',
        'hnr', 'cpps', 'slope', 'tilt', 'alpha_ratio',
        'gne', 'hno_6000', 'hnr_d', 'h1_h2', 'psd',
        # 'rpde', 'dfa'
    ]

    # Process files
    results = []
    for i, wav_path in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] ", end="")
        row = extract_all_parameters(wav_path)
        results.append(row)

    # Write CSV
    print("-" * 60)
    print(f"Writing results to {output_path}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Summary
    n_short = sum(1 for r in results if r.get('flag_short'))
    n_failed = sum(1 for r in results if np.isnan(r.get('voiced_duration_sec', np.nan)))

    print(f"\nSummary:")
    print(f"  Total files processed: {len(results)}")
    print(f"  Short recordings (<0.5s): {n_short}")
    print(f"  Failed to process: {n_failed}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract acoustic parameters from PVQD sustained vowel recordings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameters computed:
  Group A (Basic): F0 mean, jitter local %, shimmer local %, HNR, CPPS
  Group B (AVQI): shimmer local dB, slope, tilt
  Group C (ABI): GNE, Hno-6000 Hz, HNR-D, H1-H2, PSD
  Group D (Spectral): Alpha Ratio

Example:
  python extract_pvqd_params.py -i /home/jorge/PVQD/data -o vowel_params.csv -p "_vowel_a.wav"
        """
    )

    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Directory containing sustained vowel WAV files'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--pattern', '-p',
        default=None,
        help='Filename suffix pattern to filter (e.g., "_vowel_a.wav")'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List files that would be processed without actually processing them'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    wav_files = find_wav_files(args.input_dir, args.pattern)

    if args.dry_run:
        print(f"Dry run - would process {len(wav_files)} files" +
              (f" matching '*{args.pattern}'" if args.pattern else "") + ":")
        for f in wav_files[:10]:
            print(f"  {os.path.basename(f)}")
        if len(wav_files) > 10:
            print(f"  ... and {len(wav_files) - 10} more")
        return

    process_directory(args.input_dir, args.output, args.pattern)


if __name__ == '__main__':
    main()
