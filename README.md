# Optimal Acoustic Parameter Subsets for Dimension-Specific Voice Quality Prediction

Analysis code for the manuscript:

> Lucero, J. C. (2026). Optimal Acoustic Parameter Subsets for Dimension-Specific Voice Quality Prediction. *Journal of Voice* (submitted).

## Overview

This repository contains the Python scripts used to identify optimal acoustic parameter subsets for predicting perceptual voice quality dimensions (severity, breathiness, roughness, strain) from the Perceptual Voice Qualities Database (PVQD).

The analysis pipeline consists of four scripts, designed to be run in order:

1. **`extract_pvqd_params.py`** — Extracts 14 acoustic parameters from PVQD sustained vowel recordings using Parselmouth (Praat 6.1.38).
2. **`qr_analysis.py`** — Performs unsupervised QR factorization (redundancy analysis), supervised OMP parameter ranking per perceptual dimension, and bootstrap stability analysis.
3. **`optimal_n_params.py`** — Determines the BIC-optimal number of parameters for each dimension and fits the final regression models.
4. **`roc_analysis.py`** — Computes diagnostic accuracy (AUC, sensitivity, specificity) with 10-fold stratified cross-validation repeated 100 times.

Four right-skewed parameters (jitter, shimmer %, shimmer dB, PSD) are log-transformed prior to the OMP/regression analyses (scripts 2–4). The transformation is applied on the fly from the raw extracted values.

## Data

The PVQD is publicly available at:

> Walden, P., Mehta, D. D., Engel, L., & Groll, M. (2022). Perceptual Voice Qualities Database (PVQD). Mendeley Data, V4. https://data.mendeley.com/datasets/9dz247gnyb

Download the PVQD and place the sustained vowel WAV files and rating CSV files (CAPE_V.CSV, GRBAS.CSV) in a local directory. Update the file paths in each script's configuration section before running.

## Usage

### Step 1: Extract acoustic parameters

```bash
python extract_pvqd_params.py -i /path/to/pvqd/data -o pvqd_vowel_params.csv -p "_vowel_a.wav"
```

### Step 2: QR factorization and OMP analysis

```bash
python qr_analysis.py --params pvqd_vowel_params.csv
```

Outputs are saved to `results/tables/` (CSV and LaTeX tables) and `results/figures/` (PDF and PNG figures).

### Step 3: Optimal number of parameters

```bash
python optimal_n_params.py
```

Update `PARAMS_CSV`, `CAPE_V_CSV`, and `GRBAS_CSV` paths in the script before running.

### Step 4: ROC analysis

```bash
python roc_analysis.py
```

Update file paths in the script before running.

## Configuration

Each script contains hardcoded file paths near the top of the file (in a clearly marked configuration section). These must be updated to match your local directory structure before running.

## Key Results

Dimension-specific indices (BIC-optimal, log-transformed parameters):

| Dimension | k | Parameters | r_s | AUC |
|-----------|---|-----------|-----|-----|
| Severity | 6 | log(ShimmerDB), GNE, Hno-6000, log(PSD), F0, HNR-D | 0.77 | 0.870 |
| Breathiness | 3 | CPPS, GNE, Hno-6000 | 0.73 | 0.865 |
| Roughness | 5 | log(ShimmerDB), H1-H2, GNE, log(PSD), log(Jitter) | 0.71 | 0.830 |
| Strain | 6 | HNR, F0, log(PSD), Tilt, H1-H2, GNE | 0.66 | 0.790 |

## Requirements

Python 3.9+ is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Parselmouth (required only for Step 1) embeds Praat 6.1.38. The remaining scripts (Steps 2–4) require only standard scientific Python packages and can be run independently from the extracted CSV file.

## Citation

If you use this code, please cite the manuscript:

```
Lucero, J. C. (2026). Optimal Acoustic Parameter Subsets for Dimension-Specific
Voice Quality Prediction. Journal of Voice (submitted).
```

## License

This code is provided for research purposes. See LICENSE for details.

## Contact

Jorge C. Lucero — lucero@unb.br
Department of Computer Science, University of Brasília, Brazil
