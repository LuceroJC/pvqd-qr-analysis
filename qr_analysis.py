#!/usr/bin/env python3
"""
QR Factorization Analysis of PVQD Acoustic Parameters

This script performs QR factorization with column pivoting to rank-order
acoustic parameters by their importance for predicting perceptual voice
quality dimensions from the PVQD database.

Author: Generated for Journal of Voice manuscript
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
#
# Log-transformed parameters: jitter, shimmer (%), shimmer (dB), and PSD are
# log-transformed prior to analysis to linearize their right-skewed distributions,
# consistent with the approximately logarithmic relationship between acoustic
# perturbation and perceived voice quality.
LOG_PARAMS = ['jitter_local_pct', 'shimmer_local_pct', 'shimmer_local_dB', 'psd']

ACOUSTIC_PARAMS = [
    'f0_mean', 'log_jitter_local_pct', 'log_shimmer_local_pct', 'log_shimmer_local_dB',
    'hnr', 'cpps', 'slope', 'tilt', 'alpha_ratio', 'gne',
    'hno_6000', 'hnr_d', 'h1_h2', 'log_psd',
]

CAPE_V_TARGETS = ['CAPE-V Severity', 'CAPE-V Roughness', 'CAPE-V Breathiness', 'CAPE-V Strain']
GRBAS_TARGETS = ['GRBAS Grade', 'GRBAS Roughness', 'GRBAS Breathiness']
ALL_TARGETS = CAPE_V_TARGETS + GRBAS_TARGETS

# Parameter display names for tables/figures
PARAM_DISPLAY = {
    'f0_mean': 'F0 (Hz)',
    'log_jitter_local_pct': 'log Jitter',
    'log_shimmer_local_pct': 'log Shimmer (%)',
    'log_shimmer_local_dB': 'log Shimmer (dB)',
    'hnr': 'HNR',
    'cpps': 'CPPS',
    'slope': 'Slope',
    'tilt': 'Tilt',
    'alpha_ratio': 'Alpha Ratio',
    'gne': 'GNE',
    'hno_6000': 'Hno-6000',
    'hnr_d': 'HNR-D',
    'h1_h2': 'H1-H2',
    'log_psd': 'log PSD',
}

# Index compositions for comparison (mapped to log versions where applicable)
AVQI_COMPONENTS = ['cpps', 'hnr', 'log_shimmer_local_pct', 'log_shimmer_local_dB', 'slope', 'tilt']
ABI_COMPONENTS = AVQI_COMPONENTS + ['log_jitter_local_pct', 'gne', 'hno_6000', 'hnr_d', 'h1_h2', 'log_psd']
CSID_COMPONENTS = ['cpps', 'alpha_ratio']  # L/H ratio is related to alpha ratio


def load_acoustic_params(filepath):
    """Load acoustic parameters CSV."""
    print(f"Loading acoustic parameters from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def add_log_columns(df):
    """Add log-transformed columns for right-skewed parameters."""
    for param in LOG_PARAMS:
        col = df[param]
        if (col <= 0).any():
            n_zero = (col <= 0).sum()
            min_pos = col[col > 0].min()
            print(f"  WARNING: {param} has {n_zero} non-positive values; "
                  f"replacing with half of minimum positive value")
            col = col.clip(lower=min_pos / 2)
        df[f'log_{param}'] = np.log(col)
    return df


def extract_speaker_id(filename):
    """Extract speaker ID from filename for matching with ratings."""
    # Remove file extension and suffixes
    name = str(filename).replace('.wav', '').replace('_vowel_a', '').replace('_speech', '')
    # Remove various separators and standardize
    name = name.replace(' ENSS', '').replace('_ENSS', '').replace('ENSS', '')
    # Strip whitespace
    name = name.strip()
    return name


def load_ratings(cape_v_path, grbas_path):
    """Load and pivot CAPE-V and GRBAS ratings."""
    print(f"Loading CAPE-V ratings from {cape_v_path}...")

    # Read CAPE-V (semicolon separated, European decimal format)
    cape_v = pd.read_csv(cape_v_path, sep=';', decimal=',')
    cape_v.columns = cape_v.columns.str.strip()
    cape_v['File'] = cape_v['File'].str.strip()

    # Pivot to wide format - use the average rating
    cape_v_wide = cape_v.pivot(
        index='File',
        columns='Characteristics',
        values='Average all ratings all times'
    ).reset_index()
    cape_v_wide.columns.name = None

    print(f"  Loaded {len(cape_v_wide)} speakers with CAPE-V ratings")

    print(f"Loading GRBAS ratings from {grbas_path}...")

    # Read GRBAS
    grbas = pd.read_csv(grbas_path, sep=';', decimal=',')
    grbas.columns = grbas.columns.str.strip()
    grbas['File'] = grbas['File'].str.strip()

    # Pivot to wide format
    grbas_wide = grbas.pivot(
        index='File',
        columns='Characteristics',
        values='Average all ratings all times'
    ).reset_index()
    grbas_wide.columns.name = None

    print(f"  Loaded {len(grbas_wide)} speakers with GRBAS ratings")

    # Merge CAPE-V and GRBAS
    ratings = cape_v_wide.merge(grbas_wide, on='File', how='outer')
    print(f"  Combined ratings: {len(ratings)} speakers")

    return ratings


def merge_data(acoustic_df, ratings_df):
    """Merge acoustic parameters with perceptual ratings."""
    print("\nMerging acoustic parameters with ratings...")

    # Extract speaker ID from acoustic filenames
    acoustic_df = acoustic_df.copy()
    acoustic_df['speaker_id'] = acoustic_df['filename'].apply(extract_speaker_id)

    # Clean speaker IDs in ratings
    ratings_df = ratings_df.copy()
    ratings_df['speaker_id'] = ratings_df['File'].str.strip()

    # Merge
    merged = acoustic_df.merge(ratings_df, on='speaker_id', how='inner')

    # Report merge statistics
    n_acoustic = len(acoustic_df)
    n_ratings = len(ratings_df)
    n_merged = len(merged)
    n_lost_acoustic = n_acoustic - n_merged
    n_lost_ratings = n_ratings - n_merged

    print(f"  Acoustic records: {n_acoustic}")
    print(f"  Rating records: {n_ratings}")
    print(f"  Successfully merged: {n_merged}")
    print(f"  Lost from acoustic (no ratings): {n_lost_acoustic}")
    print(f"  Lost from ratings (no acoustic): {n_lost_ratings}")

    # Report speakers not matched
    if n_lost_acoustic > 0:
        unmatched_acoustic = set(acoustic_df['speaker_id']) - set(merged['speaker_id'])
        print(f"  Unmatched acoustic IDs (first 10): {list(unmatched_acoustic)[:10]}")

    return merged


def compute_descriptive_stats(df, params):
    """Compute descriptive statistics for acoustic parameters."""
    print("\nComputing descriptive statistics...")

    stats_list = []
    for param in params:
        valid = df[param].dropna()
        stats_list.append({
            'Parameter': PARAM_DISPLAY.get(param, param),
            'N': len(valid),
            'Mean': valid.mean(),
            'SD': valid.std(),
            'Min': valid.min(),
            'Max': valid.max(),
            'Missing': df[param].isna().sum()
        })

    stats_df = pd.DataFrame(stats_list)
    return stats_df


def compute_correlation_matrix(df, params):
    """Compute Pearson correlation matrix among acoustic parameters."""
    print("Computing correlation matrix...")

    corr_matrix = df[params].corr(method='pearson')
    return corr_matrix


def compute_vif(df, params):
    """Compute Variance Inflation Factors."""
    print("Computing VIF...")

    X = df[params].dropna()
    vif_data = []

    for i, param in enumerate(params):
        try:
            vif = variance_inflation_factor(X.values, i)
        except:
            vif = np.nan
        vif_data.append({
            'Parameter': PARAM_DISPLAY.get(param, param),
            'VIF': vif
        })

    return pd.DataFrame(vif_data)


def compute_condition_number(X):
    """Compute condition number of design matrix."""
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    return s[0] / s[-1]


def qr_with_pivoting(X):
    """
    Compute QR decomposition with column pivoting.

    Returns:
        Q: orthogonal matrix
        R: upper triangular matrix
        pivot_order: permutation indices (0-indexed)
    """
    Q, R, P = linalg.qr(X, pivoting=True, mode='economic')
    return Q, R, P


def _safe_standardize_matrix(X):
    """
    Standardize each column of X using population standard deviation (ddof=0).

    Zero-variance columns are mapped to zeros for numerical stability.
    """
    X = np.asarray(X, dtype=float)
    means = X.mean(axis=0)
    centered = X - means
    stds = X.std(axis=0, ddof=0)

    X_std = np.zeros_like(centered)
    nonzero_std = stds > 0
    X_std[:, nonzero_std] = centered[:, nonzero_std] / stds[nonzero_std]

    return X_std


def _safe_standardize_vector(y):
    """
    Standardize y using population standard deviation (ddof=0).

    Zero-variance targets are mapped to zeros for numerical stability.
    """
    y = np.asarray(y, dtype=float)
    y_centered = y - y.mean()
    y_std = y.std(ddof=0)

    if y_std > 0:
        return y_centered / y_std
    return np.zeros_like(y_centered)


def _absolute_correlation(x, y, method='pearson'):
    """
    Compute absolute Pearson or Spearman correlation with stability guards.

    Returns 0.0 when either vector has zero variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == 'spearman':
        x = stats.rankdata(x)
        y = stats.rankdata(y)
    elif method != 'pearson':
        raise ValueError(f"Unsupported correlation method: {method}")

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)

    if denom == 0:
        return 0.0

    return abs(np.dot(x_centered, y_centered) / denom)


def unsupervised_qr_analysis(df, params):
    """
    Perform unsupervised QR analysis on standardized acoustic parameters.
    This reveals the linear independence structure (redundancy) among parameters.
    """
    print("\n" + "="*60)
    print("UNSUPERVISED QR ANALYSIS (Acoustic Redundancy Structure)")
    print("="*60)

    # Get complete cases
    X_raw = df[params].dropna()
    n_complete = len(X_raw)
    n_total = len(df)
    print(f"Using {n_complete}/{n_total} complete cases")

    # Standardize (z-score)
    X = (X_raw - X_raw.mean()) / X_raw.std()
    X = X.values

    # QR with column pivoting
    Q, R, pivot_order = qr_with_pivoting(X)

    # Extract diagonal of R
    R_diag = np.abs(np.diag(R))

    # Compute relative contribution and cumulative energy
    R_diag_sq = R_diag ** 2
    relative_contrib = R_diag / R_diag[0]
    cumulative_energy = np.cumsum(R_diag_sq) / np.sum(R_diag_sq)

    # Build results table
    results = []
    for k in range(len(params)):
        param_idx = pivot_order[k]
        param_name = params[param_idx]
        results.append({
            'Rank': k + 1,
            'Parameter': PARAM_DISPLAY.get(param_name, param_name),
            'Parameter_raw': param_name,
            '|R_kk|': R_diag[k],
            'Relative |R_kk|/|R_11|': relative_contrib[k],
            'Cumulative Energy': cumulative_energy[k]
        })

    results_df = pd.DataFrame(results)

    print("\nUnsupervised QR Pivot Order (parameter importance by linear independence):")
    print(results_df[['Rank', 'Parameter', '|R_kk|', 'Relative |R_kk|/|R_11|', 'Cumulative Energy']].to_string(index=False))

    return results_df, X, R, pivot_order


def svd_analysis(X, params):
    """
    Perform SVD analysis for comparison with QR.
    """
    print("\n" + "="*60)
    print("SVD ANALYSIS")
    print("="*60)

    # SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Singular value analysis
    s_normalized = s / s[0]
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)

    # Effective rank criteria
    rank_99 = np.searchsorted(cumulative_energy, 0.99) + 1
    rank_ratio = np.sum(s / s[0] >= 0.01)

    print(f"\nSingular values: {s}")
    print(f"Normalized singular values: {s_normalized}")
    print(f"Cumulative energy: {cumulative_energy}")
    print(f"\nEffective rank (99% cumulative energy): {rank_99}")
    print(f"Effective rank (s_k/s_1 >= 0.01): {rank_ratio}")
    print(f"Condition number: {s[0]/s[-1]:.2f}")

    svd_df = pd.DataFrame({
        'Component': range(1, len(s) + 1),
        'Singular Value': s,
        'Normalized': s_normalized,
        'Cumulative Energy': cumulative_energy
    })

    return svd_df, s, rank_99, rank_ratio


def supervised_omp_ranking(df, params, target_name, correlation='pearson', max_features=None):
    """
    Rank acoustic parameters for a target using OMP / forward stepwise regression.

    The method uses complete cases only, standardizes X and y with ddof=0,
    normalizes X columns to unit L2 norm, then iteratively:
    1. Selects the remaining feature with largest absolute correlation with the
       current residual.
    2. Refits least squares on all selected features using np.linalg.lstsq.
    3. Updates the residual.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing acoustic parameters and the perceptual target.
    params : list of str
        Acoustic parameter column names.
    target_name : str
        Target column name.
    correlation : {'pearson', 'spearman'}, default='pearson'
        Correlation type used during feature selection.
    max_features : int or None, default=None
        Optional early stopping limit. If provided, the remaining features are
        appended in descending correlation order with the final residual.

    Returns
    -------
    ranking : dict or None
        Mapping from parameter name to 1-indexed rank.
    selected : list or None
        Parameters in selection order.
    """
    cols_needed = params + [target_name]
    complete_data = df[cols_needed].dropna()

    if len(complete_data) < 10:
        print(f"  Warning: Only {len(complete_data)} complete cases for {target_name}")
        return None, None

    X_raw = complete_data[params].to_numpy(dtype=float)
    y_raw = complete_data[target_name].to_numpy(dtype=float)

    X = _safe_standardize_matrix(X_raw)
    y = _safe_standardize_vector(y_raw)

    norms = np.linalg.norm(X, axis=0)
    nonzero_norm = norms > 0
    X[:, nonzero_norm] = X[:, nonzero_norm] / norms[nonzero_norm]
    X[:, ~nonzero_norm] = 0.0

    residual = y.copy()
    selected = []
    remaining = list(params)
    param_to_idx = {param: idx for idx, param in enumerate(params)}

    n_steps = len(params) if max_features is None else min(max_features, len(params))

    for _ in range(n_steps):
        correlations = [
            _absolute_correlation(X[:, param_to_idx[param]], residual, method=correlation)
            for param in remaining
        ]

        best_idx = int(np.argmax(correlations))
        best_param = remaining.pop(best_idx)
        selected.append(best_param)

        selected_idx = [param_to_idx[param] for param in selected]
        X_selected = X[:, selected_idx]
        beta, _, _, _ = np.linalg.lstsq(X_selected, y, rcond=None)
        residual = y - X_selected @ beta

    if remaining:
        remaining_scores = []
        for order, param in enumerate(remaining):
            corr = _absolute_correlation(X[:, param_to_idx[param]], residual, method=correlation)
            remaining_scores.append((param, corr, order))

        remaining_scores.sort(key=lambda item: (-item[1], item[2]))
        selected.extend([param for param, _, _ in remaining_scores])

    ranking = {param: rank for rank, param in enumerate(selected, start=1)}
    return ranking, selected


def supervised_qr_analysis_weighted(df, params, target_name):
    """
    Supervised QR analysis using correlation-weighted approach.

    Method: Weight columns of X by |Spearman correlation| with y.
    """
    # Get complete cases
    cols_needed = params + [target_name]
    complete_data = df[cols_needed].dropna()

    if len(complete_data) < 10:
        return None

    # Standardize X
    X_raw = complete_data[params]
    y = complete_data[target_name]

    X = (X_raw - X_raw.mean()) / X_raw.std()

    # Compute Spearman correlations
    correlations = {}
    for param in params:
        r, _ = stats.spearmanr(X[param], y)
        correlations[param] = abs(r)

    # Weight columns by |correlation|
    weights = np.array([correlations[p] for p in params])
    X_weighted = X.values * weights

    # QR with column pivoting
    Q, R, pivot_order = qr_with_pivoting(X_weighted)

    # Get parameter rankings
    param_ranks = {}
    for k, idx in enumerate(pivot_order):
        param_name = params[idx]
        param_ranks[param_name] = k + 1

    return param_ranks


def run_all_supervised_analyses(df, params, targets):
    """Run supervised OMP analysis for all perceptual targets."""
    print("\n" + "="*60)
    print("SUPERVISED OMP ANALYSIS (Per Perceptual Dimension)")
    print("="*60)

    results_augmented = {}
    results_weighted = {}
    y_positions = {}

    for target in targets:
        print(f"\nAnalyzing: {target}")

        # Method A: OMP / forward stepwise regression
        ranks_a, selected = supervised_omp_ranking(df, params, target)
        if ranks_a:
            results_augmented[target] = ranks_a
            y_positions[target] = None
            print("  Method A (OMP): residual-based sequential feature selection")
            top_3 = sorted(ranks_a.items(), key=lambda x: x[1])[:3]
            print(f"  Top 3: {[(PARAM_DISPLAY.get(p, p), r) for p, r in top_3]}")

        # Method B: Correlation-weighted
        ranks_b = supervised_qr_analysis_weighted(df, params, target)
        if ranks_b:
            results_weighted[target] = ranks_b

    return results_augmented, results_weighted, y_positions


def bootstrap_qr_analysis(df, params, target=None, n_bootstrap=2000, seed=42):
    """
    Bootstrap stability analysis for QR rankings.

    If target is None, performs unsupervised analysis.
    Otherwise, performs supervised analysis using OMP / forward stepwise regression.
    """
    np.random.seed(seed)

    if target is None:
        # Unsupervised
        complete_data = df[params].dropna()
        X_raw = complete_data
        n = len(X_raw)

        bootstrap_ranks = {p: [] for p in params}

        for b in range(n_bootstrap):
            # Resample
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X_raw.iloc[idx]

            # Standardize
            std = X_boot.std()
            std[std < 1e-8] = 1.0
            X_boot_std = (X_boot - X_boot.mean()) / std
            # X_boot_std = (X_boot - X_boot.mean()) / X_boot.std()

            # QR
            try:
                _, _, pivot_order = qr_with_pivoting(X_boot_std.values)

                # Record ranks
                for rank, param_idx in enumerate(pivot_order):
                    param_name = params[param_idx]
                    bootstrap_ranks[param_name].append(rank + 1)
            except:
                continue

    else:
        # Supervised (OMP / forward stepwise regression)
        cols_needed = params + [target]
        complete_data = df[cols_needed].dropna()
        n = len(complete_data)

        bootstrap_ranks = {p: [] for p in params}

        for b in range(n_bootstrap):
            # Resample
            idx = np.random.choice(n, size=n, replace=True)
            boot_data = complete_data.iloc[idx]

            try:
                ranks, _ = supervised_omp_ranking(boot_data, params, target)
                if ranks is None:
                    continue

                for param_name, rank in ranks.items():
                    bootstrap_ranks[param_name].append(rank)
            except:
                continue

    # Compute statistics
    stability_results = []
    for param in params:
        ranks = bootstrap_ranks[param]
        if len(ranks) > 0:
            # Convert to an array once so the summary statistics are consistent.
            ranks_array = np.asarray(ranks, dtype=float)
            rank_ci_lower, rank_ci_upper = np.percentile(ranks_array, [2.5, 97.5])

            stability_results.append({
                'Parameter': PARAM_DISPLAY.get(param, param),
                'Parameter_raw': param,
                'Mean Rank': np.mean(ranks_array),
                'Median Rank': np.median(ranks_array),
                'SD Rank': np.std(ranks_array),
                'Rank CI Lower': rank_ci_lower,
                'Rank CI Upper': rank_ci_upper,
                '% in Top 3': 100 * np.mean(ranks_array <= 3),
                '% in Top 5': 100 * np.mean(ranks_array <= 5),
                'N Bootstrap': len(ranks_array)
            })

    return pd.DataFrame(stability_results), bootstrap_ranks


def run_bootstrap_analyses(df, params, targets, n_bootstrap=2000):
    """Run bootstrap analysis for unsupervised and all supervised analyses."""
    print("\n" + "="*60)
    print(f"BOOTSTRAP STABILITY ANALYSIS (B = {n_bootstrap})")
    print("="*60)

    all_results = {}
    all_ranks = {}

    # Unsupervised
    print("\nBootstrapping unsupervised QR...")
    stability_unsup, ranks_unsup = bootstrap_qr_analysis(df, params, target=None, n_bootstrap=n_bootstrap)
    all_results['Unsupervised'] = stability_unsup
    all_ranks['Unsupervised'] = ranks_unsup
    print(f"  Done. Mean ranks computed from {stability_unsup['N Bootstrap'].iloc[0]} successful resamples.")

    # Supervised for each target
    for target in targets:
        print(f"\nBootstrapping {target}...")
        stability_sup, ranks_sup = bootstrap_qr_analysis(df, params, target=target, n_bootstrap=n_bootstrap)
        if len(stability_sup) > 0:
            all_results[target] = stability_sup
            all_ranks[target] = ranks_sup
            print(f"  Done. Mean ranks computed from {stability_sup['N Bootstrap'].iloc[0]} successful resamples.")

    return all_results, all_ranks


def create_index_comparison_table(supervised_rankings, params):
    """Compare QR rankings with existing index compositions (AVQI, ABI, CSID)."""
    print("\n" + "="*60)
    print("COMPARISON WITH EXISTING INDEX COMPOSITIONS")
    print("="*60)

    comparison_data = []

    for param in params:
        row = {
            'Parameter': PARAM_DISPLAY.get(param, param),
            'In AVQI': 'Yes' if param in AVQI_COMPONENTS else 'No',
            'In ABI': 'Yes' if param in ABI_COMPONENTS else 'No',
            'In CSID': 'Yes' if param in CSID_COMPONENTS else 'No'
        }

        # Add QR rankings for key targets
        if 'CAPE-V Severity' in supervised_rankings:
            ranks = supervised_rankings['CAPE-V Severity']
            row['QR Rank (Severity)'] = ranks.get(param, np.nan)

        if 'CAPE-V Breathiness' in supervised_rankings:
            ranks = supervised_rankings['CAPE-V Breathiness']
            row['QR Rank (Breathiness)'] = ranks.get(param, np.nan)

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by QR Rank for Severity
    if 'QR Rank (Severity)' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('QR Rank (Severity)')

    print(comparison_df.to_string(index=False))

    return comparison_df


def save_table_csv(df, filepath, index=False):
    """Save DataFrame to CSV."""
    df.to_csv(filepath, index=index)
    print(f"  Saved: {filepath}")


def save_table_latex(df, filepath, caption="", label=""):
    """Save DataFrame as LaTeX table with booktabs style."""
    latex_str = df.to_latex(
        index=False,
        escape=True,
        column_format='l' + 'r' * (len(df.columns) - 1),
        float_format="%.3f"
    )

    # Add booktabs commands
    latex_str = latex_str.replace('\\toprule', '\\toprule')
    latex_str = latex_str.replace('\\midrule', '\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')

    # Add caption and label
    if caption or label:
        latex_str = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{latex_str}\\end{{table}}"

    with open(filepath, 'w') as f:
        f.write(latex_str)
    print(f"  Saved: {filepath}")


def make_safe_name(name):
    """Convert analysis names into filesystem-safe stems."""
    return name.replace(' ', '_').replace('-', '_').replace('/', '_')


def compute_target_correlations(df, params, target):
    """
    Compute Spearman correlations between each acoustic parameter and a target.

    Only complete cases across the full parameter set and target are retained so
    the correlation and OMP comparisons are based on the same samples.
    """
    cols_needed = params + [target]
    complete_data = df[cols_needed].dropna()

    correlation_rows = []
    for param in params:
        # Spearman correlation is robust to monotonic but non-linear scaling.
        r, _ = stats.spearmanr(complete_data[param], complete_data[target])
        if np.isnan(r):
            r = 0.0

        correlation_rows.append({
            'Parameter_raw': param,
            'Parameter': PARAM_DISPLAY.get(param, param),
            'Spearman r': r,
            '|Spearman r|': abs(r)
        })

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_df = correlation_df.sort_values(
        by=['|Spearman r|', 'Parameter'],
        ascending=[False, True]
    ).reset_index(drop=True)

    return correlation_df


def create_correlation_vs_omp_table(correlation_df, ranking_dict):
    """
    Merge univariate Spearman correlations with multivariate OMP ranks.

    The returned table keeps raw identifiers for internal joins and display
    labels for exported tables and annotated figures.
    """
    ranking_df = pd.DataFrame({
        'Parameter_raw': list(ranking_dict.keys()),
        'OMP Rank': list(ranking_dict.values())
    })

    comparison_df = correlation_df.merge(ranking_df, on='Parameter_raw', how='left')
    comparison_df = comparison_df.sort_values(
        by=['OMP Rank', '|Spearman r|'],
        ascending=[True, False],
        na_position='last'
    ).reset_index(drop=True)

    return comparison_df


def plot_correlation_vs_omp(comparison_df, target, output_dir):
    """
    Plot univariate association strength against OMP rank for one target.

    Lower OMP ranks appear at the top so the plot can be read as a ranking
    figure while still showing the absolute univariate correlation on x.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter the absolute correlation magnitude against the supervised rank.
    ax.scatter(
        comparison_df['|Spearman r|'],
        comparison_df['OMP Rank'],
        color='steelblue',
        edgecolor='black',
        s=70,
        alpha=0.85
    )

    # Label each point directly to preserve parameter identity in the figure.
    for _, row in comparison_df.iterrows():
        ax.annotate(
            row['Parameter'],
            (row['|Spearman r|'], row['OMP Rank']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    ax.set_xlabel('|Spearman r|', fontsize=12)
    ax.set_ylabel('OMP Rank', fontsize=12)
    ax.set_title(f'Correlation vs OMP Ranking: {target}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    safe_name = make_safe_name(target)
    fig.savefig(output_dir / f'correlation_vs_omp_{safe_name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'correlation_vs_omp_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: correlation_vs_omp_{safe_name}.pdf/png")


def plot_unsupervised_vs_supervised(unsupervised_df, supervised_rankings, target, output_dir):
    """
    Plot unsupervised QR rank against supervised OMP rank for one target.

    The diagonal highlights agreement, while deviations from the line show
    parameters that are either redundant or especially target-relevant.
    """
    if target not in supervised_rankings:
        return

    supervised_df = pd.DataFrame({
        'Parameter_raw': list(supervised_rankings[target].keys()),
        'OMP Rank': list(supervised_rankings[target].values())
    })

    comparison_df = unsupervised_df[['Parameter_raw', 'Parameter', 'Rank']].merge(
        supervised_df,
        on='Parameter_raw',
        how='inner'
    ).rename(columns={'Rank': 'Unsupervised QR Rank'})

    fig, ax = plt.subplots(figsize=(8, 6))

    # Compare redundancy-driven ordering with target-aware supervised ordering.
    ax.scatter(
        comparison_df['Unsupervised QR Rank'],
        comparison_df['OMP Rank'],
        color='darkorange',
        edgecolor='black',
        s=70,
        alpha=0.85
    )

    # Add a diagonal to show where the two ranking schemes would agree exactly.
    rank_min = 1
    rank_max = max(
        comparison_df['Unsupervised QR Rank'].max(),
        comparison_df['OMP Rank'].max()
    )
    ax.plot([rank_min, rank_max], [rank_min, rank_max], linestyle='--', color='gray', alpha=0.7)

    # Label each point with the display name for interpretability.
    for _, row in comparison_df.iterrows():
        ax.annotate(
            row['Parameter'],
            (row['Unsupervised QR Rank'], row['OMP Rank']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    ax.set_xlabel('Unsupervised QR Rank', fontsize=12)
    ax.set_ylabel('OMP Rank', fontsize=12)
    ax.set_title(f'Unsupervised vs Supervised Ranking: {target}', fontsize=14)
    ax.set_xlim(0.5, rank_max + 0.5)
    ax.set_ylim(0.5, rank_max + 0.5)
    ax.set_xticks(range(rank_min, rank_max + 1))
    ax.set_yticks(range(rank_min, rank_max + 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_name = make_safe_name(target)
    fig.savefig(output_dir / f'unsupervised_vs_supervised_{safe_name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / f'unsupervised_vs_supervised_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: unsupervised_vs_supervised_{safe_name}.pdf/png")


def run_correlation_vs_omp_analyses(df, params, supervised_rankings, tables_dir, figures_dir):
    """
    Export per-target correlation tables and correlation-vs-OMP figures.

    Each table contrasts simple univariate association with the multivariate,
    residual-based OMP ordering for the same perceptual target.
    """
    comparison_results = {}

    for target, ranking_dict in supervised_rankings.items():
        print(f"\nCorrelation vs OMP comparison: {target}")

        # Compute univariate correlations on the same complete-case sample.
        correlation_df = compute_target_correlations(df, params, target)

        # Join the correlations to the supervised ranking for table export.
        comparison_df = create_correlation_vs_omp_table(correlation_df, ranking_dict)
        comparison_results[target] = comparison_df

        safe_name = make_safe_name(target)
        save_table_csv(
            comparison_df[['Parameter', 'Spearman r', '|Spearman r|', 'OMP Rank']],
            tables_dir / f'correlation_vs_omp_{safe_name}.csv'
        )
        plot_correlation_vs_omp(comparison_df, target, figures_dir)

    return comparison_results


def plot_scree(svd_df, s, rank_99, rank_ratio, output_dir):
    """Plot singular value decay (scree plot)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(1, len(s) + 1)

    # Singular values
    ax.plot(x, s, 'o-', linewidth=2, markersize=8, label='Singular values')

    # Mark effective ranks
    ax.axvline(x=rank_99, color='red', linestyle='--', alpha=0.7,
               label=f'99% energy (k={rank_99})')
    ax.axvline(x=rank_ratio, color='green', linestyle=':', alpha=0.7,
               label=f's_k/s_1 >= 0.01 (k={rank_ratio})')

    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Singular Value', fontsize=12)
    ax.set_title('Singular Value Decay (Scree Plot)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'scree_plot.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: scree_plot.pdf/png")


def plot_qr_diagonal_decay(unsupervised_df, output_dir):
    """Plot QR diagonal decay for unsupervised analysis."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = unsupervised_df['Rank']
    y = unsupervised_df['Relative |R_kk|/|R_11|']
    labels = unsupervised_df['Parameter']

    bars = ax.bar(x, y, color='steelblue', edgecolor='black', alpha=0.8)

    # Add parameter labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel('Pivot Position (Rank)', fontsize=12)
    ax.set_ylabel('Relative Contribution |R_kk| / |R_11|', fontsize=12)
    ax.set_title('Unsupervised QR Diagonal Decay', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add cumulative energy line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(x, unsupervised_df['Cumulative Energy'], 'ro-', linewidth=2,
             markersize=6, label='Cumulative Energy')
    ax2.set_ylabel('Cumulative Energy', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')

    plt.tight_layout()

    fig.savefig(output_dir / 'qr_diagonal_decay.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'qr_diagonal_decay.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: qr_diagonal_decay.pdf/png")


def plot_rankings_heatmap(supervised_rankings, params, output_dir):
    """Plot heatmap of parameter rankings across perceptual dimensions."""
    # Build matrix
    targets = list(supervised_rankings.keys())

    # Create display names for parameters sorted by mean rank
    mean_ranks = {}
    for param in params:
        ranks = [supervised_rankings[t].get(param, len(params)) for t in targets if t in supervised_rankings]
        mean_ranks[param] = np.mean(ranks)

    sorted_params = sorted(params, key=lambda p: mean_ranks[p])

    # Build matrix
    matrix = np.zeros((len(sorted_params), len(targets)))
    for i, param in enumerate(sorted_params):
        for j, target in enumerate(targets):
            if target in supervised_rankings and param in supervised_rankings[target]:
                matrix[i, j] = supervised_rankings[target][param]
            else:
                matrix[i, j] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use reversed colormap (lower rank = better = darker)
    cmap = sns.color_palette("YlOrRd_r", as_cmap=True)

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap=cmap,
        xticklabels=[t.replace('CAPE-V ', 'CV-').replace('GRBAS ', 'G-') for t in targets],
        yticklabels=[PARAM_DISPLAY.get(p, p) for p in sorted_params],
        cbar_kws={'label': 'Rank (lower = more important)'},
        ax=ax,
        vmin=1,
        vmax=len(params),
        linewidths=0.5
    )

    ax.set_xlabel('Perceptual Dimension', fontsize=12)
    ax.set_ylabel('Acoustic Parameter', fontsize=12)
    ax.set_title('QR Parameter Rankings Across Perceptual Dimensions', fontsize=14)

    plt.tight_layout()

    fig.savefig(output_dir / 'rankings_heatmap.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'rankings_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: rankings_heatmap.pdf/png")


def plot_bootstrap_stability(bootstrap_results, bootstrap_ranks, output_dir):
    """Plot bootstrap rank distributions."""
    n_plots = len(bootstrap_results)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, (analysis_name, ranks_dict) in enumerate(bootstrap_ranks.items()):
        ax = axes[idx]

        # Get parameters sorted by mean rank
        stability_df = bootstrap_results[analysis_name]
        sorted_params = stability_df.sort_values('Mean Rank')['Parameter_raw'].tolist()

        # Prepare data for box plot
        data_for_plot = []
        labels = []
        for param in sorted_params:
            if param in ranks_dict and len(ranks_dict[param]) > 0:
                data_for_plot.append(ranks_dict[param])
                labels.append(PARAM_DISPLAY.get(param, param))

        # Box plot
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data_for_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Rank', fontsize=10)
        ax.set_title(analysis_name, fontsize=11)
        ax.set_ylim(0.5, 14.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.invert_yaxis()  # Lower rank = better = top

    # Remove empty subplots
    for idx in range(len(bootstrap_results), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Bootstrap Rank Distributions (B=2000)', fontsize=14, y=1.02)
    plt.tight_layout()

    fig.savefig(output_dir / 'bootstrap_stability.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'bootstrap_stability.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: bootstrap_stability.pdf/png")


def plot_correlation_matrix(corr_matrix, params, output_dir):
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display names
    labels = [PARAM_DISPLAY.get(p, p) for p in params]

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Pearson r'},
        ax=ax,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={'fontsize': 8}
    )

    ax.set_title('Correlation Matrix of Acoustic Parameters', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()

    fig.savefig(output_dir / 'correlation_matrix.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: correlation_matrix.pdf/png")


def write_summary_report(output_dir, merged_df, params, targets, unsupervised_df, svd_df,
                         supervised_rankings, y_positions, bootstrap_results,
                         rank_99, rank_ratio, vif_df):
    """Write comprehensive summary report."""
    report_path = output_dir / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QR FACTORIZATION ANALYSIS OF PVQD ACOUSTIC PARAMETERS\n")
        f.write("Summary Report for Journal of Voice Manuscript\n")
        f.write("="*70 + "\n\n")

        # Data summary
        f.write("1. DATA SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total records after merge: {len(merged_df)}\n")
        f.write(f"Number of acoustic parameters: {len(params)}\n")
        f.write(f"Perceptual targets analyzed: {len(targets)}\n")

        # Complete cases
        complete_cases = merged_df[params].dropna()
        f.write(f"Complete cases (all {len(params)} params): {len(complete_cases)}\n")

        # Missing data
        f.write("\nMissing data per parameter:\n")
        for param in params:
            n_missing = merged_df[param].isna().sum()
            if n_missing > 0:
                f.write(f"  {PARAM_DISPLAY.get(param, param)}: {n_missing}\n")

        # SVD effective rank
        f.write("\n\n2. MULTICOLLINEARITY & EFFECTIVE RANK\n")
        f.write("-"*40 + "\n")
        f.write(f"Effective rank (99% cumulative energy): {rank_99}\n")
        f.write(f"Effective rank (s_k/s_1 >= 0.01): {rank_ratio}\n")

        # VIF summary
        f.write("\nVariance Inflation Factors:\n")
        for _, row in vif_df.iterrows():
            vif_val = row['VIF']
            flag = " [HIGH]" if vif_val > 10 else ""
            f.write(f"  {row['Parameter']}: {vif_val:.2f}{flag}\n")

        # Unsupervised QR results
        f.write("\n\n3. UNSUPERVISED QR ANALYSIS (Linear Independence Structure)\n")
        f.write("-"*40 + "\n")
        f.write("Parameter ranking by QR pivot order:\n")
        for _, row in unsupervised_df.iterrows():
            f.write(f"  {row['Rank']:2d}. {row['Parameter']:15s}  "
                   f"|R_kk|={row['|R_kk|']:8.3f}  "
                   f"Rel={row['Relative |R_kk|/|R_11|']:.3f}  "
                   f"CumE={row['Cumulative Energy']:.3f}\n")

        # Supervised OMP results
        f.write("\n\n4. SUPERVISED OMP ANALYSIS (Per Perceptual Dimension)\n")
        f.write("-"*40 + "\n")

        for target in targets:
            if target in supervised_rankings:
                f.write(f"\n{target}:\n")
                if y_positions.get(target) is not None:
                    f.write(f"  y position in pivot order: {y_positions[target]}\n")

                ranks = supervised_rankings[target]
                sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])

                f.write("  Parameter rankings:\n")
                for param, rank in sorted_ranks:
                    f.write(f"    {rank:2d}. {PARAM_DISPLAY.get(param, param)}\n")

        # Bootstrap stability summary
        f.write("\n\n5. BOOTSTRAP STABILITY (B=2000)\n")
        f.write("-"*40 + "\n")

        for analysis_name, stability_df in bootstrap_results.items():
            f.write(f"\n{analysis_name}:\n")
            f.write("  " + "-"*60 + "\n")
            f.write(f"  {'Parameter':15s} {'Mean Rank':>10s} {'SD':>8s} {'%Top3':>8s} {'%Top5':>8s}\n")
            f.write("  " + "-"*60 + "\n")

            for _, row in stability_df.sort_values('Mean Rank').iterrows():
                f.write(f"  {row['Parameter']:15s} {row['Mean Rank']:10.2f} "
                       f"{row['SD Rank']:8.2f} {row['% in Top 3']:8.1f} {row['% in Top 5']:8.1f}\n")

        # Key findings
        f.write("\n\n6. KEY FINDINGS\n")
        f.write("-"*40 + "\n")

        # Most stable predictors for Overall Severity
        if 'CAPE-V Severity' in bootstrap_results:
            sev_stability = bootstrap_results['CAPE-V Severity'].sort_values('Mean Rank')
            top_3_sev = sev_stability.head(3)['Parameter'].tolist()
            f.write(f"Top 3 most stable predictors of Overall Severity:\n")
            for i, param in enumerate(top_3_sev):
                f.write(f"  {i+1}. {param}\n")

        # Most stable predictors for Breathiness
        if 'CAPE-V Breathiness' in bootstrap_results:
            breath_stability = bootstrap_results['CAPE-V Breathiness'].sort_values('Mean Rank')
            top_3_breath = breath_stability.head(3)['Parameter'].tolist()
            f.write(f"\nTop 3 most stable predictors of Breathiness:\n")
            for i, param in enumerate(top_3_breath):
                f.write(f"  {i+1}. {param}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")

    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='QR Factorization Analysis of PVQD Acoustic Parameters')
    parser.add_argument('--params', type=str,
                        default='/home/jorge/pvqd_analysis/pvqd_vowel_params.csv',
                        help='Path to acoustic parameters CSV')
    parser.add_argument('--cape_v', type=str,
                        default='/home/jorge/PVQD/CAPE_V.CSV',
                        help='Path to CAPE-V ratings CSV')
    parser.add_argument('--grbas', type=str,
                        default='/home/jorge/PVQD/GRBAS.CSV',
                        help='Path to GRBAS ratings CSV')
    parser.add_argument('--output', type=str,
                        default='/home/jorge/pvqd_analysis/results/',
                        help='Output directory')
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='Number of bootstrap resamples')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output)
    tables_dir = output_dir / 'tables'
    figures_dir = output_dir / 'figures'

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    print("="*70)
    print("QR FACTORIZATION ANALYSIS OF PVQD ACOUSTIC PARAMETERS")
    print("="*70)

    # Step 1: Load data
    acoustic_df = load_acoustic_params(args.params)
    acoustic_df = add_log_columns(acoustic_df)
    ratings_df = load_ratings(args.cape_v, args.grbas)
    merged_df = merge_data(acoustic_df, ratings_df)

    # Step 2: Descriptive statistics
    desc_stats = compute_descriptive_stats(merged_df, ACOUSTIC_PARAMS)
    print("\nDescriptive Statistics:")
    print(desc_stats.to_string(index=False))

    save_table_csv(desc_stats, tables_dir / 'descriptive_stats.csv')
    save_table_latex(desc_stats, tables_dir / 'descriptive_stats.tex',
                     caption="Descriptive statistics for acoustic parameters",
                     label="tab:descriptive")

    # Step 3: Correlation structure
    corr_matrix = compute_correlation_matrix(merged_df, ACOUSTIC_PARAMS)
    vif_df = compute_vif(merged_df, ACOUSTIC_PARAMS)

    save_table_csv(corr_matrix, tables_dir / 'correlation_matrix.csv', index=True)
    save_table_csv(vif_df, tables_dir / 'vif.csv')

    plot_correlation_matrix(corr_matrix, ACOUSTIC_PARAMS, figures_dir)

    # Condition number
    X_complete = merged_df[ACOUSTIC_PARAMS].dropna()
    X_std = (X_complete - X_complete.mean()) / X_complete.std()
    cond_num = compute_condition_number(X_std.values)
    print(f"\nCondition number of design matrix: {cond_num:.2f}")

    # Step 4: Unsupervised QR analysis
    unsupervised_df, X, R, pivot_order = unsupervised_qr_analysis(merged_df, ACOUSTIC_PARAMS)

    save_table_csv(unsupervised_df, tables_dir / 'unsupervised_qr.csv')
    save_table_latex(
        unsupervised_df[['Rank', 'Parameter', '|R_kk|', 'Relative |R_kk|/|R_11|', 'Cumulative Energy']],
        tables_dir / 'unsupervised_qr.tex',
        caption="Unsupervised QR factorization results",
        label="tab:unsupervised_qr"
    )

    plot_qr_diagonal_decay(unsupervised_df, figures_dir)

    # Step 5: SVD analysis
    svd_df, singular_values, rank_99, rank_ratio = svd_analysis(X, ACOUSTIC_PARAMS)

    save_table_csv(svd_df, tables_dir / 'svd_analysis.csv')
    plot_scree(svd_df, singular_values, rank_99, rank_ratio, figures_dir)

    # Step 6: Supervised OMP analysis
    supervised_rankings, supervised_weighted, y_positions = run_all_supervised_analyses(
        merged_df, ACOUSTIC_PARAMS, ALL_TARGETS
    )

    # Create rankings summary table
    rankings_summary = []
    for param in ACOUSTIC_PARAMS:
        row = {'Parameter': PARAM_DISPLAY.get(param, param)}
        for target in ALL_TARGETS:
            if target in supervised_rankings:
                row[target.replace('CAPE-V ', 'CV-').replace('GRBAS ', 'G-')] = \
                    supervised_rankings[target].get(param, np.nan)
        rankings_summary.append(row)

    rankings_df = pd.DataFrame(rankings_summary)
    save_table_csv(rankings_df, tables_dir / 'supervised_rankings.csv')
    save_table_latex(rankings_df, tables_dir / 'supervised_rankings.tex',
                     caption="QR parameter rankings for each perceptual dimension",
                     label="tab:supervised_rankings")

    plot_rankings_heatmap(supervised_rankings, ACOUSTIC_PARAMS, figures_dir)

    # Step 7: Correlation-vs-OMP comparisons
    correlation_vs_omp_results = run_correlation_vs_omp_analyses(
        merged_df, ACOUSTIC_PARAMS, supervised_rankings, tables_dir, figures_dir
    )

    # Compare redundancy-driven QR ordering with one target-aware OMP ranking.
    comparison_target = 'CAPE-V Severity' if 'CAPE-V Severity' in supervised_rankings else next(
        iter(supervised_rankings),
        None
    )
    if comparison_target is not None:
        plot_unsupervised_vs_supervised(
            unsupervised_df, supervised_rankings, comparison_target, figures_dir
        )

    # Step 8: Bootstrap stability analysis
    bootstrap_results, bootstrap_ranks = run_bootstrap_analyses(
        merged_df, ACOUSTIC_PARAMS, ALL_TARGETS, n_bootstrap=args.bootstrap
    )

    # Save bootstrap results
    for analysis_name, stability_df in bootstrap_results.items():
        safe_name = make_safe_name(analysis_name)
        save_table_csv(stability_df, tables_dir / f'bootstrap_{safe_name}.csv')

    plot_bootstrap_stability(bootstrap_results, bootstrap_ranks, figures_dir)

    # Step 9: Index comparison
    comparison_df = create_index_comparison_table(supervised_rankings, ACOUSTIC_PARAMS)
    save_table_csv(comparison_df, tables_dir / 'index_comparison.csv')
    save_table_latex(comparison_df, tables_dir / 'index_comparison.tex',
                     caption="Comparison of QR rankings with existing voice quality indices",
                     label="tab:index_comparison")

    # Write summary report
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)

    write_summary_report(
        output_dir, merged_df, ACOUSTIC_PARAMS, ALL_TARGETS,
        unsupervised_df, svd_df, supervised_rankings, y_positions,
        bootstrap_results, rank_99, rank_ratio, vif_df
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == '__main__':
    main()
