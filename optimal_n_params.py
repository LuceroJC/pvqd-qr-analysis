#!/usr/bin/env python3
"""
Optimal Number of Parameters for Perceptual Voice Quality Mini-Indices

This script determines the data-driven optimal number of acoustic parameters
for each perceptual dimension, using the OMP ordering from the main QR analysis.

Three criteria are evaluated:
  1. Adjusted R² (stops when adding parameters no longer improves fit)
  2. BIC (Bayesian Information Criterion — penalizes model complexity)
  3. 10-fold cross-validated RMSE (out-of-sample prediction error)

The script also fits the final optimal models and reports their formulas,
coefficients, and Spearman correlations with the perceptual targets.

Usage:
  python optimal_n_params.py

Requires: numpy, pandas, scipy, sklearn, matplotlib
Input files: same as qr_analysis.py (paths hardcoded below)

Author: Jorge C. Lucero
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Configuration — adjust paths as needed
# ============================================================

PARAMS_CSV = '/home/jorge/pvqd_analysis/pvqd_vowel_params.csv'
CAPE_V_CSV = '/home/jorge/PVQD/CAPE_V.CSV'
GRBAS_CSV  = '/home/jorge/PVQD/GRBAS.CSV'
OUTPUT_DIR = '/home/jorge/pvqd_analysis/results/optimal_n'

# Log-transformed parameters
LOG_PARAMS = ['jitter_local_pct', 'shimmer_local_pct', 'shimmer_local_dB', 'psd']

ALL_PARAMS = [
    'cpps', 'hnr_d', 'hno_6000', 'gne', 'log_shimmer_local_pct', 'h1_h2',
    'hnr', 'f0_mean', 'log_psd', 'log_jitter_local_pct', 'log_shimmer_local_dB',
    'slope', 'tilt', 'alpha_ratio',
]

TARGETS = [
    'CAPE-V Severity', 'CAPE-V Breathiness', 'CAPE-V Roughness', 'CAPE-V Strain',
    'GRBAS Grade', 'GRBAS Breathiness', 'GRBAS Roughness',
]

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

MAX_K = 10  # evaluate up to 10 parameters
N_FOLDS = 10
N_CV_REPEATS = 20  # repeat CV for stability
SEED = 42


# ============================================================
# Data loading (same logic as qr_analysis.py)
# ============================================================

def extract_speaker_id(filename):
    name = str(filename).replace('.wav', '').replace('_vowel_a', '')
    for suffix in [' ENSS', '_ENSS', 'ENSS', '_E_NSS', ' E_NSS',
                   '_eNSS', ' eNSS', '_enss', ' enss']:
        name = name.replace(suffix, '')
    name = name.rstrip('.').strip()
    return name


def load_and_merge():
    """Load acoustic params + ratings, merge, return DataFrame."""
    acoustic = pd.read_csv(PARAMS_CSV)
    acoustic['speaker_id'] = acoustic['filename'].apply(extract_speaker_id)

    # Log-transform right-skewed parameters
    for param in LOG_PARAMS:
        col = acoustic[param]
        if (col <= 0).any():
            min_pos = col[col > 0].min()
            col = col.clip(lower=min_pos / 2)
        acoustic[f'log_{param}'] = np.log(col)

    cape_v = pd.read_csv(CAPE_V_CSV, sep=';', decimal=',')
    cape_v.columns = cape_v.columns.str.strip()
    cape_v['File'] = cape_v['File'].str.strip()
    cape_v_wide = cape_v.pivot(
        index='File', columns='Characteristics',
        values='Average all ratings all times'
    ).reset_index()
    cape_v_wide.columns.name = None
    cape_v_wide.rename(columns={'File': 'speaker_id'}, inplace=True)

    grbas = pd.read_csv(GRBAS_CSV, sep=';', decimal=',')
    grbas.columns = grbas.columns.str.strip()
    grbas['File'] = grbas['File'].str.strip()
    grbas_wide = grbas.pivot(
        index='File', columns='Characteristics',
        values='Average all ratings all times'
    ).reset_index()
    grbas_wide.columns.name = None
    grbas_wide.rename(columns={'File': 'speaker_id'}, inplace=True)

    ratings = cape_v_wide.merge(grbas_wide, on='speaker_id', how='outer')
    df = acoustic.merge(ratings, on='speaker_id', how='inner')
    print(f"Merged records: {len(df)}")
    return df


# ============================================================
# OMP ordering
# ============================================================

def omp_ordering(X, y):
    """
    Orthogonal Matching Pursuit: return column indices in selection order.
    Uses standardized X and y, L2-normalized columns.
    """
    n, p = X.shape
    # Standardize
    mu_x, sd_x = X.mean(axis=0), X.std(axis=0, ddof=0)
    sd_x[sd_x == 0] = 1.0
    X_s = (X - mu_x) / sd_x

    mu_y, sd_y = y.mean(), y.std(ddof=0)
    y_s = (y - mu_y) / sd_y if sd_y > 0 else np.zeros_like(y)

    # Normalize columns to unit L2 norm
    norms = np.linalg.norm(X_s, axis=0)
    norms[norms == 0] = 1.0
    X_s = X_s / norms

    residual = y_s.copy()
    selected = []
    remaining = list(range(p))

    for _ in range(p):
        corrs = []
        for j in remaining:
            xj = X_s[:, j]
            if np.std(xj) == 0 or np.std(residual) == 0:
                corrs.append(0.0)
            else:
                r = np.dot(xj - xj.mean(), residual - residual.mean())
                r /= (np.linalg.norm(xj - xj.mean()) * np.linalg.norm(residual - residual.mean()))
                corrs.append(abs(r))

        best = remaining[int(np.argmax(corrs))]
        selected.append(best)
        remaining.remove(best)

        X_sel = X_s[:, selected]
        beta, _, _, _ = np.linalg.lstsq(X_sel, y_s, rcond=None)
        residual = y_s - X_sel @ beta

    return selected


# ============================================================
# Model evaluation at each k
# ============================================================

def evaluate_models(X_raw, y_raw, omp_order, max_k=MAX_K):
    """
    For k = 1, 2, ..., max_k parameters (in OMP order), compute:
      - R², adjusted R²
      - BIC
      - 10-fold CV RMSE (repeated N_CV_REPEATS times for stability)
      - Spearman correlation between predicted and actual
    
    Returns a list of dicts, one per k.
    """
    n = len(y_raw)
    max_k = min(max_k, len(omp_order))
    results = []

    for k in range(1, max_k + 1):
        cols = omp_order[:k]
        X = X_raw[:, cols]

        # --- OLS fit on full data ---
        X_int = np.column_stack([np.ones(n), X])
        beta, _, _, _ = np.linalg.lstsq(X_int, y_raw, rcond=None)
        y_pred = X_int @ beta

        ss_res = np.sum((y_raw - y_pred)**2)
        ss_tot = np.sum((y_raw - np.mean(y_raw))**2)

        r2 = 1.0 - ss_res / ss_tot
        p_model = k + 1  # k predictors + intercept
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p_model)

        # BIC = n * ln(ss_res/n) + p * ln(n)
        bic = n * np.log(ss_res / n) + p_model * np.log(n)

        # Spearman
        rs, _ = stats.spearmanr(y_raw, y_pred)

        # --- Repeated k-fold CV ---
        cv_rmses = []
        rng = np.random.RandomState(SEED)
        for rep in range(N_CV_REPEATS):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rng.randint(0, 1_000_000))
            fold_errors = []
            for train_idx, test_idx in kf.split(X):
                X_tr = np.column_stack([np.ones(len(train_idx)), X[train_idx]])
                X_te = np.column_stack([np.ones(len(test_idx)), X[test_idx]])
                y_tr, y_te = y_raw[train_idx], y_raw[test_idx]

                b, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
                y_hat = X_te @ b
                fold_errors.append(np.mean((y_te - y_hat)**2))

            cv_rmses.append(np.sqrt(np.mean(fold_errors)))

        cv_rmse_mean = np.mean(cv_rmses)
        cv_rmse_sd = np.std(cv_rmses)

        results.append({
            'k': k,
            'R2': r2,
            'Adj_R2': adj_r2,
            'BIC': bic,
            'CV_RMSE_mean': cv_rmse_mean,
            'CV_RMSE_sd': cv_rmse_sd,
            'Spearman_rs': rs,
        })

    return results


def find_optimal_k(eval_results):
    """
    Determine optimal k by three criteria:
      1. Adj R²: largest k where Δ(Adj R²) ≥ 0.01 (1% improvement threshold)
      2. BIC: k with minimum BIC
      3. CV-RMSE: k with minimum mean CV-RMSE (within 1-SE rule)
    Returns dict with optimal k for each criterion.
    """
    df = pd.DataFrame(eval_results)

    # 1. Adjusted R²: find where improvement drops below 1%
    adj_r2_vals = df['Adj_R2'].values
    best_adjr2_k = 1
    for i in range(1, len(adj_r2_vals)):
        delta = adj_r2_vals[i] - adj_r2_vals[i - 1]
        if delta >= 0.01:
            best_adjr2_k = i + 1  # 1-indexed

    # 2. BIC: minimum
    best_bic_k = int(df.loc[df['BIC'].idxmin(), 'k'])

    # 3. CV-RMSE: 1-SE rule (smallest k within 1 SE of minimum)
    cv_means = df['CV_RMSE_mean'].values
    cv_sds = df['CV_RMSE_sd'].values
    min_idx = np.argmin(cv_means)
    threshold = cv_means[min_idx] + cv_sds[min_idx]
    best_cv_k = 1
    for i in range(len(cv_means)):
        if cv_means[i] <= threshold:
            best_cv_k = i + 1  # 1-indexed
            break

    return {
        'adj_r2': best_adjr2_k,
        'bic': best_bic_k,
        'cv_1se': best_cv_k,
    }


# ============================================================
# Final model fitting
# ============================================================

def fit_final_model(X_raw, y_raw, col_indices, param_names, target_name):
    """Fit OLS model with selected parameters, report full results."""
    n = len(y_raw)
    k = len(col_indices)
    X = X_raw[:, col_indices]
    predictors = [param_names[i] for i in col_indices]

    X_int = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_int, y_raw, rcond=None)
    y_pred = X_int @ beta

    ss_res = np.sum((y_raw - y_pred)**2)
    ss_tot = np.sum((y_raw - np.mean(y_raw))**2)
    r2 = 1.0 - ss_res / ss_tot
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)

    mse = ss_res / (n - k - 1)
    try:
        cov_beta = mse * np.linalg.inv(X_int.T @ X_int)
        se = np.sqrt(np.diag(cov_beta))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - k - 1)

    r_pearson, p_pearson = stats.pearsonr(y_raw, y_pred)
    r_spearman, p_spearman = stats.spearmanr(y_raw, y_pred)

    return {
        'target': target_name,
        'predictors': predictors,
        'n': n,
        'k': k,
        'beta': beta,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'r2': r2,
        'adj_r2': adj_r2,
        'r_pearson': r_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
    }


def print_model_results(model):
    """Print formatted model results."""
    preds = model['predictors']
    disp = [PARAM_DISPLAY.get(p, p) for p in preds]

    formula_lhs = model['target']
    formula_rhs = ' + '.join([f'β{i+1}·{d}' for i, d in enumerate(disp)])
    print(f"\n{'━'*90}")
    print(f"{formula_lhs} = β₀ + {formula_rhs}")
    print(f"  N = {model['n']},  k = {model['k']} parameters")
    print(f"  R² = {model['r2']:.4f},  Adjusted R² = {model['adj_r2']:.4f}")
    print(f"  r (Pearson)  = {model['r_pearson']:.4f}")
    print(f"  rₛ (Spearman) = {model['r_spearman']:.4f}  (p = {model['p_spearman']:.2e})")

    print(f"\n  {'Coefficient':12s} {'β':>12s} {'SE':>10s} {'t':>10s} {'p':>12s}")
    coef_names = ['Intercept'] + disp
    for cname, b, s, t, p in zip(coef_names, model['beta'], model['se'],
                                  model['t_stats'], model['p_values']):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {cname:12s} {b:12.6f} {s:10.6f} {t:10.4f} {p:12.4e} {sig}")

    # Print formula with numeric coefficients
    parts = [f"{model['beta'][0]:.4f}"]
    for i, pred in enumerate(disp):
        b = model['beta'][i + 1]
        sign = '+' if b >= 0 else ''
        parts.append(f"{sign}{b:.4f}·{pred}")
    print(f"\n  Formula: {model['target']} ≈ {' '.join(parts)}")


# ============================================================
# Plotting
# ============================================================

def plot_model_selection(eval_results, omp_params, target_name, optimal_k, output_dir):
    """Plot Adj R², BIC, and CV-RMSE vs number of parameters."""
    df = pd.DataFrame(eval_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f'{target_name}', fontsize=13, fontweight='bold', y=1.02)

    # x-axis labels: parameter names in OMP order
    k_vals = df['k'].values
    x_labels = [PARAM_DISPLAY.get(p, p) for p in omp_params[:len(k_vals)]]

    # Panel 1: Adjusted R²
    ax = axes[0]
    ax.plot(k_vals, df['Adj_R2'], 'o-', color='#2166ac', linewidth=1.5, markersize=5)
    ax.axvline(optimal_k['adj_r2'], color='#d62728', linestyle='--', alpha=0.7,
               label=f"Optimal k={optimal_k['adj_r2']}")
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Adjusted R²')
    ax.set_title('Adjusted R²')
    ax.set_xticks(k_vals)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: BIC
    ax = axes[1]
    ax.plot(k_vals, df['BIC'], 's-', color='#2ca02c', linewidth=1.5, markersize=5)
    ax.axvline(optimal_k['bic'], color='#d62728', linestyle='--', alpha=0.7,
               label=f"Optimal k={optimal_k['bic']}")
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('BIC')
    ax.set_title('Bayesian Information Criterion')
    ax.set_xticks(k_vals)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: CV-RMSE
    ax = axes[2]
    ax.errorbar(k_vals, df['CV_RMSE_mean'], yerr=df['CV_RMSE_sd'],
                fmt='D-', color='#ff7f0e', linewidth=1.5, markersize=5, capsize=3)
    ax.axvline(optimal_k['cv_1se'], color='#d62728', linestyle='--', alpha=0.7,
               label=f"Optimal k={optimal_k['cv_1se']} (1-SE rule)")
    # Show 1-SE threshold line
    min_idx = df['CV_RMSE_mean'].idxmin()
    threshold = df.loc[min_idx, 'CV_RMSE_mean'] + df.loc[min_idx, 'CV_RMSE_sd']
    ax.axhline(threshold, color='gray', linestyle=':', alpha=0.5, label='1-SE threshold')
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('CV-RMSE')
    ax.set_title('Cross-Validated RMSE')
    ax.set_xticks(k_vals)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = target_name.replace(' ', '_').replace('-', '')
    fig.savefig(output_dir / f'model_selection_{safe_name}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / f'model_selection_{safe_name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*90)
    print("OPTIMAL NUMBER OF PARAMETERS FOR PERCEPTUAL VOICE QUALITY INDICES")
    print("="*90)

    df = load_and_merge()

    # Published references for comparison
    published_refs = {
        'CAPE-V Severity':    ('AVQI (6 params)', 0.73),
        'CAPE-V Breathiness': ('ABI (9 params)', 0.75),
        'CAPE-V Roughness':   ('Best single: Shimmer', 0.68),
        'CAPE-V Strain':      ('(no published index)', None),
        'GRBAS Grade':        ('AVQI (6 params)', 0.75),
        'GRBAS Breathiness':  ('ABI (9 params, GRBAS B)', 0.66),
        'GRBAS Roughness':    ('Best single: Shimmer', 0.69),
    }

    all_summaries = []

    for target_name in TARGETS:
        print(f"\n{'='*90}")
        print(f"TARGET: {target_name}")
        print(f"{'='*90}")

        # Get complete cases
        cols = ALL_PARAMS + [target_name]
        data = df[cols].dropna()
        n = len(data)
        print(f"Complete cases: {n}")

        X_raw = data[ALL_PARAMS].values
        y_raw = data[target_name].values

        # OMP ordering
        order = omp_ordering(X_raw.copy(), y_raw.copy())
        ordered_params = [ALL_PARAMS[i] for i in order]
        ordered_display = [PARAM_DISPLAY.get(p, p) for p in ordered_params]

        print(f"OMP order: {', '.join(ordered_display)}")

        # Evaluate models at each k
        eval_results = evaluate_models(X_raw, y_raw, order, max_k=MAX_K)

        # Find optimal k by each criterion
        optimal_k = find_optimal_k(eval_results)

        print(f"\nOptimal k by criterion:")
        print(f"  Adjusted R² (Δ ≥ 0.01):  k = {optimal_k['adj_r2']}")
        print(f"  BIC (minimum):            k = {optimal_k['bic']}")
        print(f"  CV-RMSE (1-SE rule):      k = {optimal_k['cv_1se']}")

        # Print the evaluation table
        edf = pd.DataFrame(eval_results)
        edf['Added'] = ordered_display[:len(edf)]
        edf['Delta_Adj_R2'] = edf['Adj_R2'].diff().fillna(edf['Adj_R2'].iloc[0])

        print(f"\n  {'k':>3s}  {'Added':>10s}  {'R²':>8s}  {'Adj R²':>8s}  {'ΔAdj R²':>8s}  "
              f"{'BIC':>10s}  {'CV-RMSE':>10s}  {'±SD':>7s}  {'rₛ':>7s}")
        for _, row in edf.iterrows():
            markers = []
            if int(row['k']) == optimal_k['adj_r2']:
                markers.append('A')
            if int(row['k']) == optimal_k['bic']:
                markers.append('B')
            if int(row['k']) == optimal_k['cv_1se']:
                markers.append('C')
            marker_str = ' ← ' + ','.join(markers) if markers else ''

            print(f"  {int(row['k']):3d}  {row['Added']:>10s}  {row['R2']:8.4f}  {row['Adj_R2']:8.4f}  "
                  f"{row['Delta_Adj_R2']:+8.4f}  {row['BIC']:10.2f}  {row['CV_RMSE_mean']:10.4f}  "
                  f"±{row['CV_RMSE_sd']:6.4f}  {row['Spearman_rs']:7.4f}{marker_str}")

        # Decide on final k: use BIC as primary (most conservative and principled)
        # but report all three
        final_k = optimal_k['bic']
        final_cols = order[:final_k]
        final_params = ordered_params[:final_k]

        print(f"\n  → Selected k = {final_k} (BIC criterion)")
        print(f"    Parameters: {', '.join([PARAM_DISPLAY.get(p,p) for p in final_params])}")

        # Fit and report final model
        model = fit_final_model(X_raw, y_raw, final_cols, ALL_PARAMS, target_name)
        print_model_results(model)

        # Comparison with published
        ref_name, ref_rs = published_refs.get(target_name, ('', None))
        if ref_rs is not None:
            delta = model['r_spearman'] - ref_rs
            print(f"\n  Comparison: rₛ = {model['r_spearman']:.4f} vs {ref_name} rₛ = {ref_rs:.2f} (Δ = {delta:+.4f})")

        # Plot
        plot_model_selection(eval_results, ordered_params, target_name, optimal_k, output_dir)
        print(f"  Saved: model_selection_{target_name.replace(' ','_').replace('-','')}.pdf/png")

        # Save evaluation table
        edf_out = edf[['k', 'Added', 'R2', 'Adj_R2', 'Delta_Adj_R2', 'BIC',
                        'CV_RMSE_mean', 'CV_RMSE_sd', 'Spearman_rs']].copy()
        safe_name = target_name.replace(' ', '_').replace('-', '')
        edf_out.to_csv(output_dir / f'eval_{safe_name}.csv', index=False)

        all_summaries.append({
            'Target': target_name,
            'N': n,
            'k_adj_r2': optimal_k['adj_r2'],
            'k_bic': optimal_k['bic'],
            'k_cv_1se': optimal_k['cv_1se'],
            'k_final': final_k,
            'Parameters': ', '.join([PARAM_DISPLAY.get(p, p) for p in final_params]),
            'R2': model['r2'],
            'Adj_R2': model['adj_r2'],
            'r_Spearman': model['r_spearman'],
            'Ref_index': ref_name,
            'Ref_rs': ref_rs,
        })

    # ============================================================
    # Summary table
    # ============================================================

    print("\n\n" + "="*90)
    print("SUMMARY: OPTIMAL MINI-INDICES")
    print("="*90)

    summary_df = pd.DataFrame(all_summaries)

    print(f"\n{'Target':25s} {'k':>3s} {'Parameters':40s} {'rₛ':>7s} {'Adj R²':>8s} {'vs Published':>20s}")
    print("─" * 110)
    for _, row in summary_df.iterrows():
        ref_str = ''
        if row['Ref_rs'] is not None:
            delta = row['r_Spearman'] - row['Ref_rs']
            ref_str = f"Δ={delta:+.3f} vs {row['Ref_rs']:.2f}"
        print(f"{row['Target']:25s} {row['k_final']:3d} {row['Parameters']:40s} "
              f"{row['r_Spearman']:7.4f} {row['Adj_R2']:8.4f} {ref_str:>20s}")

    summary_df.to_csv(output_dir / 'summary_optimal_indices.csv', index=False)
    print(f"\nSaved: {output_dir / 'summary_optimal_indices.csv'}")

    print("\n" + "="*90)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*90)
    # print("\nNonlinear parameters positions in OMP:")
    # for p in ['rpde', 'dfa']:
    #     if p in ordered_params:
    #         print(f"  {p}: position {ordered_params.index(p)+1}")

if __name__ == '__main__':
    main()