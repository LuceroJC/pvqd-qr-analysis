#!/usr/bin/env python3
"""
ROC Analysis for Data-Driven Voice Quality Indices

Computes diagnostic accuracy (AUC, sensitivity, specificity) for the BIC-optimal
indices derived from the QR/OMP analysis with log-transformed parameters.

Jitter, shimmer (%), shimmer (dB), and PSD were log-transformed prior to analysis
to linearize their right-skewed distributions, consistent with the approximately
logarithmic relationship between acoustic perturbation and perceived voice quality.

Final index definitions (BIC-optimal from optimal_n_params.py, log-transformed):
  Severity:    6 params — log(ShimmerDB), GNE, Hno-6000, log(PSD), F0, HNR-D
  Breathiness: 3 params — CPPS, GNE, Hno-6000
  Roughness:   5 params — log(ShimmerDB), H1-H2, GNE, log(PSD), log(Jitter)
  Strain:      6 params — HNR, F0, log(PSD), Tilt, H1-H2, GNE

Methodology matches the companion JoV validation paper (Lucero, submitted):
  - Same PVQD database and perceptual rating thresholds
  - Same AUC computation (exact nonparametric, scikit-learn)
  - Same bootstrap 95% CIs (2000 resamples)
  - Same Youden's J index for cutoff selection
  - Same 10-fold stratified cross-validation (repeated 100 times)

Usage:
  python roc_analysis.py

Author: Jorge C. Lucero
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

PARAMS_CSV = '/home/jorge/pvqd_analysis/pvqd_vowel_params.csv'
CAPE_V_CSV = '/home/jorge/PVQD/CAPE_V.CSV'
GRBAS_CSV  = '/home/jorge/PVQD/GRBAS.CSV'
OUTPUT_DIR = '/home/jorge/pvqd_analysis/results/roc'

N_BOOTSTRAP = 2000
N_CV_REPEATS = 100
N_CV_FOLDS = 10
SEED = 42

# Parameters to log-transform (strictly positive, right-skewed)
LOG_PARAMS = ['jitter_local_pct', 'shimmer_local_pct', 'shimmer_local_dB', 'psd']

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

# BIC-optimal indices from optimal_n_params.py (log-transformed analysis)
MINI_INDICES = {
    'Severity Index': {
        'predictors': ['log_shimmer_local_dB', 'gne', 'hno_6000', 'log_psd',
                        'f0_mean', 'hnr_d'],
        'targets': [
            ('CAPE-V Severity', 10.0, '>='),
            ('GRBAS Grade', 0.5, '>='),
        ],
    },
    'Breathiness Index': {
        'predictors': ['cpps', 'gne', 'hno_6000'],
        'targets': [
            ('CAPE-V Breathiness', 10.0, '>='),
            ('GRBAS Breathiness', 0.5, '>='),
        ],
    },
    'Roughness Index': {
        'predictors': ['log_shimmer_local_dB', 'h1_h2', 'gne', 'log_psd',
                        'log_jitter_local_pct'],
        'targets': [
            ('CAPE-V Roughness', 10.0, '>='),
            ('GRBAS Roughness', 0.5, '>='),
        ],
    },
    'Strain Index': {
        'predictors': ['hnr', 'f0_mean', 'log_psd', 'tilt', 'h1_h2', 'gne'],
        'targets': [
            ('CAPE-V Strain', 10.0, '>='),
        ],
    },
}

# Published reference AUCs from JoV validation paper (for comparison)
PUBLISHED_AUCS = {
    ('CAPE-V Severity', 'AVQI'): 0.825,
    ('CAPE-V Breathiness', 'ABI'): 0.862,
    ('GRBAS Grade', 'AVQI'): 0.838,
    ('GRBAS Breathiness', 'ABI'): 0.850,
}


# ============================================================
# Data loading
# ============================================================

def extract_speaker_id(filename):
    name = str(filename).replace('.wav', '').replace('_vowel_a', '')
    for suffix in [' ENSS', '_ENSS', 'ENSS', '_E_NSS', ' E_NSS',
                   '_eNSS', ' eNSS', '_enss', ' enss']:
        name = name.replace(suffix, '')
    return name.rstrip('.').strip()


def load_and_merge():
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
    cape_v_wide = cape_v.pivot(index='File', columns='Characteristics',
                                values='Average all ratings all times').reset_index()
    cape_v_wide.columns.name = None
    cape_v_wide.rename(columns={'File': 'speaker_id'}, inplace=True)

    grbas = pd.read_csv(GRBAS_CSV, sep=';', decimal=',')
    grbas.columns = grbas.columns.str.strip()
    grbas['File'] = grbas['File'].str.strip()
    grbas_wide = grbas.pivot(index='File', columns='Characteristics',
                              values='Average all ratings all times').reset_index()
    grbas_wide.columns.name = None
    grbas_wide.rename(columns={'File': 'speaker_id'}, inplace=True)

    ratings = cape_v_wide.merge(grbas_wide, on='speaker_id', how='outer')
    df = acoustic.merge(ratings, on='speaker_id', how='inner')
    print(f"Merged records: {len(df)}")
    return df


# ============================================================
# Mini-index score computation
# ============================================================

def compute_mini_index_score(X, y):
    """Fit OLS on (X, y) and return predicted scores and coefficients."""
    n = len(y)
    X_int = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    return X_int @ beta, beta


# ============================================================
# ROC utilities
# ============================================================

def youden_optimal_cutoff(y_true, y_scores):
    """Find cutoff maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]


def bootstrap_auc_ci(y_true, y_scores, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute bootstrap 95% CI for AUC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def cross_validated_roc(X, y_continuous, threshold, predictors_count,
                        n_folds=N_CV_FOLDS, n_repeats=N_CV_REPEATS, seed=SEED):
    """
    10-fold stratified CV repeated n_repeats times.
    Returns mean AUC, sensitivity, specificity across all folds/repeats.
    """
    y_binary = (y_continuous >= threshold).astype(int)
    rng = np.random.RandomState(seed)

    all_aucs = []
    all_sens = []
    all_spec = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=rng.randint(0, 1_000_000))

        for train_idx, test_idx in skf.split(X, y_binary):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr_cont, y_te_cont = y_continuous[train_idx], y_continuous[test_idx]
            y_tr_bin, y_te_bin = y_binary[train_idx], y_binary[test_idx]

            n_tr = len(X_tr)
            X_tr_int = np.column_stack([np.ones(n_tr), X_tr])
            beta, _, _, _ = np.linalg.lstsq(X_tr_int, y_tr_cont, rcond=None)

            scores_tr = X_tr_int @ beta
            n_te = len(X_te)
            X_te_int = np.column_stack([np.ones(n_te), X_te])
            scores_te = X_te_int @ beta

            if len(np.unique(y_tr_bin)) < 2 or len(np.unique(y_te_bin)) < 2:
                continue

            cutoff_tr, _, _ = youden_optimal_cutoff(y_tr_bin, scores_tr)

            auc_te = roc_auc_score(y_te_bin, scores_te)
            pred_te = (scores_te >= cutoff_tr).astype(int)
            tp = np.sum((pred_te == 1) & (y_te_bin == 1))
            fn = np.sum((pred_te == 0) & (y_te_bin == 1))
            tn = np.sum((pred_te == 0) & (y_te_bin == 0))
            fp = np.sum((pred_te == 1) & (y_te_bin == 0))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            all_aucs.append(auc_te)
            all_sens.append(sens)
            all_spec.append(spec)

    return {
        'cv_auc': np.mean(all_aucs),
        'cv_auc_sd': np.std(all_aucs),
        'cv_sens': np.mean(all_sens),
        'cv_spec': np.mean(all_spec),
    }


def delong_test(y_true, scores1, scores2):
    """DeLong test for comparing two correlated AUCs."""
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    scores1_pos = scores1[y_true == 1]
    scores1_neg = scores1[y_true == 0]
    scores2_pos = scores2[y_true == 1]
    scores2_neg = scores2[y_true == 0]

    def placement_values(pos, neg):
        v10 = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
        v01 = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
        return v10, v01

    v10_1, v01_1 = placement_values(scores1_pos, scores1_neg)
    v10_2, v01_2 = placement_values(scores2_pos, scores2_neg)

    auc1 = np.mean(v10_1)
    auc2 = np.mean(v10_2)

    d10 = np.column_stack([v10_1 - auc1, v10_2 - auc2])
    s10 = d10.T @ d10 / (n1 - 1)

    d01 = np.column_stack([v01_1 - auc1, v01_2 - auc2])
    s01 = d01.T @ d01 / (n0 - 1)

    S = s10 / n1 + s01 / n0

    L = np.array([1, -1])
    var_diff = L @ S @ L
    if var_diff <= 0:
        return auc1, auc2, 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * stats.norm.sf(abs(z))

    return auc1, auc2, z, p


# ============================================================
# Plotting
# ============================================================

def plot_roc_curves(roc_data, output_dir):
    """Plot ROC curves for all mini-indices, grouped by dimension."""
    indices = {}
    for entry in roc_data:
        idx_name = entry['index_name']
        if idx_name not in indices:
            indices[idx_name] = []
        indices[idx_name].append(entry)

    n_indices = len(indices)
    fig, axes = plt.subplots(1, n_indices, figsize=(5 * n_indices, 4.5))
    if n_indices == 1:
        axes = [axes]

    for ax, (idx_name, entries) in zip(axes, indices.items()):
        for entry in entries:
            fpr, tpr, _ = roc_curve(entry['y_binary'], entry['scores'])
            auc = entry['resub_auc']
            ci_lo, ci_hi = entry['auc_ci']
            label = f"{entry['target']} (AUC={auc:.3f}, 95%CI [{ci_lo:.3f}\u2013{ci_hi:.3f}])"
            ax.plot(fpr, tpr, linewidth=1.5, label=label)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('1 \u2212 Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(idx_name)
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'roc_curves_mini_indices.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / 'roc_curves_mini_indices.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("  Saved: roc_curves_mini_indices.pdf/png")


# ============================================================
# Main
# ============================================================

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    print("="*90)
    print("ROC ANALYSIS FOR DATA-DRIVEN MINI-INDICES (LOG-TRANSFORMED)")
    print("="*90)

    df = load_and_merge()

    all_roc_data = []
    all_results = []
    delong_results = []

    for idx_name, idx_config in MINI_INDICES.items():
        predictors = idx_config['predictors']
        pred_display = [PARAM_DISPLAY.get(p, p) for p in predictors]

        print(f"\n{'='*90}")
        print(f"{idx_name} ({len(predictors)} params: {', '.join(pred_display)})")
        print(f"{'='*90}")

        for target_name, threshold, direction in idx_config['targets']:
            print(f"\n  Target: {target_name} (threshold: {direction} {threshold})")

            cols = predictors + [target_name]
            data = df[cols].dropna()
            n = len(data)

            X = data[predictors].values
            y_cont = data[target_name].values

            y_binary = (y_cont >= threshold).astype(int)
            n_pos = np.sum(y_binary == 1)
            n_neg = np.sum(y_binary == 0)
            print(f"  N = {n} ({n_pos} positive, {n_neg} negative)")

            if n_pos < 10 or n_neg < 10:
                print(f"  WARNING: Too few cases in one group, skipping.")
                continue

            # Resubstitution
            scores, beta = compute_mini_index_score(X, y_cont)
            resub_auc = roc_auc_score(y_binary, scores)
            cutoff, sens, spec = youden_optimal_cutoff(y_binary, scores)
            ci_lo, ci_hi = bootstrap_auc_ci(y_binary, scores)

            print(f"  Resubstitution:")
            print(f"    AUC = {resub_auc:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
            print(f"    Cutoff = {cutoff:.4f} (Youden's J)")
            print(f"    Sensitivity = {sens:.1%}")
            print(f"    Specificity = {spec:.1%}")

            # Regression coefficients
            coef_names = ['Intercept'] + pred_display
            print(f"  Regression coefficients:")
            for name, b in zip(coef_names, beta):
                print(f"    {name:>15s}: {b:12.6f}")

            # Cross-validation
            cv_results = cross_validated_roc(X, y_cont, threshold, len(predictors))
            print(f"  Cross-validation (10-fold x 100 repeats):")
            print(f"    AUC = {cv_results['cv_auc']:.3f} (SD = {cv_results['cv_auc_sd']:.3f})")
            print(f"    Sensitivity = {cv_results['cv_sens']:.1%}")
            print(f"    Specificity = {cv_results['cv_spec']:.1%}")
            print(f"    AUC optimism = {resub_auc - cv_results['cv_auc']:.4f}")

            # Store for ROC plot
            all_roc_data.append({
                'index_name': idx_name,
                'target': target_name,
                'scores': scores,
                'y_binary': y_binary,
                'resub_auc': resub_auc,
                'auc_ci': (ci_lo, ci_hi),
            })

            # Comparison with published
            ref_key_avqi = (target_name, 'AVQI')
            ref_key_abi = (target_name, 'ABI')
            ref_key = ref_key_avqi if ref_key_avqi in PUBLISHED_AUCS else ref_key_abi

            if ref_key in PUBLISHED_AUCS:
                ref_auc = PUBLISHED_AUCS[ref_key]
                ref_name = ref_key[1]
                delta_auc = resub_auc - ref_auc
                print(f"  Comparison with {ref_name}:")
                print(f"    Mini-index AUC = {resub_auc:.3f} vs {ref_name} AUC = {ref_auc:.3f} "
                      f"(Delta = {delta_auc:+.3f})")

                delong_results.append({
                    'Index': idx_name,
                    'Target': target_name,
                    'Mini_AUC': resub_auc,
                    'Ref_Index': ref_name,
                    'Ref_AUC': ref_auc,
                    'Delta_AUC': delta_auc,
                })

            all_results.append({
                'Index': idx_name,
                'Target': target_name,
                'Threshold': f'{direction} {threshold}',
                'N': n,
                'N_pos': n_pos,
                'N_neg': n_neg,
                'k': len(predictors),
                'Predictors': ', '.join(pred_display),
                'Resub_AUC': resub_auc,
                'AUC_CI_lo': ci_lo,
                'AUC_CI_hi': ci_hi,
                'Cutoff': cutoff,
                'Resub_Sens': sens,
                'Resub_Spec': spec,
                'CV_AUC': cv_results['cv_auc'],
                'CV_AUC_SD': cv_results['cv_auc_sd'],
                'CV_Sens': cv_results['cv_sens'],
                'CV_Spec': cv_results['cv_spec'],
                'AUC_Optimism': resub_auc - cv_results['cv_auc'],
            })

    # Plot ROC curves
    print(f"\n{'='*90}")
    print("GENERATING FIGURES")
    print(f"{'='*90}")
    plot_roc_curves(all_roc_data, output_dir)

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")

    results_df = pd.DataFrame(all_results)

    print(f"\n{'Index':22s} {'Target':22s} {'k':>3s} {'AUC':>7s} {'95% CI':>17s} "
          f"{'Sens':>7s} {'Spec':>7s} {'CV-AUC':>7s} {'CV-Sens':>7s} {'CV-Spec':>7s}")
    print("\u2500" * 120)
    for _, row in results_df.iterrows():
        ci_str = f"[{row['AUC_CI_lo']:.3f}\u2013{row['AUC_CI_hi']:.3f}]"
        print(f"{row['Index']:22s} {row['Target']:22s} {row['k']:3d} "
              f"{row['Resub_AUC']:7.3f} {ci_str:>17s} "
              f"{row['Resub_Sens']:7.1%} {row['Resub_Spec']:7.1%} "
              f"{row['CV_AUC']:7.3f} {row['CV_Sens']:7.1%} {row['CV_Spec']:7.1%}")

    results_df.to_csv(output_dir / 'roc_results.csv', index=False)
    print(f"\n  Saved: roc_results.csv")

    # Comparison with published
    if delong_results:
        print(f"\n{'='*90}")
        print("COMPARISON WITH PUBLISHED INDICES")
        print(f"{'='*90}")

        delong_df = pd.DataFrame(delong_results)
        print(f"\n{'Index':22s} {'Target':22s} {'Mini AUC':>9s} {'Ref':>6s} {'Ref AUC':>9s} {'Delta AUC':>10s}")
        print("\u2500" * 80)
        for _, row in delong_df.iterrows():
            print(f"{row['Index']:22s} {row['Target']:22s} {row['Mini_AUC']:9.3f} "
                  f"{row['Ref_Index']:>6s} {row['Ref_AUC']:9.3f} {row['Delta_AUC']:+10.3f}")

        delong_df.to_csv(output_dir / 'comparison_published.csv', index=False)
        print(f"\n  Saved: comparison_published.csv")

    # LaTeX table
    latex_path = output_dir / 'roc_table.tex'
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\caption{Diagnostic accuracy of dimension-specific indices.}\n")
        f.write("\\label{tab:roc_mini}\n")
        f.write("\\centering\n")
        f.write("\\makebox[\\textwidth]{\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llccccccc}\n")
        f.write("\\toprule\n")
        f.write(" & & \\multicolumn{4}{c}{Resubstitution} & "
                "\\multicolumn{3}{c}{Cross-Validation} \\\\\n")
        f.write("\\cmidrule(lr){3-6} \\cmidrule(lr){7-9}\n")
        f.write("Index & Reference Standard & AUC & Cutoff & Sens & Spec & AUC & Sens & Spec \\\\\n")
        f.write("\\midrule\n")

        for _, row in results_df.iterrows():
            idx_short = row['Index'].replace(' Index', '')
            target_short = row['Target']
            threshold_val = row['Threshold'].split()[-1]

            f.write(f"{idx_short} & {target_short} $\\geq$ {threshold_val} & "
                    f"{row['Resub_AUC']:.3f} & {row['Cutoff']:.2f} & "
                    f"{row['Resub_Sens']*100:.1f}\\% & {row['Resub_Spec']*100:.1f}\\% & "
                    f"{row['CV_AUC']:.3f} & {row['CV_Sens']*100:.1f}\\% & "
                    f"{row['CV_Spec']*100:.1f}\\% \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\multicolumn{9}{p{1.23\\textwidth}}{\n")
        f.write("\\footnotesize\n")
        f.write("Resubstitution: cutoff derived and evaluated on the full sample using "
                "Youden's J index. Cross-validation: 10-fold stratified, repeated 100 times. "
                "$N = 284$--288 depending on complete cases. Cutoff values are from "
                "resubstitution and should be considered hypothesis-generating; external "
                "validation is required before clinical use.}\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table}\n")

    print(f"  Saved: roc_table.tex")

    print(f"\n{'='*90}")
    print("ROC ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
