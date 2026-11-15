# XAI-IDS — paper tables and figures from a revised-pipeline output folder.
#
#   py -3.12 revision/analyze_results.py results/revised [results/revised/paper]
#
# Reads only files written by revision/xai_ids_revised.py and writes the CSV
# tables and PNG figures the paper uses, so every reported number can be
# regenerated from the committed outputs.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'paper'
OUT.mkdir(parents=True, exist_ok=True)
T = ROOT / 'tables'
SUMMARY = json.loads((ROOT / 'summary.json').read_text(encoding='utf-8'))
DATASETS = list(SUMMARY['datasets'])
NICE = {'CICIDS2017': 'CIC-IDS-2017', 'UNSWNB15': 'UNSW-NB15', 'CICIDS2018': 'CSE-CIC-IDS-2018'}
SCORES = {'confidence': 'Confidence', 'margin': 'Margin', '1-instability': '1 − instability',
          'jaccard': 'SHAP–LIME Jaccard', 'xcs_fixed': 'XCS (a priori)', 'xcs_learned': 'XCS (learned)'}
COL = {'confidence': '#1f2933', 'xcs_fixed': '#c53030', 'xcs_learned': '#2b6cb0'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def clean(label):
    """Kaggle's CIC-IDS-2017 labels contain a replacement character for the en dash."""
    return str(label).replace('�', '–').replace('Web Attack – Sql Injection', 'Web Attack – SQL Injection')


def family(name):
    """Coarse feature family. TCP flag counters (e.g. 'PSH Flag Count') are flags, and
    count features (ct_*) are checked before 'port' so that UNSW-NB15's
    ct_src_dport_ltm is a connection count, not a port identity."""
    n = name.lower()
    if 'flag' in n or 'win' in n: return 'tcp flags/window'
    if n.startswith('ct_') or any(t in n for t in ('cnt', 'count', 'trans_depth', 'loss')):
        return 'count'
    if 'port' in n: return 'port'
    if any(t in n for t in ('iat', 'duration', 'dur', 'active', 'idle', 'jit', 'rtt', 'synack', 'ackdat', 'inpkt')):
        return 'timing'
    if any(t in n for t in ('/s', 'rate', 'load')): return 'rate'
    if any(t in n for t in ('length', 'len', 'size', 'bytes', 'byts', 'smean', 'dmean')): return 'size/volume'
    if any(t in n for t in ('packets', 'pkts', 'pkt')): return 'count'
    return 'other'


# ── Table: datasets ──────────────────────────────────────────────────────────
rows = []
for d in DATASETS:
    s = SUMMARY['datasets'][d]
    dist = pd.read_csv(T / f'class_distribution_{d}.csv')
    rare = dist.sort_values('count').iloc[0]
    rows.append({'dataset': NICE[d], 'rows_loaded': s['cleaning']['raw_rows'], 'rows_after_dedup': s['cleaning']['after_dedup'],
                 'label_conflicts_removed': s['cleaning']['label_conflict_rows'], 'rows_used': s['cleaning']['final_rows'],
                 'classes': len(s['classes']), 'train': s['split']['train'], 'val': s['split']['val'], 'test': s['split']['test'],
                 'rarest_class': f"{clean(rare['class'])} ({int(rare['count'])})",
                 'imbalance_mode': s.get('imbalance_mode', 'n/a')})
pd.DataFrame(rows).to_csv(OUT / 'table_datasets.csv', index=False)

# ── Table: imbalance selection (validation) ─────────────────────────────────
if (T / 'imbalance_selection.csv').exists():
    imb = pd.read_csv(T / 'imbalance_selection.csv')
    imb['dataset'] = imb['dataset'].map(NICE)
    imb.pivot(index='mode', columns='dataset', values='val_macro_f1').reindex(['none', 'smote', 'weights', 'both']) \
       .to_csv(OUT / 'table_imbalance_selection.csv')

# ── Table: detection performance ─────────────────────────────────────────────
mc = pd.read_csv(T / 'model_comparison.csv')
mc['dataset'] = mc['dataset'].map(NICE)
mc[['dataset', 'model', 'n_test', 'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_macro_lo', 'f1_macro_hi',
    'f1_weighted', 'ece', 'train_seconds']].to_csv(OUT / 'table_detection.csv', index=False)

# ── Tables: per class (XGBoost), McNemar ─────────────────────────────────────
for d in DATASETS:
    pc = pd.read_csv(T / f'per_class_{d}_XGBoost.csv')
    pc['class'] = pc['class'].map(clean)
    pc.to_csv(OUT / f'table_per_class_{d}.csv', index=False)
mn = pd.read_csv(T / 'mcnemar.csv')
mn['dataset'] = mn['dataset'].map(NICE)
mn.to_csv(OUT / 'table_mcnemar.csv', index=False)

# ── Table + figure: XCS vs confidence ────────────────────────────────────────
xe = pd.read_csv(T / 'xcs_evaluation.csv')
xe['dataset_nice'] = xe['dataset'].map(NICE)
xe.to_csv(OUT / 'table_xcs.csv', index=False)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(10, 3.1), sharey=True)
order = ['margin', '1-instability', 'jaccard', 'xcs_fixed', 'xcs_learned']
for ax, d in zip(np.atleast_1d(axes), DATASETS):
    sub = xe[xe['dataset'] == d].set_index('score')
    y = np.arange(len(order))[::-1]
    diff = sub.loc[order, 'auroc_minus_conf'].values
    lo = sub.loc[order, 'diff_lo'].values; hi = sub.loc[order, 'diff_hi'].values
    ax.axvline(0, color='#9aa5b1', lw=1)
    ax.errorbar(diff, y, xerr=[diff - lo, hi - diff], fmt='o', color='#1f2933', ms=4, capsize=2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([SCORES[s] for s in order])
    ax.set_title(f"{NICE[d]}  (confidence AUROC {sub.loc['confidence', 'auroc']:.3f})", fontsize=9)
    ax.set_xlabel('Δ AUROC vs. confidence (95% CI)')
fig.tight_layout(); fig.savefig(OUT / 'fig_xcs_delta_auroc.png', dpi=300); plt.close(fig)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(10, 3.0), sharey=False)
for ax, d in zip(np.atleast_1d(axes), DATASETS):
    te = pd.read_csv(ROOT / 'xcs' / f'xcs_components_{d}_test.csv')
    nat = te[te['natural'] == True]
    correct = nat['correct'].astype(bool).values
    for s, col in COL.items():
        if s == 'xcs_fixed': v = nat['xcs'].values
        elif s == 'confidence': v = nat['confidence'].values
        elif 'xcs_learned' in nat: v = nat['xcs_learned'].values
        else: continue
        o = np.argsort(-v)
        risk = np.cumsum(~correct[o]) / np.arange(1, len(o) + 1)
        ax.plot(np.arange(1, len(o) + 1) / len(o), risk, color=col, lw=1.4, label=SCORES[s])
    ax.set_title(NICE[d], fontsize=9); ax.set_xlabel('Coverage (fraction of alerts accepted)')
    ax.set_ylabel('Error rate among accepted'); ax.grid(alpha=0.25)
np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'fig_risk_coverage.png', dpi=300); plt.close(fig)

# Thresholds and learned coefficients
thr_rows = []
for d in DATASETS:
    x = SUMMARY['xcs'][d]
    for label, t in x['thresholds'].items():
        thr_rows.append({'dataset': NICE[d], 'rule': label, **t, 'natural_error_rate': x['natural_error_rate']})
    coef = x.get('learned_lr_coefficients_standardised') or {}
    thr_rows[-1].update({f'coef_{k}': v for k, v in coef.items()})
pd.DataFrame(thr_rows).to_csv(OUT / 'table_xcs_thresholds.csv', index=False)

# Component behaviour by correctness and by predicted class
comp_rows = []
for d in DATASETS:
    te = pd.read_csv(ROOT / 'xcs' / f'xcs_components_{d}_test.csv')
    for ok, g in te.groupby('correct'):
        comp_rows.append({'dataset': NICE[d], 'correct': bool(ok), 'n': len(g),
                          'confidence_mean': g['confidence'].mean(), 'instability_mean': g['instability'].mean(),
                          'jaccard_mean': g['jaccard'].mean(), 'xcs_mean': g['xcs'].mean()})
pd.DataFrame(comp_rows).to_csv(OUT / 'table_xcs_components_by_correctness.csv', index=False)

# ── Explanations: SHAP families (corrected mapping), agreement, overlap, latency ─
fam_rows, top_rows = [], []
for d in DATASETS:
    g = pd.read_csv(T / f'shap_global_{d}.csv')
    g['family'] = g['feature'].map(family)
    share = g.groupby('family')['mean_abs_shap'].sum() / g['mean_abs_shap'].sum()
    for f, v in share.sort_values(ascending=False).items():
        fam_rows.append({'dataset': NICE[d], 'family': f, 'importance_share': v})
    for r in g.head(5).itertuples():
        top_rows.append({'dataset': NICE[d], 'feature': r.feature, 'mean_abs_shap': r.mean_abs_shap, 'family': r.family})
pd.DataFrame(fam_rows).to_csv(OUT / 'table_shap_families.csv', index=False)
pd.DataFrame(top_rows).to_csv(OUT / 'table_shap_top5.csv', index=False)
for f in ('shap_lime_agreement.csv', 'cross_dataset_feature_overlap.csv', 'xcs_latency.csv'):
    df = pd.read_csv(T / f)
    if 'dataset' in df: df['dataset'] = df['dataset'].map(NICE)
    if 'class' in df: df['class'] = df['class'].map(clean)
    df.to_csv(OUT / f'table_{f}', index=False)

# ── Figure: row-normalised confusion matrices (XGBoost) ──────────────────────
for d in DATASETS:
    cm = pd.read_csv(T / f'confusion_{d}_XGBoost.csv', index_col=0)
    labels = [clean(c) for c in cm.index]
    m = cm.values / np.maximum(cm.values.sum(1, keepdims=True), 1)
    k = len(labels)
    fig, ax = plt.subplots(figsize=(0.42 * k + 2.2, 0.42 * k + 1.6))
    ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    for i in range(k):
        for j in range(k):
            if m[i, j] >= 0.01:
                ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center', fontsize=5.5,
                        color='white' if m[i, j] > 0.6 else '#1f2933')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{NICE[d]} — XGBoost, row-normalised (test n={int(cm.values.sum())})', fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f'fig_confusion_{d}.png', dpi=300); plt.close(fig)

print('wrote', len(list(OUT.glob('*'))), 'files to', OUT)
