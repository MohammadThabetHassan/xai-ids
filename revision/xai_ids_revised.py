# XAI-IDS — revised multi-dataset pipeline (branch: revision/fix-leakage-and-xcs)
#
# Re-implements xai_ids_multidataset.ipynb with the fixes listed in
# revision/CHANGES.md. Every number reported in the revised paper must come
# from the files this script writes to OUT (see revision/RERUN_PLAN.md).
#
# Run on Kaggle (GPU optional) with the three datasets attached, or locally:
#   XAI_IDS_SMOKE=1 py -3.12 revision/xai_ids_revised.py   (tiny synthetic smoke test,
#                                                           checks the code path only)

import os, gc, re, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
import xgboost as xgb
import lightgbm as lgb
import shap
import lime
import lime.lime_tabular
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120

SMOKE = os.environ.get('XAI_IDS_SMOKE') == '1'
OUT = Path(os.environ.get('XAI_IDS_OUT',
           'xai_ids_revised_smoke' if SMOKE else '/kaggle/working/xai_ids_revised'))
for sub in ['plots', 'models', 'tables', 'xcs']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)


# ── Configuration ─────────────────────────────────────────────────────────────
USE_CICIDS2017 = True
USE_UNSWNB15   = True
USE_CICIDS2018 = True

MAX_PER_DATASET = {'CICIDS2017': 200_000, 'UNSWNB15': 145_000, 'CICIDS2018': 200_000}
ROWS_PER_CSV    = {'CICIDS2017': 25_000, 'UNSWNB15': 50_000, 'CICIDS2018': 20_000}
MIN_CLASS_COUNT = 10        # classes with fewer cleaned rows are dropped (logged)

TOP_K_FEATURES = 20
TEST_SIZE      = 0.20       # held-out test, touched only for final reporting
VAL_SIZE       = 0.20       # fraction of the remaining 80% → early stopping + XCS calibration
RANDOM_STATE   = 42
XGB_EPOCHS     = 300
PATIENCE       = 20
XGB_BALANCED_WEIGHTS = True # paper describes balanced sample weights for XGBoost
N_BOOT         = 500        # bootstrap resamples for metric CIs

# XCS
XCS_W          = (0.4, 0.3, 0.3)   # a-priori weights (conf, 1-instability, jaccard)
XCS_THRESHOLD  = 0.3               # legacy fixed threshold, reported for continuity
XCS_TOPK       = 5
XCS_NEIGHBOURS = 5
XCS_NOISE_STD  = 0.05              # Gaussian noise in RobustScaler (IQR) units
LIME_SAMPLES   = 1000
XCS_N_RANDOM   = {'val': 800, 'test': 1500}   # natural-distribution samples
XCS_MAX_ERRORS = 400                          # extra misclassified samples per split

if SMOKE:
    MAX_PER_DATASET = {k: 6000 for k in MAX_PER_DATASET}
    XGB_EPOCHS, N_BOOT, LIME_SAMPLES = 60, 50, 200
    XCS_N_RANDOM, XCS_MAX_ERRORS = {'val': 60, 'test': 80}, 30

# Identifier / leakage-prone columns removed per file *before* concatenation.
# Ports are dropped in every dataset (shortcut features; CIC-2018 previously kept
# 'Src Port' and 'Dst Port' because only the 2017 spellings were listed).
ID_COLS = [
    'Flow ID', 'Source IP', 'Destination IP', 'Src IP', 'Dst IP',
    'Source Port', 'Destination Port', 'Src Port', 'Dst Port',
    'Timestamp', 'timestamp', 'srcip', 'dstip', 'sport', 'dsport', 'id',
    'stime', 'ltime', 'Stime', 'Ltime',
]
# Categorical / alternate-label columns (UNSW-NB15) — not numeric features.
NON_FEATURE_COLS = ['proto', 'service', 'state', 'label', 'label_val',
                    'attack', 'is_attack', 'target', 'attack_cat']
LABEL_CANDIDATES = ['Label', 'label', 'attack_cat', 'Attack', 'attack', 'type', 'label_val']


def find_path(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None

PATHS = {
    'CICIDS2017': find_path([os.environ.get('XAI_IDS_CICIDS2017'),
        '/kaggle/input/datasets/dhoogla/cicids2017', '/kaggle/input/cicids2017',
        '/kaggle/input/datasets/mohdqwe123/dataset1/cicids2017',
        '/kaggle/input/network-intrusion-dataset']),
    'UNSWNB15': find_path([os.environ.get('XAI_IDS_UNSWNB15'),
        '/kaggle/input/datasets/dhoogla/unswnb15', '/kaggle/input/unswnb15',
        '/kaggle/input/datasets/mohdqwe123/dataset1/unswnb15', '/kaggle/input/unsw-nb15']),
    # The full 10-day CSE-CIC-IDS-2018 release is listed first.
    'CICIDS2018': find_path([os.environ.get('XAI_IDS_CICIDS2018'),
        '/kaggle/input/datasets/solarmainframe/ids-intrusion-csv',
        '/kaggle/input/ids-intrusion-csv', '/kaggle/input/csecicids2018',
        '/kaggle/input/datasets/mohdqwe123/dataset1/csecicids2018']),
}
LABEL_PREFS = {'UNSWNB15': ['attack_cat', 'Label', 'label', 'label_val', 'Attack']}
USE = {'CICIDS2017': USE_CICIDS2017, 'UNSWNB15': USE_UNSWNB15, 'CICIDS2018': USE_CICIDS2018}

try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    device = 'cpu'
print('device:', device, '| OUT:', OUT)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        raise TypeError(type(o))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=conv)


# ── 1. Loading ────────────────────────────────────────────────────────────────
def _strip_ids(df, label_col):
    df.columns = df.columns.str.strip()
    drop = [c for c in ID_COLS if c in df.columns and c != label_col]
    return df.drop(columns=drop)


def load_parquet_files(directory, max_total, label_cands):
    files = sorted(directory.rglob('*.parquet'))
    print(' ', len(files), 'parquet files')
    n_per = max(1, max_total // len(files))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df.columns = df.columns.str.strip()
        label_col = next((c for c in label_cands if c in df.columns), None)
        df = _strip_ids(df, label_col)
        if len(df) > n_per:
            df = df.sample(n=n_per, random_state=RANDOM_STATE)
        dfs.append(df)
        print('   sampled', f.name, ':', len(df))
    return dfs


def load_csv_files_stratified(directory, rows_per_file, label_cands):
    """Per-file, per-label sampling. IDs are removed per file so that files with
    extra identifier columns (CIC-2018 Tuesday-20-02) no longer turn every other
    file's rows into NaN after concatenation — the cause of the 2-class result."""
    files = sorted(directory.rglob('*.csv'))
    print(' ', len(files), 'CSV files')
    dfs = []
    for f in files:
        enc = 'utf-8'
        try:
            head = pd.read_csv(f, nrows=5, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            enc = 'latin-1'
            head = pd.read_csv(f, nrows=5, low_memory=False, encoding=enc)
        head.columns = head.columns.str.strip()
        label_col = next((c for c in label_cands if c in head.columns), None)
        if label_col is None:
            print('   skip', f.name, '(no label)')
            continue
        buckets = {}
        for chunk in pd.read_csv(f, encoding=enc, low_memory=False, chunksize=100_000):
            chunk = _strip_ids(chunk, label_col)
            lab = chunk[label_col].astype(str).str.strip()
            chunk = chunk[lab != label_col]            # repeated header rows inside CIC-2018 files
            for lbl, grp in chunk.groupby(chunk[label_col].astype(str).str.strip()):
                buckets.setdefault(lbl, []).append(grp)
        per_cls = max(50, rows_per_file // max(1, len(buckets)))
        sampled = []
        for lbl, parts in buckets.items():
            full = pd.concat(parts, ignore_index=True)
            sampled.append(full.sample(n=min(per_cls, len(full)), random_state=RANDOM_STATE))
        if sampled:
            df = pd.concat(sampled, ignore_index=True)
            if label_col != 'Label':
                df = df.rename(columns={label_col: 'Label'})
            dfs.append(df)
            print('   sampled', f.name, ':', len(df), '| labels:', sorted(buckets))
    return dfs


def load_and_clean(name):
    directory = PATHS[name]
    cands = LABEL_PREFS.get(name, LABEL_CANDIDATES)
    if list(directory.rglob('*.parquet')):
        dfs = load_parquet_files(directory, MAX_PER_DATASET[name] * 2, cands)
    else:
        dfs = load_csv_files_stratified(directory, ROWS_PER_CSV[name], cands)
    if not dfs:
        return None
    # Align on the columns shared by all files (logged), then concatenate.
    label_col = next((c for c in cands if c in dfs[0].columns), 'Label')
    dfs = [d.rename(columns={label_col: 'Label'}) if label_col in d.columns else d for d in dfs]
    common = set.intersection(*[set(d.columns) for d in dfs])
    dropped = sorted(set.union(*[set(d.columns) for d in dfs]) - common)
    if dropped:
        print('  columns not shared by all files (dropped):', dropped)
    raw = pd.concat([d[sorted(common)] for d in dfs], ignore_index=True)
    return clean_dataset(raw, name)


def clean_dataset(raw, name):
    log = {'raw_rows': len(raw)}
    y_raw = raw['Label'].astype(str).str.strip()
    X = raw.drop(columns=['Label'] + [c for c in NON_FEATURE_COLS if c in raw.columns])
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.select_dtypes(include=[np.number]).astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).values
    X, y_raw = X[ok].reset_index(drop=True), y_raw[ok].reset_index(drop=True)
    log['after_nan_inf'] = len(X)

    both = X.assign(_y=y_raw.values)
    dup = both.duplicated()
    X, y_raw = X[~dup].reset_index(drop=True), y_raw[~dup].reset_index(drop=True)
    log['after_dedup'] = len(X)
    conflict = X.duplicated(keep=False)                 # same features, different labels
    log['label_conflict_rows'] = int(conflict.sum())
    X, y_raw = X[~conflict].reset_index(drop=True), y_raw[~conflict].reset_index(drop=True)

    zero_var = X.columns[X.std() == 0].tolist()
    X = X.drop(columns=zero_var)
    log['zero_var_dropped'] = zero_var

    counts = y_raw.value_counts()
    rare = counts[counts < MIN_CLASS_COUNT].index.tolist()
    if rare:
        keep = ~y_raw.isin(rare)
        X, y_raw = X[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)
    log['rare_classes_dropped'] = {str(k): int(counts[k]) for k in rare}

    cap = MAX_PER_DATASET[name]
    if len(X) > cap:
        idx, _ = train_test_split(np.arange(len(X)), train_size=cap,
                                  stratify=y_raw, random_state=RANDOM_STATE)
        X, y_raw = X.iloc[idx].reset_index(drop=True), y_raw.iloc[idx].reset_index(drop=True)
    log['final_rows'] = len(X)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    dist = pd.Series(y_raw).value_counts().rename_axis('class').reset_index(name='count')
    dist.to_csv(OUT / 'tables' / f'class_distribution_{name}.csv', index=False)
    print('  cleaning log:', {k: v for k, v in log.items() if k != 'zero_var_dropped'})
    print('  classes:', len(le.classes_), list(le.classes_))
    return {'X': X, 'y': y, 'le': le, 'log': log}


def make_smoke_dataset(name, seed):
    """Synthetic data with the real datasets' shape (imbalanced, multi-class) used
    ONLY to check that the code runs. Never report these numbers."""
    from sklearn.datasets import make_classification
    n_cls = {'CICIDS2017': 6, 'UNSWNB15': 5, 'CICIDS2018': 4}[name]
    w = np.array([0.6] + [0.4 / (n_cls - 1)] * (n_cls - 1))
    Xa, ya = make_classification(n_samples=MAX_PER_DATASET[name], n_features=40, n_informative=12,
                                 n_redundant=6, n_classes=n_cls, n_clusters_per_class=1,
                                 weights=w, flip_y=0.03, random_state=seed)
    cols = [f'f{i}' for i in range(40)]
    if name == 'CICIDS2018':
        cols[:4] = ['Fwd Pkt Len Max', 'Flow IAT Min', 'Tot Fwd Pkts', 'Pkt Size Avg']
    if name == 'CICIDS2017':
        cols[:4] = ['Fwd Packet Length Max', 'Flow IAT Min', 'Total Fwd Packets', 'Avg Packet Size']
    raw = pd.DataFrame(Xa, columns=cols)
    raw['Label'] = [f'class_{c}' for c in ya]
    return clean_dataset(raw, name)


DATASETS = {}
for name in ['CICIDS2017', 'UNSWNB15', 'CICIDS2018']:
    if not USE[name]:
        continue
    print('\nLoading', name)
    if SMOKE:
        ds = make_smoke_dataset(name, seed=len(DATASETS) + 1)
    elif PATHS[name] is None:
        print('  NOT FOUND — skipped'); continue
    else:
        ds = load_and_clean(name)
    if ds is not None and len(ds['le'].classes_) >= 2:
        DATASETS[name] = ds
if not DATASETS:
    raise RuntimeError('No datasets loaded — check PATHS')


# ── 2. Split first, then fit everything on the training partition only ──────
def select_features(X_tr, y_tr, k):
    """ANOVA top-30 → SHAP top-k, both fitted on the training partition only."""
    n_s = min(100_000, len(X_tr))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), n_s, replace=False)
    Xs, ys = X_tr.iloc[idx], y_tr[idx]
    sel = SelectKBest(f_classif, k=min(30, X_tr.shape[1])).fit(Xs, ys)
    cols = Xs.columns[sel.get_support()].tolist()
    n_cls = len(np.unique(ys))
    qm = xgb.XGBClassifier(n_estimators=80, max_depth=5, n_jobs=-1, verbosity=0,
                           random_state=RANDOM_STATE,
                           objective='binary:logistic' if n_cls == 2 else 'multi:softprob')
    qm.fit(Xs[cols], ys)
    sv = shap.TreeExplainer(qm).shap_values(Xs[cols].iloc[:1500])
    ms = np.abs(sv).mean(axis=(0, 2)) if np.ndim(sv) == 3 else np.abs(sv).mean(0)
    imp = pd.Series(ms, index=cols).sort_values(ascending=False)
    return imp.head(k).index.tolist(), imp


for name, ds in DATASETS.items():
    print('\nPreparing', name)
    X, y = ds['X'], ds['y']
    idx_trv, idx_te = train_test_split(np.arange(len(X)), test_size=TEST_SIZE,
                                       stratify=y, random_state=RANDOM_STATE)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=VAL_SIZE,
                                      stratify=y[idx_trv], random_state=RANDOM_STATE)
    feats, imp = select_features(X.iloc[idx_tr].reset_index(drop=True), y[idx_tr], TOP_K_FEATURES)
    imp.to_csv(OUT / 'tables' / f'feature_selection_importance_{name}.csv', header=['mean_abs_shap'])

    scaler = RobustScaler().fit(X.iloc[idx_tr][feats].values)
    X_tr = scaler.transform(X.iloc[idx_tr][feats].values)
    X_va = scaler.transform(X.iloc[idx_va][feats].values)
    X_te = scaler.transform(X.iloc[idx_te][feats].values)
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
    X_tr_nosmote = X_tr.copy()

    cc = Counter(y_tr)
    target = max(500, int(np.median(list(cc.values()))))
    strat = {c: max(v, min(target, 3000)) for c, v in cc.items() if v < target}
    smote_info = 'not needed'
    if strat:
        try:
            k_nn = max(1, min(5, min(cc[c] for c in strat) - 1))
            X_tr, y_tr = SMOTE(sampling_strategy=strat, k_neighbors=k_nn,
                               random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
            smote_info = {ds['le'].classes_[c]: [int(cc[c]), int(n)] for c, n in strat.items()}
        except Exception as e:
            smote_info = 'skipped: ' + str(e)
    ds.update(features=feats, scaler=scaler, class_names=list(ds['le'].classes_),
              X_train=X_tr, y_train=y_tr, X_train_nosmote=X_tr_nosmote,
              X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te,
              split_sizes={'train': len(idx_tr), 'train_after_smote': len(y_tr),
                           'val': len(idx_va), 'test': len(idx_te)},
              smote=smote_info)
    print('  split:', ds['split_sizes'], '| features:', feats)


# ── 3. Models (early stopping on validation, never on test) ──────────────────
ALL_RESULTS = {}
for name, ds in DATASETS.items():
    print('\nTraining on', name)
    X_tr, y_tr, X_va, y_va, X_te = ds['X_train'], ds['y_train'], ds['X_val'], ds['y_val'], ds['X_test']
    n_cls = len(ds['class_names'])
    objective = 'binary:logistic' if n_cls == 2 else 'multi:softprob'
    metric = 'logloss' if n_cls == 2 else 'mlogloss'
    sw = compute_sample_weight('balanced', y_tr) if XGB_BALANCED_WEIGHTS else None
    xgb_params = dict(max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3, tree_method='hist',
                      device=device, n_jobs=-1, random_state=RANDOM_STATE,
                      objective=objective, eval_metric=metric, verbosity=0)
    models, train_time = {}, {}

    t0 = time.time()
    xgb_m = xgb.XGBClassifier(n_estimators=XGB_EPOCHS, early_stopping_rounds=PATIENCE, **xgb_params)
    xgb_m.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_va, y_va)], verbose=False)
    models['XGBoost'], train_time['XGBoost'] = xgb_m, time.time() - t0

    t0 = time.time()
    rf_m = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                  class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    rf_m.fit(X_tr, y_tr)
    models['RandomForest'], train_time['RandomForest'] = rf_m, time.time() - t0

    t0 = time.time()
    lgb_params = dict(max_depth=8, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, class_weight='balanced', n_jobs=-1,
                      random_state=RANDOM_STATE, verbosity=-1)
    lgb_m = lgb.LGBMClassifier(n_estimators=300, **lgb_params)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    models['LightGBM'], train_time['LightGBM'] = lgb_m, time.time() - t0

    # Ensemble members are refit by VotingClassifier, so iteration counts are fixed
    # to the validation-selected values. Balancing is applied once, through the
    # shared sample weights, so RF and LightGBM members drop their class weights.
    t0 = time.time()
    xgb_fixed = xgb.XGBClassifier(n_estimators=(xgb_m.best_iteration or XGB_EPOCHS - 1) + 1, **xgb_params)
    rf_ens = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                    n_jobs=-1, random_state=RANDOM_STATE)
    lgb_fixed = lgb.LGBMClassifier(n_estimators=max(1, lgb_m.best_iteration_ or 300),
                                   **{**lgb_params, 'class_weight': None})
    ens = VotingClassifier([('xgb', xgb_fixed), ('rf', rf_ens), ('lgb', lgb_fixed)], voting='soft', n_jobs=1)
    ens.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
    models['Ensemble'], train_time['Ensemble'] = ens, time.time() - t0

    ALL_RESULTS[name] = {m: {'model': mdl, 'proba': mdl.predict_proba(X_te), 'train_s': train_time[m]}
                         for m, mdl in models.items()}
    for m, r in ALL_RESULTS[name].items():
        r['pred'] = r['proba'].argmax(1)
        print('  {:<13} acc={:.4f} macroF1={:.4f} ({:.0f}s)'.format(
            m, accuracy_score(ds['y_test'], r['pred']),
            f1_score(ds['y_test'], r['pred'], average='macro', zero_division=0), r['train_s']))
    ds['best_model'] = xgb_m
    try:
        xgb_m.set_params(device='cpu')     # SHAP / LIME / latency measured on CPU
    except Exception:
        pass


# ── 4. Evaluation: CIs, per-class, McNemar, calibration, PR ──────────────────
def boot_ci(y_true, y_pred, fn, n=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    vals = [fn(y_true[i], y_pred[i]) for i in (rng.randint(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece_score(proba, y_true, bins=15):
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y_true[m]).mean() - conf[m].mean())
    return e


def mcnemar(y, pa, pb):
    a_ok, b_ok = pa == y, pb == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0, 'none'
    if b + c < 25:
        return b, c, float('nan'), stats.binomtest(min(b, c), b + c, 0.5).pvalue, 'exact'
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)                # Edwards continuity correction
    return b, c, chi2, stats.chi2.sf(chi2, 1), 'chi2_cc'


def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj


macro = lambda a, b: f1_score(a, b, average='macro', zero_division=0)
weighted = lambda a, b: f1_score(a, b, average='weighted', zero_division=0)
rows, mcn_rows = [], []
for name, ds in DATASETS.items():
    y_te, cn = ds['y_test'], ds['class_names']
    res = ALL_RESULTS[name]
    for m, r in res.items():
        p = r['pred']
        wlo, whi = boot_ci(y_te, p, weighted)
        mlo, mhi = boot_ci(y_te, p, macro)
        rows.append({'dataset': name, 'model': m, 'n_test': len(y_te),
                     'accuracy': accuracy_score(y_te, p), 'balanced_accuracy': balanced_accuracy_score(y_te, p),
                     'f1_weighted': weighted(y_te, p), 'f1_weighted_lo': wlo, 'f1_weighted_hi': whi,
                     'f1_macro': macro(y_te, p), 'f1_macro_lo': mlo, 'f1_macro_hi': mhi,
                     'ece': ece_score(r['proba'], y_te), 'train_seconds': r['train_s']})
        pr, rc, f1c, sup = precision_recall_fscore_support(y_te, p, labels=range(len(cn)), zero_division=0)
        cm = confusion_matrix(y_te, p, labels=range(len(cn)))
        fp = cm.sum(0) - np.diag(cm)
        fpr = fp / (len(y_te) - cm.sum(1))
        ap = [average_precision_score(y_te == k, r['proba'][:, k]) if sup[k] else np.nan for k in range(len(cn))]
        pd.DataFrame({'class': cn, 'support': sup, 'precision': pr, 'recall': rc, 'f1': f1c,
                      'fpr': fpr, 'average_precision': ap}
                     ).to_csv(OUT / 'tables' / f'per_class_{name}_{m}.csv', index=False)
        pd.DataFrame(cm, index=cn, columns=cn).to_csv(OUT / 'tables' / f'confusion_{name}_{m}.csv')

    names = list(res)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    out = [mcnemar(y_te, res[a]['pred'], res[b]['pred']) for a, b in pairs]
    adj = holm(np.array([o[3] for o in out]))
    for (a, b), (bb, cc_, st, pv, kind), pa in zip(pairs, out, adj):
        mcn_rows.append({'dataset': name, 'model_a': a, 'model_b': b, 'a_right_b_wrong': bb,
                         'a_wrong_b_right': cc_, 'statistic': st, 'p': pv, 'p_holm': pa, 'test': kind})

    # Figures from real predictions: confusion (counts + row-normalised), PR, reliability.
    p = res['XGBoost']['pred']; proba = res['XGBoost']['proba']
    cm = confusion_matrix(y_te, p, labels=range(len(cn)))
    dim = max(7, 0.55 * len(cn) + 3)
    for norm in (False, True):
        fig, ax = plt.subplots(figsize=(dim, dim))
        mat = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm else cm
        ConfusionMatrixDisplay(mat, display_labels=cn).plot(
            ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False, values_format='.2f' if norm else 'd')
        ax.set_title(f'{name} — XGBoost ({"row-normalised" if norm else "counts"}, n={len(y_te)})')
        plt.tight_layout(); plt.savefig(OUT / 'plots' / f'confusion_{name}{"_norm" if norm else ""}.png', dpi=200)
        plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for k, c in enumerate(cn):
        if (y_te == k).sum() == 0: continue
        pr_, rc_, _ = precision_recall_curve(y_te == k, proba[:, k])
        ax.plot(rc_, pr_, lw=1, label=f'{c} (AP={average_precision_score(y_te == k, proba[:, k]):.2f}, n={(y_te == k).sum()})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title(f'{name} — XGBoost one-vs-rest PR')
    ax.legend(fontsize=6, loc='lower left'); plt.tight_layout()
    plt.savefig(OUT / 'plots' / f'pr_{name}.png', dpi=200); plt.close()

    conf, ok = proba.max(1), (p == y_te)
    edges = np.linspace(0, 1, 16); mids, accs, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            mids.append(conf[m].mean()); accs.append(ok[m].mean()); cnt.append(m.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray'); ax.plot(mids, accs, 'o-')
    ax.set_xlabel('Mean max-probability (bin)'); ax.set_ylabel('Accuracy (bin)')
    ax.set_title(f'{name} — XGBoost reliability (ECE={ece_score(proba, y_te):.4f})')
    plt.tight_layout(); plt.savefig(OUT / 'plots' / f'reliability_{name}.png', dpi=200); plt.close()

pd.DataFrame(rows).to_csv(OUT / 'tables' / 'model_comparison.csv', index=False)
pd.DataFrame(mcn_rows).to_csv(OUT / 'tables' / 'mcnemar.csv', index=False)
print('\n', pd.DataFrame(rows)[['dataset', 'model', 'accuracy', 'f1_weighted', 'f1_macro', 'ece']].round(4))


# ── 5. Global SHAP + cross-dataset feature families ──────────────────────────
def shap_for(sv, i, cls):
    """SHAP vector of sample i for class cls; handles binary (n, f) and multiclass (n, f, c)."""
    if isinstance(sv, list):
        return sv[cls][i]
    if sv.ndim == 3:
        return sv[i, :, cls]
    return sv[i] if cls == 1 else -sv[i]      # binary margin output: class 0 = negative


ABBREV = {'pkt': 'packet', 'pkts': 'packets', 'len': 'length', 'tot': 'total', 'totlen': 'total length',
          'byts': 'bytes', 'byt': 'bytes', 'cnt': 'count', 'seg': 'segment', 'avg': 'average',
          'fwd': 'forward', 'bwd': 'backward', 'blk': 'block', 'var': 'variance', 'win': 'window',
          'hdr': 'header', 'subflow': 'subflow', 'init': 'initial', 'pk': 'packet', 'mean': 'mean'}


def canon(name):
    n = name.lower().replace('/s', ' persec').replace('_', ' ')
    toks = [ABBREV.get(t, t) for t in re.split(r'\s+', n) if t]
    return ' '.join(sorted(' '.join(toks).split()))


def family(name):
    n = name.lower()
    if 'port' in n: return 'port'
    if 'flag' in n or 'win' in n: return 'tcp flags/window'
    if any(t in n for t in ('iat', 'duration', 'dur', 'active', 'idle', 'jit', 'rtt', 'synack', 'ackdat', 'inpkt')):
        return 'timing'
    if any(t in n for t in ('/s', 'rate', 'load')): return 'rate'
    if any(t in n for t in ('length', 'len', 'size', 'bytes', 'byts', 'smean', 'dmean')): return 'size/volume'
    if any(t in n for t in ('packets', 'pkts', 'pkt', 'count', 'cnt', 'ct_', 'trans_depth', 'loss')): return 'count'
    if 'ttl' in n: return 'ttl'
    return 'other'


SHAP_DATA = {}
for name, ds in DATASETS.items():
    model, feats = ds['best_model'], ds['features']
    n_s = min(2000, len(ds['X_test']))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(ds['X_test']), n_s, replace=False)
    Xs = ds['X_test'][idx]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    mean_abs = np.abs(sv).mean(axis=(0, 2)) if np.ndim(sv) == 3 else np.abs(sv).mean(0)
    imp = pd.Series(mean_abs, index=feats).sort_values(ascending=False)
    pd.DataFrame({'feature': imp.index, 'mean_abs_shap': imp.values,
                  'family': [family(f) for f in imp.index]}
                 ).to_csv(OUT / 'tables' / f'shap_global_{name}.csv', index=False)
    pred = model.predict(Xs)
    sv_pred = np.array([shap_for(sv, i, pred[i]) for i in range(n_s)])  # predicted-class attributions
    plt.figure()
    shap.summary_plot(sv_pred, Xs, feature_names=feats, max_display=12, show=False)
    plt.title(f'{name} — SHAP (predicted-class attributions, n={n_s})')
    plt.tight_layout(); plt.savefig(OUT / 'plots' / f'shap_beeswarm_{name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    SHAP_DATA[name] = {'explainer': explainer, 'imp': imp}

fam_rows, ovl_rows = [], []
for name, sd in SHAP_DATA.items():
    imp = sd['imp']
    share = imp.groupby([family(f) for f in imp.index]).sum() / imp.sum()
    for fam, v in share.items():
        fam_rows.append({'dataset': name, 'family': fam, 'importance_share': v})
dsn = list(SHAP_DATA)
for i in range(len(dsn)):
    for j in range(i + 1, len(dsn)):
        a = {canon(f) for f in SHAP_DATA[dsn[i]]['imp'].head(10).index}
        b = {canon(f) for f in SHAP_DATA[dsn[j]]['imp'].head(10).index}
        ovl_rows.append({'a': dsn[i], 'b': dsn[j], 'top10_jaccard_harmonised': len(a & b) / len(a | b),
                         'shared': sorted(a & b),
                         'note': 'UNSW-NB15 uses a different feature schema; name overlap is not meaningful'
                                 if 'UNSWNB15' in (dsn[i], dsn[j]) else ''})
pd.DataFrame(fam_rows).to_csv(OUT / 'tables' / 'shap_family_share.csv', index=False)
pd.DataFrame(ovl_rows).to_csv(OUT / 'tables' / 'cross_dataset_feature_overlap.csv', index=False)


# ── 6. XCS components (fixed) and evaluation ─────────────────────────────────
def xcs_components(model, explainer, lime_exp, X, rng):
    """Per-sample confidence, margin, SHAP instability and SHAP–LIME Jaccard.

    Instability = 1 − mean Spearman ρ between |SHAP(x)| and |SHAP(x+ε)| over
    neighbours (predicted class), in [0, 1] after clipping. The previous code
    took np.std over the flattened neighbour×feature array, i.e. the spread
    across features, which saturated the 0.5 cap for every sample.

    LIME features are read from as_map() (feature indices) instead of matching
    condition strings like 'Fwd IAT Std <= 14.59' against names, which never
    matched and produced Jaccard = 0 for every sample."""
    proba = model.predict_proba(X)
    pred = proba.argmax(1)
    srt = np.sort(proba, 1)
    sv = explainer.shap_values(X)
    rec = []
    for i in range(len(X)):
        c = int(pred[i])
        t0 = time.perf_counter()
        model.predict_proba(X[i:i + 1])
        t_conf = time.perf_counter() - t0

        t0 = time.perf_counter()
        base = np.abs(shap_for(sv, i, c))
        nb = X[i] + rng.normal(0, XCS_NOISE_STD, (XCS_NEIGHBOURS, X.shape[1]))
        svn = explainer.shap_values(nb)
        rhos, cvs = [], []
        nb_abs = np.array([np.abs(shap_for(svn, j, c)) for j in range(XCS_NEIGHBOURS)])
        for v in nb_abs:
            r = stats.spearmanr(base, v).correlation
            rhos.append(1.0 if np.isnan(r) else r)
        instab = float(np.clip(1 - np.mean(rhos), 0, 1))
        instab_cv = float(min(nb_abs.std(0).mean() / (base.mean() + 1e-9), 1.0))
        t_shap = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            e = lime_exp.explain_instance(X[i], model.predict_proba, num_features=XCS_TOPK,
                                          labels=(c,), num_samples=LIME_SAMPLES)
            lime_top = {f for f, _ in sorted(e.as_map()[c], key=lambda t: -abs(t[1]))[:XCS_TOPK]}
        except Exception:
            lime_top = set()
        t_lime = time.perf_counter() - t0
        shap_top = set(np.argsort(-base)[:XCS_TOPK])
        jac = len(shap_top & lime_top) / len(shap_top | lime_top) if lime_top else np.nan
        rec.append({'pred': c, 'confidence': proba[i, c], 'margin': srt[i, -1] - srt[i, -2],
                    'instability': instab, 'instability_cv': instab_cv, 'jaccard': jac,
                    't_conf_ms': 1e3 * t_conf, 't_shap_ms': 1e3 * t_shap, 't_lime_ms': 1e3 * t_lime})
    return pd.DataFrame(rec)


def xcs_sample(y_true, y_pred, n_random, rng):
    n = len(y_true)
    rand = rng.choice(n, min(n_random, n), replace=False)
    err = np.setdiff1d(np.where(y_pred != y_true)[0], rand)
    err = rng.choice(err, min(XCS_MAX_ERRORS, len(err)), replace=False) if len(err) else err
    return np.concatenate([rand, err]), np.r_[np.ones(len(rand), bool), np.zeros(len(err), bool)]


def auroc_ci(correct, score, rng, base=None):
    """AUROC of score for predicting correctness, bootstrap CI; paired diff vs base."""
    if correct.all() or (~correct).all():
        return dict(auroc=np.nan, lo=np.nan, hi=np.nan, diff=np.nan, diff_lo=np.nan, diff_hi=np.nan)
    a = roc_auc_score(correct, score); boots, diffs = [], []
    d = roc_auc_score(correct, score) - roc_auc_score(correct, base) if base is not None else np.nan
    for _ in range(N_BOOT):
        i = rng.randint(0, len(correct), len(correct))
        if correct[i].all() or (~correct[i]).all(): continue
        boots.append(roc_auc_score(correct[i], score[i]))
        if base is not None:
            diffs.append(boots[-1] - roc_auc_score(correct[i], base[i]))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
    (lo, hi), (dlo, dhi) = q(boots), q(diffs)
    return dict(auroc=a, lo=lo, hi=hi, diff=d, diff_lo=dlo, diff_hi=dhi)


def aurc(correct, score):
    """Area under the risk–coverage curve (lower is better), natural sample only."""
    order = np.argsort(-score)
    risk = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    return float(risk.mean())


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2: return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


xcs_eval_rows, agree_rows, timing_rows, xcs_summary = [], [], [], {}
for name, ds in DATASETS.items():
    print('\nXCS:', name)
    model, explainer = ds['best_model'], SHAP_DATA[name]['explainer']
    rng = np.random.RandomState(RANDOM_STATE)
    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=ds['X_train_nosmote'][:5000], feature_names=ds['features'],
        class_names=ds['class_names'], mode='classification',
        discretize_continuous=True, random_state=RANDOM_STATE)
    comp = {}
    for split in ('val', 'test'):
        Xp, yp = ds['X_' + split], ds['y_' + split]
        idx, natural = xcs_sample(yp, model.predict(Xp), XCS_N_RANDOM[split], rng)
        df = xcs_components(model, explainer, lime_exp, Xp[idx], rng)
        df['true'], df['correct'], df['natural'] = yp[idx], df['pred'].values == yp[idx], natural
        df['jaccard'] = df['jaccard'].fillna(0.0)
        df['xcs'] = XCS_W[0] * df['confidence'] + XCS_W[1] * (1 - df['instability']) + XCS_W[2] * df['jaccard']
        comp[split] = df
        print(f'  {split}: n={len(df)} natural={natural.sum()} errors={(~df.correct).sum()}')

    va, te = comp['val'], comp['test']
    feats_lr = ['confidence', 'instability', 'jaccard']
    learned = None
    if (~va.correct).sum() >= 5 and va.correct.sum() >= 5:
        sc = StandardScaler().fit(va[feats_lr])
        lr = LogisticRegression(class_weight='balanced', max_iter=1000).fit(sc.transform(va[feats_lr]), va.correct)
        te['xcs_learned'] = lr.predict_proba(sc.transform(te[feats_lr]))[:, 1]
        va['xcs_learned'] = lr.predict_proba(sc.transform(va[feats_lr]))[:, 1]
        learned = dict(zip(feats_lr, lr.coef_[0].round(4)))
    for split, df in comp.items():
        df.to_csv(OUT / 'xcs' / f'xcs_components_{name}_{split}.csv', index=False)

    scores = {'confidence': te['confidence'].values, 'margin': te['margin'].values,
              '1-instability': 1 - te['instability'].values, 'jaccard': te['jaccard'].values,
              'xcs_fixed': te['xcs'].values}
    if 'xcs_learned' in te:
        scores['xcs_learned'] = te['xcs_learned'].values
    corr, nat = te['correct'].values, te['natural'].values
    for s, v in scores.items():
        r = auroc_ci(corr, v, rng, base=scores['confidence'] if s != 'confidence' else None)
        xcs_eval_rows.append({'dataset': name, 'score': s, 'n': len(te), 'n_errors': int((~corr).sum()),
                              'auroc': r['auroc'], 'auroc_lo': r['lo'], 'auroc_hi': r['hi'],
                              'auroc_minus_conf': r['diff'], 'diff_lo': r['diff_lo'], 'diff_hi': r['diff_hi'],
                              'cohens_d': cohens_d(v[corr], v[~corr]),
                              'aurc_natural': aurc(corr[nat], v[nat]) if (~corr[nat]).any() else np.nan})

    # Thresholds: legacy 0.3 and a validation-selected one (Youden J), evaluated on test.
    thr_rows = {}
    for label, thr in [('fixed_0.3', XCS_THRESHOLD), ('val_youden', None)]:
        if thr is None:
            if va.correct.all() or (~va.correct).all():
                continue
            fpr_, tpr_, th_ = roc_curve(~va.correct.values, -va['xcs'].values)
            thr = float(-th_[np.argmax(tpr_ - fpr_)])
        flag = te['xcs'].values < thr
        thr_rows[label] = {'threshold': thr,
                           'flag_rate_natural': float(flag[nat].mean()),
                           'error_recall': float(flag[~corr].mean()) if (~corr).any() else np.nan,
                           'flag_precision_natural': float((~corr[nat & flag]).mean()) if (nat & flag).any() else np.nan}
    xcs_summary[name] = {'learned_lr_coefficients_standardised': learned, 'thresholds': thr_rows,
                         'n_val': len(va), 'n_test': len(te), 'test_errors': int((~corr).sum()),
                         'natural_error_rate': float((~corr[nat]).mean())}

    # SHAP–LIME agreement per predicted class (natural test sample, correct predictions).
    sub = te[te.natural & te.correct]
    for c, g in sub.groupby('pred'):
        agree_rows.append({'dataset': name, 'class': ds['class_names'][c], 'n': len(g),
                           'jaccard_mean': g['jaccard'].mean(), 'jaccard_sd': g['jaccard'].std(),
                           'instability_mean': g['instability'].mean()})
    for col in ('t_conf_ms', 't_shap_ms', 't_lime_ms'):
        timing_rows.append({'dataset': name, 'component': col, 'median_ms': te[col].median(),
                            'p95_ms': te[col].quantile(0.95)})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ok, col, lab in ((True, '#2b7bba', 'correct'), (False, '#d7301f', 'misclassified')):
        axes[0].hist(te.loc[te.correct == ok, 'xcs'], bins=30, alpha=0.6, color=col, density=True, label=lab)
    axes[0].axvline(XCS_THRESHOLD, ls='--', color='gray', lw=1)
    axes[0].set_xlabel('XCS (a-priori weights)'); axes[0].set_ylabel('Density'); axes[0].legend()
    axes[0].set_title(f'{name} — test XCS by correctness')
    if (~corr[nat]).any():
        for s in [k for k in ('confidence', 'xcs_fixed', 'xcs_learned') if k in scores]:
            o = np.argsort(-scores[s][nat]); risk = np.cumsum(~corr[nat][o]) / np.arange(1, nat.sum() + 1)
            axes[1].plot(np.arange(1, nat.sum() + 1) / nat.sum(), risk, label=s)
        axes[1].set_xlabel('Coverage'); axes[1].set_ylabel('Error rate among accepted'); axes[1].legend()
        axes[1].set_title('Risk–coverage (natural test sample)')
    plt.tight_layout(); plt.savefig(OUT / 'plots' / f'xcs_{name}.png', dpi=200); plt.close()

pd.DataFrame(xcs_eval_rows).to_csv(OUT / 'tables' / 'xcs_evaluation.csv', index=False)
pd.DataFrame(agree_rows).to_csv(OUT / 'tables' / 'shap_lime_agreement.csv', index=False)
pd.DataFrame(timing_rows).to_csv(OUT / 'tables' / 'xcs_latency.csv', index=False)
print('\n', pd.DataFrame(xcs_eval_rows)[['dataset', 'score', 'n_errors', 'auroc', 'auroc_minus_conf', 'aurc_natural']].round(4))


# ── 7. Summary for the paper ─────────────────────────────────────────────────
summary = {'config': {'max_per_dataset': MAX_PER_DATASET, 'test_size': TEST_SIZE, 'val_size': VAL_SIZE,
                      'top_k_features': TOP_K_FEATURES, 'xcs_weights': XCS_W, 'xcs_neighbours': XCS_NEIGHBOURS,
                      'xcs_noise_std': XCS_NOISE_STD, 'lime_samples': LIME_SAMPLES, 'seed': RANDOM_STATE,
                      'smoke_test': SMOKE,
                      'versions': {'xgboost': xgb.__version__, 'lightgbm': lgb.__version__,
                                   'shap': shap.__version__, 'numpy': np.__version__}},
           'datasets': {n: {'path': PATHS.get(n), 'cleaning': ds['log'], 'split': ds['split_sizes'],
                            'smote': ds['smote'], 'classes': ds['class_names'], 'features': ds['features']}
                        for n, ds in DATASETS.items()},
           'xcs': xcs_summary}
save_json(summary, OUT / 'summary.json')
for name, ds in DATASETS.items():
    import joblib
    joblib.dump({'model': ds['best_model'], 'scaler': ds['scaler'], 'features': ds['features'],
                 'label_encoder': ds['le']}, OUT / 'models' / f'xgb_bundle_{name}.joblib')
print('\nDone. Outputs in', OUT.resolve())
