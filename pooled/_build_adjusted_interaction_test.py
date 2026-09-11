#!/usr/bin/env python3
"""Builder for adjustedInteractionTest.ipynb (CCM R2 revision).

Answers Reviewer #1's only remaining request on manuscript CCMED-D-26-00748R1:

    "add another version of the interaction test where you adjust for main confounders
    or use IPW."

The paper's Table 2 reports a plain likelihood-ratio test for a CATE x TTM interaction
(logit P(Y=1) = b0 + b1*T + b2*Chat vs. + b3*(T x Chat), LRT chi-square 1 df) with no
confounder adjustment and no weighting. eTable 21 already has an IPW-weighted version, but
only for the reduced-feature per-dataset sensitivity analysis, observational cohorts only.
This notebook is the confounder-adjusted AND IPW-weighted counterpart to the PRIMARY
Table 2 test, run on the same cohorts/features/preprocessing and the same CausalForestDML
CATE model the paper already uses, at the paper's own 70/30 split evaluation.

The statistical-test functions live in the shared, tested module `pooled/hte_tests.py`
(imported below) so the LRT reproduces Table 2 bit-for-bit and the new adjusted/IPW/DR
tests are not copy-pasted inline. Everything else here (loaders, preprocessing,
CausalForestDML fitting, propensity extraction) is copied verbatim from
`pooled/_build_per_dataset_analysis.py` so this notebook sees exactly the same cohorts,
features, and preprocessing as the published analysis.

Edit THIS file and re-run it; do not hand-edit the .ipynb JSON.

Run on the cluster from this folder:
    bash run_adjusted_interaction_test.sh smoke   # fast path check, eICU only by default
    bash run_adjusted_interaction_test.sh full     # all 4 datasets

Writes adjusted_interaction_results.csv and adjusted_interaction_manifest.jsonl next to the
notebook.
"""
import json, os, uuid

CELLS = []
def md(src):
    return
def code(src): CELLS.append(("code", src))

# ============================================================================================ #
# Cell 1 — CONFIG
# ============================================================================================ #
code("""# CONFIG
import os

SEED = 42
TEST_SIZE = 0.30           # the paper's 70/30 evaluation split
PS_CLIP = (0.05, 0.95)
# Where the per-patient evaluation vectors (y, TTM, cate, ps, confounders) are written, one
# CSV per dataset/outcome. Re-fitting the causal forest is the only slow step here; with these
# saved, any later variant of the interaction test runs in seconds without a re-fit.
CATE_DIR = 'cate_vectors'
REPO_ROOT = os.path.abspath('..')

# Predictor CSVs (the filenames the *AnalysisDML notebooks actually read).
CSV = {
    'eICU':     os.path.join(REPO_ROOT, 'eICU',     'eICUPredictorsDiag.csv'),
    'PMAP':     os.path.join(REPO_ROOT, 'pmap',     'PMAP_Predictors2.csv'),
    'MIMIC-IV': os.path.join(REPO_ROOT, 'mimiciv',  'MIMIC_Predictors.csv'),
    'HYPERION': os.path.join(REPO_ROOT, 'hyperion', 'predictorsDf.csv'),
}
DATASETS = [
    dict(name='eICU',     observational=True),
    dict(name='PMAP',     observational=True),
    dict(name='MIMIC-IV', observational=True),
    dict(name='HYPERION', observational=False),  # randomized: no IPW, no DR (Cell 7 records NaN)
]

# Smoke runs can restrict to a subset of datasets (run_adjusted_interaction_test.sh sets this
# to 'eICU' for `smoke`); defaults to all four.
SMOKE_DATASETS = os.environ.get('SMOKE_DATASETS', 'eICU,PMAP,MIMIC-IV,HYPERION').split(',')
DATASETS = [d for d in DATASETS if d['name'] in SMOKE_DATASETS]

# Curated feature lists = the 'columns' each *AnalysisDML notebook trains on. Identical to
# pooled/_build_per_dataset_analysis.py so this notebook's cohorts/features match the paper.
CURATED = {
 'eICU': ['gender','age','bmi',
    'nurse_first_Non-Invasive BP Systolic','nurse_first_Non-Invasive BP Diastolic',
    'nurse_first_Non-Invasive BP Mean','nurse_first_Heart Rate','nurse_first_O2 Saturation',
    'lab_first_Respiratory Rate','lab_first_FiO2','nurse_first_GCS Total','nurse_first_Motor',
    'nurse_first_QTc','lab_first_pH','lab_first_paO2','lab_first_paCO2','lab_first_bicarbonate',
    'lab_first_lactate','lab_first_WBC x 1000','lab_first_Hgb','lab_first_platelets x 1000',
    'lab_first_sodium','lab_first_potassium','lab_first_BUN','lab_first_creatinine',
    'lab_first_calcium','lab_first_magnesium','lab_first_glucose','lab_first_troponin - T',
    'diagnosis_initial rhythm: ventricular fibrillation',
    'diagnosis_initial rhythm: ventricular tachycardia',
    'diagnosis_initial rhythm: pulseless electrical activity',
    'diagnosis_initial rhythm: asystole'],
 'PMAP': ['gender','age','first_mGCS','flo_first_r_cpn_glasgow_coma_scale_score',
    'flo_first_bp_systolic','flo_first_bp_diastolic','flo_first_r_map',
    'flo_first_r_ed_pre-arrival_pulse_(heart_rate)','flo_first_r_sao2','flo_first_r_fio2',
    'flo_first_r_sofa_score','flo_first_r_bmi','flo_first_r_pao2','flo_first_r_paco2',
    'flo_first_r_resp_ph','lab_first_lactate','lab_first_troponin','lab_first_hemoglobin',
    'lab_first_platelet_count','lab_first_creatinine,whole_blood','lab_first_glucose,whole_blood',
    'lab_first_potassium,whole_blood','lab_first_sodium,whole_blood','lab_first_calcium,_serum',
    'lab_first_magnesium','asystole','pea','VF'],
 'MIMIC-IV': ['gender','age','bmi','first_mGCS',
    'chart_first_heart_rate','chart_first_o2_saturation_pulseoxymetry','chart_first_respiratory_rate',
    'chart_first_fio2_(ch)','chart_first_non_invasive_blood_pressure_systolic',
    'chart_first_non_invasive_blood_pressure_diastolic','chart_first_non_invasive_blood_pressure_mean',
    'chart_first_ph_(arterial)','chart_first_arterial_o2_pressure','chart_first_arterial_co2_pressure',
    'chart_first_lactic_acid','chart_first_wbc','chart_first_hemoglobin','lab_first_platelet_count',
    'chart_first_sodium_(serum)','lab_first_potassium_(serum)','chart_first_bun',
    'chart_first_creatinine_(serum)','chart_first_calcium_non-ionized','chart_first_magnesium',
    'chart_first_glucose_(serum)','lab_first_troponin-t','chart_first_qtc',
    'long_title_ventricular_fibrillation'],
 'HYPERION': ['J0_AGE','J0_SEX','J0_BMI','J0_PAS','J0_PAD','J0_PAM','J0_FC','J0_SPO2',
    'J0_GLASGOW','J0_MOTRICE','J0_RYTHM','J0_NOFLOW','J0_LOWFLOW','J0_IGSII',
    'BIO_LEUCO','BIO_HEMO','BIO_PLAQ','BIO_SODIUM','BIO_POTAS','BIO_UREE','BIO_CREAT',
    'BIO_CALCIUM','BIO_MAGNE','BIO_GLYCEMI','BIO_LACTAT','BIO_TROPO','BIO_PH','BIO_PAO2',
    'BIO_PACO2','BIO_BICARB'],
}

# Require the last mGCS to be measured >= this many time-units after the first, to avoid
# baseline~=outcome leakage in the neurologic analysis. Identical to the per-dataset notebook.
MIN_MGCS_GAP = {'eICU': 0, 'PMAP': 0, 'MIMIC-IV': 0}

# --- new for this notebook (adjusted / IPW-weighted interaction test) ---
# This notebook only ever evaluates the paper's own 70/30 split (the primary evaluation in
# Table 2); crossfit() below also supports a 5-fold out-of-fold path for future robustness
# work, but it is not exercised here, so there is no list of eval methods to configure.
EVAL_METHOD = 'split'
N_SPLITS = 5    # unused by this notebook (only referenced inside crossfit()'s 'crossfit' branch)

# CONFOUNDERS (the prespecified adjustment set) is defined in Cell 3, once we can comment on
# how each name resolves against the curated feature lists above.
""")

# ============================================================================================ #
# Cell 2 — imports + path bootstrap
# ============================================================================================ #
md("""The notebook is run from either `pooled/` or `summarized_results/` (mirrored copies), so
it must find `pooled/hte_tests.py` either way before importing it.""")

code("""import os, sys
_here = os.path.abspath('')
for _c in [_here, os.path.join(_here, '..', 'pooled'), os.path.join(_here, 'pooled')]:
    if os.path.isfile(os.path.join(_c, 'hte_tests.py')) and _c not in sys.path:
        sys.path.insert(0, _c); break
import hte_tests as ht

import re
import time
import traceback
import numpy as np
import pandas as pd
import sklearn
import xgboost
import econml
import statsmodels
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from econml.dml import CausalForestDML

np.random.seed(SEED)
RESULTS = []      # one row per (dataset, outcome)
""")

# ============================================================================================ #
# Cell 3 — prespecified confounder set
# ============================================================================================ #
code("""# Prespecified confounder set (Methods: age, sex, initial rhythm, initial motor GCS, lactate,
# and an acid-base/illness-severity marker), expressed as CONCEPTS rather than fixed column
# names. Each concept carries (exact candidate names, regex fallback). The resolver in Cell 7
# tries the exact names first, then the regex against the dataset's real columns, and prints
# exactly which column it matched for every concept. A first run with hardcoded names silently
# lost three confounders (PMAP lactate, MIMIC-IV rhythm, HYPERION sex) because the CSV column
# was spelled differently -- concepts + regex + a printed audit is what stops that recurring.
#
# IMPORTANT: confounders are resolved against the dataset's RAW columns, not the CURATED model
# feature list. They enter only the second-stage interaction regression, never the CATE model,
# so adding one cannot change the CATE and cannot disturb the eTable 21 reproduction. A
# confounder the CATE model never saw is exactly what "adjust for main confounders" means.
#
# HYPERION has no rhythm concept on purpose: the trial enrolled only non-shockable rhythms, so
# any rhythm indicator is constant there.
CONFOUNDER_CONCEPTS = {
 'eICU': {
   'age':       (['age'], r'^age$'),
   'sex':       (['gender'], r'^(gender|sex)$'),
   'motor_gcs': (['nurse_first_Motor'], r'motor'),
   'lactate':   (['lab_first_lactate'], r'lactat'),
   'acid_base': (['lab_first_pH'], r'(^|_)ph$|_ph[_ ]|\\bph\\b'),
   # Every rhythm pattern REQUIRES the word 'ventricular'. A looser r'fibrill' matched
   # ATRIAL fibrillation in MIMIC-IV on the first run -- not a shockable arrest rhythm, and a
   # clinically wrong adjustment. Do not relax these.
   'rhythm':    (['diagnosis_initial rhythm: ventricular fibrillation',
                  'diagnosis_initial rhythm: ventricular tachycardia'],
                 r'ventricular[_ ]?(fibrillation|tachycardia)'),
 },
 'PMAP': {
   'age':       (['age'], r'^age$'),
   'sex':       (['gender'], r'^(gender|sex)$'),
   'motor_gcs': (['first_mGCS'], r'first_mgcs|motor'),
   'lactate':   (['lab_first_lactate'], r'lactat'),
   'acid_base': (['flo_first_r_resp_ph'], r'(^|_)ph$|_ph[_ ]|resp_ph'),
   'severity':  (['flo_first_r_sofa_score'], r'sofa|apache|saps'),
   'rhythm':    (['VF'], r'^vf$|ventricular[_ ]?fibrill|shockable'),
 },
 'MIMIC-IV': {
   'age':       (['age'], r'^age$'),
   'sex':       (['gender'], r'^(gender|sex)$'),
   'motor_gcs': (['first_mGCS'], r'first_mgcs|motor'),
   'lactate':   (['chart_first_lactic_acid'], r'lactat|lactic'),
   'acid_base': (['chart_first_ph_(arterial)'], r'(^|_)ph$|_ph[_( ]|ph_\\(arterial\\)'),
   'rhythm':    (['long_title_ventricular_fibrillation'],
                 r'ventricular[_ ]?fibrill|shockable'),
 },
 'HYPERION': {
   'age':       (['J0_AGE'], r'^j0_age$|\\bage\\b'),
   'sex':       (['J0_SEX', 'J0_SEXE', 'SEXE', 'sexe'], r'sex|genre|gender'),
   'motor_gcs': (['J0_MOTRICE'], r'motrice|motor'),
   'lactate':   (['BIO_LACTAT'], r'lactat'),
   'acid_base': (['BIO_PH'], r'(^|_)ph$|bio_ph'),
   'severity':  (['J0_IGSII'], r'igsii|igs2|saps'),
 },
}
""")

# ============================================================================================ #
# Cell 4 — loaders (verbatim from _build_per_dataset_analysis.py)
# ============================================================================================ #
md("""Loaders, self-contained and restricted to the curated DML columns -- copied verbatim from
`pooled/_build_per_dataset_analysis.py` so this notebook sees the exact same cohorts.""")

code("""UNSCORABLE = 'Unable to score due to medication'

def _mgcs_gap_ok(dt_first, dt_last, name):
    gap = (dt_last - dt_first).abs()
    keep = gap >= MIN_MGCS_GAP.get(name, 0)
    if MIN_MGCS_GAP.get(name, 0) > 0:
        print(f"  [{name}] mGCS time-gap filter (>= {MIN_MGCS_GAP[name]}): "
              f"excluding {(~keep).sum()} of {len(keep)} (median gap {gap.median():.0f})")
    return keep

def load_eicu():
    df = pd.read_csv(CSV['eICU'])
    f = (df['LastMGCS'] != UNSCORABLE) & (~df['LastMGCS'].isna())
    f = f & (df['FirstMGCSTime'] != df['LastMGCSTime'])
    f = f & _mgcs_gap_ok(df['FirstMGCSTime'], df['LastMGCSTime'], 'eICU')   # leakage guard
    for c in ['FirstGCS', 'FirstMGCS', 'LastMGCS', 'LastGCS']:
        df.loc[df[c] == UNSCORABLE, c] = np.nan
    df.loc[df['DeathAtDischarge'] == 1, 'LastMGCS'] = 1
    df['gender'] = (df['gender'] == 'Male').astype(int)
    df.loc[f, 'LastMGCSPositive'] = (df.loc[f, 'LastMGCS'].astype(float) == 6).astype(int)
    df = df[f & (df['nurse_first_Motor'] != 6) & ~df['Hypothermia'].isna()].copy()
    return df.rename(columns={'Hypothermia': 'TTM', 'DeathAtDischarge': 'mortality',
                              'LastMGCSPositive': 'neuro_favorable'})

def _load_epic(path, name):
    df = pd.read_csv(path)
    f = (df['first_mGCS_time'] != df['last_mGCS_time'])
    f = f & _mgcs_gap_ok(df['first_mGCS_time'], df['last_mGCS_time'], name)   # leakage guard
    df.loc[df['death_at_disch'] == 1, 'last_mGCS'] = 1
    df.loc[f, 'LastMGCSPositive'] = (df.loc[f, 'last_mGCS'].astype(float) == 6).astype(int)
    df = df[f & (df['first_mGCS'] != 6) & ~df['hypothermia'].isna()].copy()
    return df.rename(columns={'hypothermia': 'TTM', 'death_at_disch': 'mortality',
                              'LastMGCSPositive': 'neuro_favorable'})

def load_pmap():  return _load_epic(CSV['PMAP'], 'PMAP')
def load_mimic(): return _load_epic(CSV['MIMIC-IV'], 'MIMIC-IV')

def load_hyperion():
    df = pd.read_csv(CSV['HYPERION'])
    df = df[df['groupe'] != 2].copy()                 # drop non-randomized/excluded arm
    df['TTM'] = (df['groupe'] == 1).astype(int)        # 1 = hypothermia, 0 = normothermia
    return df.rename(columns={'hospital_mortality': 'mortality', 'CPC12': 'neuro_favorable'})

LOADERS = {'eICU': load_eicu, 'PMAP': load_pmap, 'MIMIC-IV': load_mimic, 'HYPERION': load_hyperion}

def _pick(cols, exact, pattern):
    \"\"\"Resolve one confounder concept to real column name(s): exact names first, then the
    regex fallback. Regex hits prefer a 'first'-prefixed column (baseline value) and then the
    shortest name, and are capped at 2 so a loose pattern cannot drag in a whole family.\"\"\"
    hits = [c for c in exact if c in cols]
    if hits:
        return hits, 'exact', []
    if pattern:
        rx = re.compile(pattern, re.I)
        hits = [c for c in cols if rx.search(str(c))]
        if hits:
            hits = sorted(hits, key=lambda c: (0 if 'first' in str(c).lower() else 1,
                                               len(str(c))))
            # Take ONE column for a fallback match. The first run matched two PMAP lactate
            # assays and two MIMIC rhythm columns, which both then failed downstream. The
            # runners-up are reported so the choice is visible, not silent.
            return hits[:1], 'regex', hits[1:]
    return [], 'missing', []


def resolve_confounder_columns(name, cols):
    \"\"\"Map CONFOUNDER_CONCEPTS[name] onto this dataset's real columns.
    Returns (list of columns, audit dict concept -> {columns, how}).\"\"\"
    picked, audit = [], {}
    for concept, (exact, pattern) in CONFOUNDER_CONCEPTS[name].items():
        hits, how, alts = _pick(cols, exact, pattern)
        audit[concept] = {'columns': hits, 'how': how, 'alternatives': alts}
        picked.extend([c for c in hits if c not in picked])
    return picked, audit


# Filled in by load_full so the audit reflects resolution against each dataset's RAW columns,
# not the already-subset confounder frame.
CONFOUNDER_RESOLUTION = {}


def load_full(name, outcome_col):
    df = LOADERS[name]()
    cand = [c for c in CURATED[name] if c in df.columns]
    X = df[cand].apply(pd.to_numeric, errors='coerce')
    # Confounders resolve against the RAW columns, independent of CURATED. They enter only the
    # second-stage regression, never the CATE model, so this cannot change the CATE.
    zcols, _audit = resolve_confounder_columns(name, list(df.columns))
    zcols = [c for c in dict.fromkeys(zcols) if c in df.columns]   # dedupe + guard
    Zc = (df[zcols].apply(pd.to_numeric, errors='coerce') if zcols
          else pd.DataFrame(index=df.index))
    # A column that is entirely non-numeric coerces to all-NaN; it carries no information and
    # breaks the scaler/imputer downstream. Drop it and record why.
    allnan = [c for c in Zc.columns if Zc[c].notna().sum() == 0]
    if allnan:
        Zc = Zc.drop(columns=allnan)
        for concept, a in _audit.items():
            a['dropped_all_nan'] = [c for c in a['columns'] if c in allnan]
    CONFOUNDER_RESOLUTION[name] = _audit
    T = pd.to_numeric(df['TTM'], errors='coerce')
    y = pd.to_numeric(df[outcome_col], errors='coerce')
    m = (T.notna() & y.notna()).values
    return (X[m].reset_index(drop=True), T[m].astype(int).reset_index(drop=True),
            y[m].astype(int).reset_index(drop=True), Zc[m].reset_index(drop=True))


def preprocess(X_tr, X_te):
    \"\"\"Scale numeric (>2 levels) on train (StandardScaler is NaN-safe), KNN-impute (k=10) on
    train; apply to test. Column set is fixed (keep_empty_features) so out-of-fold rows align.\"\"\"
    cols = list(X_tr.columns)
    num = [c for c in cols if X_tr[c].dropna().nunique() > 2]
    scaler = StandardScaler().fit(X_tr[num])
    imputer = KNNImputer(n_neighbors=10, keep_empty_features=True)

    def tf(X, fit=False):
        X = X.copy()
        if num:
            X[num] = scaler.transform(X[num])
        arr = imputer.fit_transform(X) if fit else imputer.transform(X)
        return pd.DataFrame(arr, columns=cols, index=X.index)

    return tf(X_tr, fit=True), tf(X_te)
""")

# ============================================================================================ #
# Cell 5 — estimators (fit_cf, cf_propensity, _fit_eval, verbatim)
# ============================================================================================ #
code("""def fit_cf(X_tr, T_tr, y_tr, seed=SEED):
    cf = CausalForestDML(
        model_y=XGBClassifier(max_depth=3, n_estimators=50, random_state=seed),
        model_t=XGBClassifier(max_depth=2, n_estimators=20, random_state=seed),
        discrete_treatment=True, discrete_outcome=True,
        random_state=seed, n_jobs=-1)
    cf.fit(y_tr, T_tr, X=X_tr, cache_values=True)
    return cf


def cf_propensity(cf, X):
    preds = []
    for mc in cf.models_t:
        for mdl in mc:
            p = mdl.predict_proba(np.asarray(X))
            preds.append(p[:, 1] if p.ndim == 2 else np.ravel(p))
    return np.clip(np.mean(np.vstack(preds), axis=0), 1e-6, 1 - 1e-6)


def _fit_eval(Xtr, Xte, Ttr, ytr, observational):
    cf = fit_cf(Xtr, Ttr, ytr)
    cate = np.ravel(cf.effect(Xte))
    lo, hi = cf.effect_interval(Xte, alpha=0.05)
    ps = cf_propensity(cf, Xte) if observational else np.full(len(Xte), np.nan)
    risk = LogisticRegression(max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return cate, np.ravel(lo), np.ravel(hi), ps, risk
""")

# ============================================================================================ #
# Cell 6 — evaluation-set builder (crossfit, generalised to take eval_method as an argument)
# ============================================================================================ #
md("""Datasets here are one row per patient (no repeated measures), so patients are the rows
and ordinary/robust SEs from `hte_tests` are already at the patient level -- no cluster-robust
SEs are needed. This notebook only calls crossfit() with eval_method='split' (EVAL_METHOD in
Cell 1); the 'crossfit' (5-fold OOF) branch is kept, copied from the reference builder's data
layer, for future robustness work but is not exercised by Cell 7.""")

code("""def crossfit(name, outcome_col, observational, eval_method, cols=None):
    \"\"\"HTE-estimation set. eval_method='split' uses a single stratified 70/30 train/test
    (matches the main paper) and returns the held-out test rows; eval_method='crossfit'
    predicts every patient's CATE out-of-fold (full n, not seed-dependent) and is kept here as
    copied data-layer code but is not called by this notebook. `cols` optionally restricts the
    features (defaults to the dataset's full CURATED list via load_full).

    NOTE: cross-fitting does not increase the sample size -- n is always the number of
    independent patients.
    \"\"\"
    X, T, y, Zc = load_full(name, outcome_col)
    if cols is not None:
        X = X[[c for c in cols if c in X.columns]]
    strat = (y.astype(str) + '_' + T.astype(str)).values
    assert len(np.unique(np.arange(len(y)))) == len(y)   # one row per patient; no clustering needed

    if eval_method == 'split':
        tr, te = train_test_split(np.arange(len(y)), test_size=TEST_SIZE,
                                  random_state=SEED, stratify=strat)
        Xtr, Xte = preprocess(X.iloc[tr], X.iloc[te])
        # Confounders get their own train-fitted scaler/imputer, kept entirely separate from X
        # so the CATE model is byte-for-byte what it was before this adjustment set existed.
        if Zc.shape[1]:
            _, Zte = preprocess(Zc.iloc[tr], Zc.iloc[te])
        else:
            Zte = Zc.iloc[te]
        cate, lo, hi, ps, risk = _fit_eval(Xtr, Xte, T.iloc[tr], y.iloc[tr], observational)
        return dict(X=Xte.reset_index(drop=True), Z=Zte.reset_index(drop=True),
                    T=T.iloc[te].values, y=y.iloc[te].values,
                    n_features=Xte.shape[1], cate=cate, lo=lo, hi=hi, ps=ps, risk=risk)

    n = len(y)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = {k: np.full(n, np.nan) for k in ['cate', 'ps', 'risk', 'lo', 'hi']}
    oof_X = pd.DataFrame(np.nan, index=range(n), columns=X.columns, dtype=float)
    oof_Z = pd.DataFrame(np.nan, index=range(n), columns=Zc.columns, dtype=float)
    for tr, te in skf.split(X, strat):
        Xtr, Xte = preprocess(X.iloc[tr], X.iloc[te])
        oof_X.iloc[te] = Xte.values
        if Zc.shape[1]:
            _, Zte = preprocess(Zc.iloc[tr], Zc.iloc[te])
            oof_Z.iloc[te] = Zte.values
        cate, lo, hi, ps, risk = _fit_eval(Xtr, Xte, T.iloc[tr], y.iloc[tr], observational)
        oof['cate'][te] = cate; oof['lo'][te] = lo; oof['hi'][te] = hi
        oof['ps'][te] = ps; oof['risk'][te] = risk
    return dict(X=oof_X, Z=oof_Z, T=T.values, y=y.values, n_features=oof_X.shape[1], **oof)
""")

# ============================================================================================ #
# Cell 7 — run
# ============================================================================================ #
md("""For each dataset x outcome, build the evaluation set ONCE (fits the CausalForest once,
which also gives the propensity) and then run all four interaction tests on those same
evaluation rows. Each (dataset, outcome) unit is wrapped in try/except so one failure does not
abort the whole cluster run.""")

code("""def resolve_confounders():
    \"\"\"Concept-level resolution audit for CONFOUNDER_CONCEPTS (Cell 3).

    For every dataset and every concept, prints the column actually matched and HOW it was
    matched -- 'exact' (the intended name was present), 'regex' (the intended name was absent
    and the fallback pattern found a column: CHECK THESE, the match may be the wrong variable),
    or 'missing' (nothing matched; that concept is genuinely not adjusted for).

    Read the [RESOLVE][WARN] lines before using any adjusted p-value. A first run with
    hardcoded names silently dropped PMAP lactate, MIMIC-IV rhythm and HYPERION sex, and the
    adjusted column was quietly wrong for three of four cohorts.

    Returns {dataset: {concept: {columns, how}}}.
    \"\"\"
    audits = {}
    print('=== Confounder resolution (prespecified adjustment set, Cell 3) ===')
    for cfg in DATASETS:
        name = cfg['name']
        Xraw, _T, _y, Zc = load_full(name, 'mortality')
        # Audit recorded by load_full against the dataset's RAW columns.
        audit = CONFOUNDER_RESOLUTION[name]
        audits[name] = audit
        n_ok = sum(1 for a in audit.values() if a['how'] == 'exact')
        print(f"  [{name}] {n_ok}/{len(audit)} concepts matched exactly")
        for concept, a in audit.items():
            cols, how = a['columns'], a['how']
            const = [c for c in cols if c in Zc.columns and Zc[c].dropna().nunique() <= 1]
            flag = ''
            if how == 'regex':
                flag = '   <-- fallback match, VERIFY'
            elif how == 'missing':
                flag = '   <-- NOT ADJUSTED FOR'
            if const:
                flag += f'   <-- constant on cohort {const}, will be dropped'
            print(f"      {concept:12s} {how:8s} {cols}{flag}")
        for concept, a in audit.items():
            if a['how'] != 'exact':
                print(f"[RESOLVE][WARN] {name}/{concept}: matched by {a['how']} -> "
                      f"{a['columns'] if a['columns'] else 'NOTHING'}")
    return audits

CONFOUNDER_AUDIT = resolve_confounders()

RUN_META = {
    'seed': SEED, 'eval_method': EVAL_METHOD, 'test_size': TEST_SIZE, 'ps_clip': list(PS_CLIP),
    'model': 'CausalForestDML', 'datasets': [d['name'] for d in DATASETS],
    'outcomes': ['mortality', 'neuro'],
    'confounder_concepts': {k: {c: list(v[0]) for c, v in spec.items()}
                            for k, spec in CONFOUNDER_CONCEPTS.items()},
    'confounder_resolution': CONFOUNDER_AUDIT,
    'package_versions': {
        'numpy': np.__version__, 'pandas': pd.__version__,
        'statsmodels': statsmodels.__version__, 'scikit-learn': sklearn.__version__,
        'xgboost': xgboost.__version__, 'econml': econml.__version__,
    },
    'started': time.strftime('%Y-%m-%d %H:%M:%S'),
}
MANIFEST_PATH = 'adjusted_interaction_manifest.jsonl'
ht.init_manifest(MANIFEST_PATH, RUN_META)


def _fmt(p):
    return 'nan' if pd.isna(p) else f'{p:.3g}'


for cfg in DATASETS:
    name, observational = cfg['name'], cfg['observational']
    for outcome_key in ['mortality', 'neuro']:
        outcome_col = 'mortality' if outcome_key == 'mortality' else 'neuro_favorable'
        t0 = time.time()
        print(f"[ADJINT] START {EVAL_METHOD} | {name} | {outcome_key}", flush=True)
        try:
            ev = crossfit(name, outcome_col, observational, EVAL_METHOD)
            y_ev, T_ev, cate = ev['y'], ev['T'], ev['cate']
            Z = ev['Z']
            Zcols = list(Z.columns)
            # Surface confounders that never reached Z at all. `dropped_confounders` below only
            # covers columns dropped at fit time (constant / near-constant); a concept that was
            # never matched, or that coerced to all-NaN in load_full, would otherwise vanish
            # from the adjustment set with no trace in the results CSV.
            unresolved = []
            for _concept, _a in CONFOUNDER_RESOLUTION.get(name, {}).items():
                if _a['how'] == 'missing':
                    unresolved.append(f'{_concept}:not-found')
                elif _a.get('dropped_all_nan'):
                    unresolved.append(f'{_concept}:all-nan')
            if unresolved:
                print(f"[ADJINT][WARN] {name}/{outcome_key}: not adjusted for {unresolved}",
                      flush=True)
            n = int(len(y_ev)); n_events = int(np.nansum(y_ev))

            # Each test is isolated. A singular design in ONE test (e.g. a near-constant
            # rhythm flag making the adjusted fit non-identifiable) must not discard the other
            # three results for that dataset/outcome, which is what happened on the first run.
            NA_LRT = {'p': np.nan, 'lr_stat': np.nan, 'or': np.nan, 'ci_low': np.nan,
                      'ci_high': np.nan, 'note': '', 'n_confounders': len(Zcols),
                      'confounders': Zcols, 'dropped_confounders': []}
            NA_W = {'wald_p': np.nan, 'wald_p_naive': np.nan, 'or': np.nan, 'ci_low': np.nan,
                    'ci_high': np.nan, 'ess': np.nan, 'max_weight': np.nan, 'note': '',
                    'n_confounders': len(Zcols), 'confounders': Zcols,
                    'dropped_confounders': []}

            def _safe(label, fn, fallback):
                try:
                    return fn()
                except Exception as exc:
                    print(f"[ADJINT][WARN] {name}/{outcome_key}: {label} failed "
                          f"({type(exc).__name__}: {exc}) -- reporting NaN for this test only",
                          flush=True)
                    return dict(fallback, note=f'{type(exc).__name__}: {exc}')

            r_unadj = _safe('unadjusted LRT',
                            lambda: ht.lrt_cate_interaction(y_ev, T_ev, cate), NA_LRT)
            r_adj = _safe('adjusted LRT',
                          lambda: ht.adjusted_lrt_cate_interaction(y_ev, T_ev, cate, Z), NA_LRT)

            if observational:
                r_ipw = _safe('IPW', lambda: ht.ipw_interaction_test(
                    y_ev, T_ev, cate, ev['ps'], clip=PS_CLIP), NA_W)
                r_dr = _safe('IPW + adjusted', lambda: ht.dr_interaction_test(
                    y_ev, T_ev, cate, ev['ps'], Z, clip=PS_CLIP), NA_W)
            else:
                na_note = 'randomized: IPW not applicable'
                r_ipw = dict(NA_W, note=na_note)
                r_dr = dict(NA_W, note=na_note)

            row = {
                'dataset': name, 'outcome': outcome_key, 'status': 'ok', 'n': n,
                'n_events': n_events, 'n_features': ev['n_features'],
                'p_unadj': r_unadj['p'], 'or_unadj': r_unadj['or'],
                'ci_low_unadj': r_unadj['ci_low'], 'ci_high_unadj': r_unadj['ci_high'],
                'note_unadj': r_unadj['note'],
                'p_adj': r_adj['p'], 'or_adj': r_adj['or'],
                'ci_low_adj': r_adj['ci_low'], 'ci_high_adj': r_adj['ci_high'],
                'note_adj': r_adj['note'], 'n_confounders': r_adj['n_confounders'],
                'confounders': r_adj['confounders'],
                'dropped_confounders': r_adj['dropped_confounders'],
                'confounders_unresolved': unresolved,
                'p_ipw': r_ipw['wald_p'], 'wald_p_naive_ipw': r_ipw['wald_p_naive'],
                'or_ipw': r_ipw['or'], 'ci_low_ipw': r_ipw['ci_low'],
                'ci_high_ipw': r_ipw['ci_high'], 'ess_ipw': r_ipw.get('ess', np.nan),
                'max_weight_ipw': r_ipw.get('max_weight', np.nan), 'note_ipw': r_ipw['note'],
                'p_dr': r_dr['wald_p'], 'wald_p_naive_dr': r_dr['wald_p_naive'],
                'or_dr': r_dr['or'], 'ci_low_dr': r_dr['ci_low'], 'ci_high_dr': r_dr['ci_high'],
                'ess_dr': r_dr.get('ess', np.nan), 'max_weight_dr': r_dr.get('max_weight', np.nan),
                'note_dr': r_dr['note'],
            }
            # Persist the evaluation-row vectors. Re-fitting the causal forest is the ONLY
            # expensive part of this notebook; every test above is a second of regression on
            # these columns. Saving them means any future variant of the interaction test
            # (a different adjustment set, different weights, a different link) can be run
            # from this CSV in seconds with no forest re-fit.
            #
            # Confounders come from Z, NOT ev['X']: since confounders resolve against the raw
            # columns they need not appear in the CURATED feature list, and indexing ev['X']
            # with one that does not raised KeyError AFTER the result row had been appended --
            # which is what produced duplicated ok/fail rows on the previous run. Saving is
            # also non-fatal now, and the row is appended only once, after it.
            try:
                os.makedirs(CATE_DIR, exist_ok=True)
                cate_out = pd.DataFrame({'y': y_ev, 'TTM': T_ev, 'cate': cate, 'ps': ev['ps']})
                for c in Zcols:
                    cate_out[f'Z_{c}'] = Z[c].values
                cate_path = os.path.join(
                    CATE_DIR, f"cate_{name.replace('-', '').lower()}_{outcome_key}.csv")
                cate_out.to_csv(cate_path, index=False)
                row['cate_file'] = cate_path
                print(f"[ADJINT] saved evaluation vectors -> {cate_path} "
                      f"({len(cate_out)} rows)", flush=True)
            except Exception as exc:
                print(f"[ADJINT][WARN] {name}/{outcome_key}: could not save evaluation vectors "
                      f"({type(exc).__name__}: {exc}); results are unaffected", flush=True)

            RESULTS.append(row)
            ht.append_manifest(MANIFEST_PATH, {**row, 'elapsed_s': time.time() - t0})
            print(f"[ADJINT] {EVAL_METHOD} | {name} | {outcome_key} ... "
                  f"p_unadj={_fmt(row['p_unadj'])}, p_adj={_fmt(row['p_adj'])}, "
                  f"p_ipw={_fmt(row['p_ipw'])}", flush=True)
        except Exception as e:
            # Print the FULL stack. The first cluster run reported only 'KeyError: <column>'
            # with no frame, which made the failure undiagnosable from the executed notebook.
            print(f"[ADJINT] {EVAL_METHOD} | {name} | {outcome_key} ... "
                  f"FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            RESULTS.append({'dataset': name, 'outcome': outcome_key,
                             'status': f'fail: {type(e).__name__}: {e}'})
        print(f"[ADJINT] DONE {EVAL_METHOD} | {name} | {outcome_key} ... "
              f"elapsed={time.time() - t0:.1f}s", flush=True)
""")

# ============================================================================================ #
# Cell 8 — multiplicity (BH-FDR within each family)
# ============================================================================================ #
code("""results_df = pd.DataFrame(RESULTS)

for col_q in ['q_unadj', 'q_adj', 'q_ipw', 'q_dr']:
    results_df[col_q] = np.nan

ok = results_df['status'] == 'ok'
idx = results_df.index[ok]
for col_p, col_q in [('p_unadj', 'q_unadj'), ('p_adj', 'q_adj'),
                     ('p_ipw', 'q_ipw'), ('p_dr', 'q_dr')]:
    results_df.loc[idx, col_q] = ht.bh_fdr(results_df.loc[idx, col_p].values)

print('BH-FDR is applied within each family: one family per test variant, across the '
      '4 datasets x 2 outcomes = 8 tests (p_unadj, p_adj); p_ipw/p_dr are observational-cohort-'
      'only, so HYPERION contributes NaN and is excluded from that ranking, leaving 6. The '
      'per-dataset x outcome interaction tests are the prespecified primary comparisons; '
      'BH-FDR is reported because this is a family of repeated tests, not a single '
      'confirmatory test.')
""")

# ============================================================================================ #
# Cell 9 — outputs
# ============================================================================================ #
code("""results_df.to_csv('adjusted_interaction_results.csv', index=False)
print(f"wrote adjusted_interaction_results.csv ({len(results_df)} rows) next to this notebook. "
      f"{MANIFEST_PATH} was written incrementally during the run in Cell 7.")

DISPLAY_NAME = {'eICU': 'eICU-CRD', 'PMAP': 'PMAP', 'MIMIC-IV': 'MIMIC-IV', 'HYPERION': 'HYPERION'}


def _fmt_cell(p, note):
    if isinstance(note, str) and 'constant' in note.lower():
        return 'No effect'
    if isinstance(note, str) and 'not applicable' in note.lower():
        return '\\u2014'
    if pd.isna(p):
        return '\\u2014'
    return f'{p:.2f}'


print('All rows below use CausalForestDML (the paper\\'s primary CATE model) at the paper\\'s '
      '70/30 split evaluation.\\n')
print('#### Paste-ready markdown table for the supplement\\n')
print('| Dataset | Outcome | Unadjusted p | Adjusted p | IPW p | IPW + adjusted p |')
print('|---|---|---|---|---|---|')
for _, r in results_df[results_df['status'] == 'ok'].iterrows():
    print(f"| {DISPLAY_NAME.get(r['dataset'], r['dataset'])} | {r['outcome']} | "
          f"{_fmt_cell(r['p_unadj'], r['note_unadj'])} | {_fmt_cell(r['p_adj'], r['note_adj'])} | "
          f"{_fmt_cell(r['p_ipw'], r['note_ipw'])} | {_fmt_cell(r['p_dr'], r['note_dr'])} |")

print('\\n#### Plain-language summary (per row)\\n')
print('These are treatment-effect estimates from observational cohorts (eICU-CRD, PMAP, '
      'MIMIC-IV) plus the randomized HYPERION trial -- not causal claims for the '
      'observational cohorts.\\n')
for _, r in results_df[results_df['status'] == 'ok'].iterrows():
    disp = DISPLAY_NAME.get(r['dataset'], r['dataset'])
    print(f"{disp} / {r['outcome']}: "
          f"unadjusted p={_fmt_cell(r['p_unadj'], r['note_unadj'])}, "
          f"adjusted p={_fmt_cell(r['p_adj'], r['note_adj'])}, "
          f"IPW p={_fmt_cell(r['p_ipw'], r['note_ipw'])}, "
          f"DR p={_fmt_cell(r['p_dr'], r['note_dr'])}")
""")

# ============================================================================================ #
# Cell 10 — sanity check against published Table 2
# ============================================================================================ #
code("""# Published Table 2, CausalForestDML, split (70/30) evaluation, unadjusted LRT p-values.
PUBLISHED_SPLIT_CF_P = {
    ('eICU', 'mortality'): 0.15, ('HYPERION', 'mortality'): 0.19,
    ('PMAP', 'mortality'): 0.16, ('MIMIC-IV', 'mortality'): 0.41,
    ('eICU', 'neuro'): 0.10, ('HYPERION', 'neuro'): 0.20,
    ('PMAP', 'neuro'): 0.15, ('MIMIC-IV', 'neuro'): 0.63,
}
# Not a hard-fail check: CausalForestDML is seed-sensitive, so a mismatch prints a loud WARN
# rather than raising -- it flags drift for the authors to look at, it does not mean this run
# is wrong. 0.10 absolute difference in p is used here only as a 'worth a second look' cutoff.
WARN_TOL = 0.10

print(f'=== Sanity check: {EVAL_METHOD} + CausalForestDML unadjusted p vs published Table 2 ===')
sub = results_df[results_df['status'] == 'ok']
for _, r in sub.iterrows():
    key = (r['dataset'], r['outcome'])
    if key not in PUBLISHED_SPLIT_CF_P:
        continue
    pub = PUBLISHED_SPLIT_CF_P[key]
    new = r['p_unadj']
    if pd.isna(new):
        print(f"  {r['dataset']:10s} {r['outcome']:9s}: published p={pub:.2f}, this run "
              f"p=NaN ({r['note_unadj']})")
        print(f"[ADJINT][WARN] unadjusted p does not reproduce Table 2 for "
              f"{r['dataset']}/{r['outcome']} (CATE constant in this run)")
        continue
    diff = abs(new - pub)
    print(f"  {r['dataset']:10s} {r['outcome']:9s}: published p={pub:.2f}, this run "
          f"p={new:.2f}, |diff|={diff:.2f}")
    if diff > WARN_TOL:
        print(f"[ADJINT][WARN] unadjusted p does not reproduce Table 2 for "
              f"{r['dataset']}/{r['outcome']} (|diff|={diff:.2f} > {WARN_TOL})")
""")

# ============================================================================================ #
def build():
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    for kind, src in CELLS:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "markdown"
                        else nbf.v4.new_code_cell(src))
    nbf.write(nb, OUT)
    if SECONDARY_OUT:
        nbf.write(nb, SECONDARY_OUT)
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    if SECONDARY_OUT:
        print(f"wrote {SECONDARY_OUT} ({len(CELLS)} cells)")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adjustedInteractionTest.ipynb")
SECONDARY_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "summarized_results", "adjustedInteractionTest.ipynb")

if __name__ == "__main__":
    build()
