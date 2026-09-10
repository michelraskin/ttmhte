"""Shared, importable CATE x TTM interaction tests (CCM R2 revision, Reviewer #1's remaining
request: "add another version of the interaction test where you adjust for main confounders or
use IPW").

This module is the single source of truth for `pooled/adjustedInteractionTest.ipynb`. It holds
the statistical-test layer only (no notebook/data-loading code, which stays in the builder so
the notebook keeps the paper's exact cohorts/preprocessing).

`lrt_cate_interaction` reproduces the published Table 2 test bit-for-bit (see
`pooled/_build_per_dataset_analysis.py`). The other functions add, on top of it: confounder
adjustment (`adjusted_lrt_cate_interaction`), IPW weighting (`ipw_interaction_test`), both
together (`dr_interaction_test`), multiplicity control (`bh_fdr`), and a run manifest
(`init_manifest`/`append_manifest`).

All functions are pure over numpy/pandas inputs: they take already-preprocessed, out-of-sample
arrays/frames and return a plain dict of results. They do not fit propensity/outcome models
themselves (the caller passes `ps`, `cate`, `Z` computed upstream) and they do not know about
cross-fitting -- the caller is responsible for making sure `cate`/`ps`/`Z`/`y`/`T` are aligned,
out-of-sample, one row per patient.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


def _design(T, cate):
    df = pd.DataFrame({'const': 1.0, 'T': np.asarray(T, float), 'cate': np.asarray(cate, float)})
    df['tx'] = df['T'] * df['cate']
    return df


def _or_ci(coef, se):
    return float(np.exp(coef)), float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se))


def _drop_constant_cols(Z):
    Z = pd.DataFrame(Z).reset_index(drop=True)
    dropped = [c for c in Z.columns if np.ptp(Z[c].astype(float).values) < 1e-12]
    return Z.drop(columns=dropped), dropped


def lrt_cate_interaction(y, T, cate):
    """Likelihood-ratio test for a CATE x TTM interaction on the raw outcome.

    Reproduces, bit-for-bit, the published Table 2 test: fit
    reduced = Logit(y ~ const + T + cate) and full = Logit(y ~ const + T + cate + T*cate);
    lr = 2*(full.llf - reduced.llf) ~ chi2(1). No confounder adjustment, no weighting -- this
    is the primary (unadjusted) interaction test the paper already publishes.

    Returns p, lr_stat, note, and the interaction odds ratio with its 95% CI (or/ci_low/ci_high).
    If `cate` is constant (the model found no heterogeneity, i.e. TTM was unused by the CATE
    model), the interaction is not estimable: returns NaN p/or/ci and
    note='CATE constant (TTM unused)' rather than raising -- this is what Table 2 prints as
    "No effect".
    """
    y = np.asarray(y, float)
    df = _design(T, cate)
    if np.ptp(df['cate'].values) < 1e-12:
        return {'lr_stat': np.nan, 'p': np.nan, 'note': 'CATE constant (TTM unused)',
                'or': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
    m0 = sm.Logit(y, df[['const', 'T', 'cate']]).fit(disp=False)
    m1 = sm.Logit(y, df[['const', 'T', 'cate', 'tx']]).fit(disp=False)
    lr = 2 * (m1.llf - m0.llf)
    orr, lo, hi = _or_ci(m1.params['tx'], m1.bse['tx'])
    return {'lr_stat': lr, 'p': chi2.sf(lr, 1), 'note': '', 'or': orr, 'ci_low': lo, 'ci_high': hi}


def adjusted_lrt_cate_interaction(y, T, cate, Z):
    """Confounder-adjusted LRT for the CATE x TTM interaction.

    `Z` is a DataFrame of prespecified confounders, already preprocessed (scaled/imputed) by
    the caller on the same evaluation rows as `y`/`T`/`cate`. Fits
    reduced = Logit(y ~ const + T + cate + Z) and full = reduced + T*cate; LRT on the single
    added term, chi2.sf(lr, 1). Any `Z` column that is constant on these evaluation rows is
    dropped before fitting (it is not identified) and reported in `dropped_confounders`.

    Returns the same keys as `lrt_cate_interaction` plus `n_confounders`, `confounders` (the
    columns actually used) and `dropped_confounders`. Same constant-CATE guard.
    """
    y = np.asarray(y, float)
    Zk, dropped = _drop_constant_cols(Z)
    kept = list(Zk.columns)
    df = _design(T, cate)
    if np.ptp(df['cate'].values) < 1e-12:
        return {'lr_stat': np.nan, 'p': np.nan, 'note': 'CATE constant (TTM unused)',
                'or': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_confounders': len(kept), 'confounders': kept, 'dropped_confounders': dropped}
    reduced = pd.concat([df[['const', 'T', 'cate']], Zk], axis=1)
    full = pd.concat([reduced, df[['tx']]], axis=1)
    m0 = sm.Logit(y, reduced).fit(disp=False)
    m1 = sm.Logit(y, full).fit(disp=False)
    lr = 2 * (m1.llf - m0.llf)
    orr, lo, hi = _or_ci(m1.params['tx'], m1.bse['tx'])
    return {'lr_stat': lr, 'p': chi2.sf(lr, 1), 'note': '', 'or': orr, 'ci_low': lo, 'ci_high': hi,
            'n_confounders': len(kept), 'confounders': kept, 'dropped_confounders': dropped}


def _ipw_weights(T, ps, clip, stabilized):
    T = np.asarray(T, float)
    ps = np.clip(np.asarray(ps, float), *clip)
    w = np.where(T == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    if stabilized:
        p1 = float(np.mean(T == 1))
        w = np.where(T == 1, w * p1, w * (1.0 - p1))
    return w


def ipw_interaction_test(y, T, cate, ps, clip=(0.05, 0.95), stabilized=True):
    """IPW-weighted CATE x TTM interaction test (observational cohorts).

    Weights come from the propensity `ps`, clipped to `clip`. Unstabilized
    w = T/ps + (1-T)/(1-ps); if `stabilized` (default), the treated arm is additionally
    multiplied by P(T=1) and the control arm by P(T=0). Fits
    sm.GLM(y, [const,T,cate,tx], family=Binomial(), freq_weights=w) and Wald-tests `tx` with
    cov_type='HC1'. `freq_weights` treats the weights as replication counts, so the HC1
    sandwich SE is what makes this inference honest (the naive fit below understates
    uncertainty).

    Returns wald_p, or, ci_low, ci_high, ess (effective sample size, sum(w)**2/sum(w**2)),
    max_weight, and wald_p_naive (same design fit with var_weights=w and no robust cov --
    kept for continuity with the already-published eTable 21). Same constant-CATE guard.
    """
    df = _design(T, cate)
    y = np.asarray(y, float)
    nan_out = {'wald_p': np.nan, 'wald_p_naive': np.nan, 'or': np.nan, 'ci_low': np.nan,
               'ci_high': np.nan, 'ess': np.nan, 'max_weight': np.nan}
    if np.ptp(df['cate'].values) < 1e-12:
        return {**nan_out, 'note': 'CATE constant (TTM unused)'}
    w = _ipw_weights(df['T'].values, ps, clip, stabilized)
    fam = sm.families.Binomial()
    m1 = sm.GLM(y, df[['const', 'T', 'cate', 'tx']], family=fam, freq_weights=w).fit(cov_type='HC1')
    m1_naive = sm.GLM(y, df[['const', 'T', 'cate', 'tx']], family=fam, var_weights=w).fit()
    orr, lo, hi = _or_ci(m1.params['tx'], m1.bse['tx'])
    ess = float((w.sum() ** 2) / np.sum(w ** 2))
    return {'wald_p': float(m1.pvalues['tx']), 'wald_p_naive': float(m1_naive.pvalues['tx']),
            'or': orr, 'ci_low': lo, 'ci_high': hi, 'ess': ess, 'max_weight': float(w.max()),
            'note': ''}


def dr_interaction_test(y, T, cate, ps, Z, clip=(0.05, 0.95), stabilized=True):
    """IPW-weighted AND confounder-adjusted CATE x TTM interaction test (observational cohorts).

    Same weighting as `ipw_interaction_test`, with the confounders `Z` also included as main
    effects in the GLM design -- weighted and adjusted together, so it stays consistent if
    either the propensity model or the outcome-confounder relationship is roughly right.
    Same return keys as `ipw_interaction_test`, plus `n_confounders`/`confounders`/
    `dropped_confounders` as in `adjusted_lrt_cate_interaction`. Same constant-CATE guard.
    """
    df = _design(T, cate)
    y = np.asarray(y, float)
    Zk, dropped = _drop_constant_cols(Z)
    kept = list(Zk.columns)
    nan_out = {'wald_p': np.nan, 'wald_p_naive': np.nan, 'or': np.nan, 'ci_low': np.nan,
               'ci_high': np.nan, 'ess': np.nan, 'max_weight': np.nan,
               'n_confounders': len(kept), 'confounders': kept, 'dropped_confounders': dropped}
    if np.ptp(df['cate'].values) < 1e-12:
        return {**nan_out, 'note': 'CATE constant (TTM unused)'}
    w = _ipw_weights(df['T'].values, ps, clip, stabilized)
    design = pd.concat([df[['const', 'T', 'cate']], Zk, df[['tx']]], axis=1)
    fam = sm.families.Binomial()
    m1 = sm.GLM(y, design, family=fam, freq_weights=w).fit(cov_type='HC1')
    m1_naive = sm.GLM(y, design, family=fam, var_weights=w).fit()
    orr, lo, hi = _or_ci(m1.params['tx'], m1.bse['tx'])
    ess = float((w.sum() ** 2) / np.sum(w ** 2))
    return {'wald_p': float(m1.pvalues['tx']), 'wald_p_naive': float(m1_naive.pvalues['tx']),
            'or': orr, 'ci_low': lo, 'ci_high': hi, 'ess': ess, 'max_weight': float(w.max()),
            'note': '', 'n_confounders': len(kept), 'confounders': kept,
            'dropped_confounders': dropped}


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values.

    NaN-safe: NaN entries pass through as NaN in the output and are excluded from the ranking
    (they do not consume a rank slot and do not affect the adjustment of the non-NaN entries).
    Returns a numpy array the same length/order as `pvals`.
    """
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1)
    q = ranked * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[ok[order]] = q
    return out


def init_manifest(path, meta):
    """Write the manifest header line (run metadata dict) to `path`, creating/overwriting it.

    Call once at the start of a run; `meta` should hold run-level info (seed, package
    versions, eval methods, models, datasets, outcomes, start time).
    """
    with open(path, 'w') as f:
        f.write(json.dumps(meta, default=str) + '\n')
        f.flush()
        os.fsync(f.fileno())


def append_manifest(path, row):
    """Append one manifest row (JSON Lines), flushed and fsync'd so a killed process never
    leaves a partially-written line and every completed fit is durably recorded.

    `row` should record, per fit: seed, splitter/eval method, dataset, outcome, n, n events,
    features, hyperparameters, and package versions.
    """
    line = json.dumps(row, default=str) + '\n'
    with open(path, 'a') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
