"""Canonical analytic cohort — single source of truth for every table in the paper.

Every descriptive artifact (Table 1, eFigure 6 attrition, the missingness eTables 16-19)
MUST derive its cohort and denominators from the loaders here, so that all reported N's
agree with the numbers actually analyzed in `pooledObservationalAnalysis.ipynb`.

Cohort definition = the **pooled neurologic-outcome strategy** (the loaders in
`_build_pooled_analysis.py`):

  eICU-CRD  : scorable last motor GCS, distinct first/last mGCS times, nurse_first_Motor != 6,
              non-missing TTM.                                           -> n = 1842
  PMAP      : first_mGCS != 6, non-missing TTM (rows with first==last mGCS time are KEPT;
              their neurologic outcome is undefined/NaN, defined only for mortality). -> n = 1412
  MIMIC-IV  : same epic-style rule as PMAP.                              -> n =  611
  HYPERION  : randomized arms only (groupe != 2); TTM = (groupe == 1).   -> n =  581

Each loader returns the FULL cohort dataframe (all original columns retained for descriptor
computation) plus four canonical columns: ``TTM``, ``mortality``, ``neuro_favorable``,
``dataset``. The neurologic outcome is NaN where undefined (mortality still valid), exactly
as in the pooled analysis. Do not apply ``select_dtypes`` here — descriptors need the object
columns (race, rhythm, arrest location).

Usage from a notebook running in ``summarized_results/``::

    import sys; sys.path.insert(0, str(ROOT / "pooled"))
    from canonical_cohort import load_all_cohorts, CANONICAL_N
    cohorts = load_all_cohorts(ROOT)          # {'eICU': df, 'PMAP': df, 'MIMIC-IV': df, 'HYPERION': df}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

UNSCORABLE = "Unable to score due to medication"

# The predictor CSVs the *AnalysisDML notebooks and the pooled analysis read.
REL_PATHS = {
    "eICU": "eICU/eICUPredictorsDiag.csv",
    "PMAP": "pmap/PMAP_Predictors2.csv",
    "MIMIC-IV": "mimiciv/MIMIC_Predictors.csv",
    "HYPERION": "hyperion/predictorsDf.csv",
}

# Expected analytic N — assert against these so a wrong CSV or a silent cohort change is caught.
CANONICAL_N = {"eICU": 1842, "PMAP": 1412, "MIMIC-IV": 611, "HYPERION": 581}


def _resolve_root(root=None) -> Path:
    if root is not None:
        return Path(root)
    here = Path.cwd()
    for cand in (here, here.parent, here.parent.parent):
        if (cand / "eICU").exists() and (cand / "pooled").exists():
            return cand
    return here


def _csv(root: Path, name: str) -> Path:
    env = os.environ.get(f"{name.replace('-', '').upper()}_PREDICTORS_CSV")
    return Path(env) if env else (root / REL_PATHS[name])


def load_eicu_cohort(root=None) -> pd.DataFrame:
    df = pd.read_csv(_csv(_resolve_root(root), "eICU"))
    f = (df["LastMGCS"] != UNSCORABLE) & (~df["LastMGCS"].isna())
    f = f & (df["FirstMGCSTime"] != df["LastMGCSTime"])
    for c in ["FirstGCS", "FirstMGCS", "LastMGCS", "LastGCS"]:
        if c in df.columns:
            df.loc[df[c] == UNSCORABLE, c] = np.nan
    df.loc[df["DeathAtDischarge"] == 1, "LastMGCS"] = 1
    df["male"] = (df["gender"] == "Male").astype(int)
    df.loc[f, "LastMGCSPositive"] = (df.loc[f, "LastMGCS"].astype(float) == 6).astype(int)
    df = df[f & (df["nurse_first_Motor"] != 6) & ~df["Hypothermia"].isna()].copy()
    df["TTM"] = df["Hypothermia"].astype(int)
    df["mortality"] = df["DeathAtDischarge"].astype(int)
    df["neuro_favorable"] = df["LastMGCSPositive"]
    df["dataset"] = "eICU"
    return df.reset_index(drop=True)


def _load_epic_cohort(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    f = df["first_mGCS_time"] != df["last_mGCS_time"]
    df.loc[df["death_at_disch"] == 1, "last_mGCS"] = 1
    df.loc[f, "LastMGCSPositive"] = (df.loc[f, "last_mGCS"].astype(float) == 6).astype(int)
    df = df[(df["first_mGCS"] != 6) & ~df["hypothermia"].isna()].copy()
    df["TTM"] = df["hypothermia"].astype(int)
    df["mortality"] = df["death_at_disch"].astype(int)
    df["neuro_favorable"] = df["LastMGCSPositive"]  # NaN where first==last mGCS time
    df["dataset"] = dataset
    return df.reset_index(drop=True)


def load_pmap_cohort(root=None) -> pd.DataFrame:
    return _load_epic_cohort(_csv(_resolve_root(root), "PMAP"), "PMAP")


def load_mimic_cohort(root=None) -> pd.DataFrame:
    return _load_epic_cohort(_csv(_resolve_root(root), "MIMIC-IV"), "MIMIC-IV")


def load_hyperion_cohort(root=None) -> pd.DataFrame:
    df = pd.read_csv(_csv(_resolve_root(root), "HYPERION"))
    df = df[df["groupe"] != 2].copy()
    df["TTM"] = (df["groupe"] == 1).astype(int)
    df["mortality"] = df["hospital_mortality"] if "hospital_mortality" in df.columns else np.nan
    df["neuro_favorable"] = df["CPC12"] if "CPC12" in df.columns else np.nan
    df["dataset"] = "HYPERION"
    return df.reset_index(drop=True)


_LOADERS = {
    "eICU": load_eicu_cohort,
    "PMAP": load_pmap_cohort,
    "MIMIC-IV": load_mimic_cohort,
    "HYPERION": load_hyperion_cohort,
}


def load_all_cohorts(root=None, strict: bool = True) -> dict[str, pd.DataFrame]:
    """Load every cohort. With ``strict`` (default), assert each N matches CANONICAL_N."""
    root = _resolve_root(root)
    out = {}
    for name, loader in _LOADERS.items():
        try:
            df = loader(root)
        except FileNotFoundError:
            print(f"{name}: predictor CSV not found ({REL_PATHS[name]}); skipping.")
            continue
        n = len(df)
        flag = "" if n == CANONICAL_N[name] else f"  !! expected {CANONICAL_N[name]}"
        print(f"{name:9s} n={n:5d}  TTM={int(df['TTM'].sum()):4d} ({df['TTM'].mean():.1%})  "
              f"mortality={df['mortality'].mean():.1%}  "
              f"neuro defined={df['neuro_favorable'].notna().mean():.1%}{flag}")
        if strict and n != CANONICAL_N[name] and os.environ.get("ALLOW_COHORT_N_MISMATCH", "0") not in {"1", "true", "yes"}:
            raise AssertionError(
                f"{name}: cohort N={n} != canonical {CANONICAL_N[name]}. "
                f"Set ALLOW_COHORT_N_MISMATCH=1 to override after confirming the CSV/pipeline."
            )
        out[name] = df
    return out
