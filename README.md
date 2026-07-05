# ttmhte — Heterogeneous Treatment Effect of Targeted Temperature Management After Cardiac Arrest

Code accompanying the study:

> **Heterogeneous Treatment Effect for Targeted Temperature Management After Cardiac Arrest: A Causal Machine Learning Analysis.**
> Raskin MB, Karhu-Leperd I, Harris C, Pirracchio R, Lascarrou J-B, Stevens RD.
> *Critical Care Medicine* (under revision; CCMED-D-26-00748).

We applied causal machine learning across one randomized trial (HYPERION) and three
observational ICU datasets (eICU-CRD, MIMIC-IV, and the Johns Hopkins PMAP export) to test
whether heterogeneous treatment effects (HTE) explain the inconclusive results of targeted
temperature management (TTM) trials after cardiac arrest. Across S-learners (XGBoost, neural
network, BART) and a forest-based R-learner (`CausalForestDML`), we found no reproducible
evidence of HTE on hospital mortality or favorable neurologic outcome.

## How to cite

If you use this code, please cite the paper (above) and, optionally, the software via the
`CITATION.cff` file (GitHub's "Cite this repository" button), e.g.:

```
Raskin MB, Karhu-Leperd I, Harris C, Pirracchio R, Lascarrou J-B, Stevens RD.
ttmhte: Causal machine-learning analysis of heterogeneous treatment effects of targeted
temperature management after cardiac arrest. GitHub; 2026. https://github.com/michelraskin/ttmhte
```

## Repository structure

```
eICU/               eICU-CRD cohort build + analyses  (eICUUtil.py, *Analysis*.ipynb)
mimiciv/            MIMIC-IV cohort build + analyses   (MIMICUtil.py, *Analysis*.ipynb)
pmap/               PMAP (JH Epic Clarity) build + analyses
hyperion/           HYPERION RCT preprocessing + analyses
pooled/             Builder scripts for the pooled + per-dataset sensitivity analyses
                    _build_pooled_analysis.py        -> pooledObservationalAnalysis.ipynb
                    _build_per_dataset_analysis.py   -> perDatasetSensitivity.ipynb
summarized_results/ Executed sensitivity notebooks + result CSVs / figures
deps.txt            Frozen Python environment
```

Workflow: edit a builder in `pooled/`, regenerate the notebook, and run it from
`summarized_results/` (where the executed notebooks and outputs live).

Per dataset, `*Analysis*.ipynb` notebooks fit the S-learners (`Classif`, `Neural`, `BART`) and
the forest R-learner (`DML`), and assess HTE by (i) a likelihood-ratio CATE×TTM interaction
test, (ii) `CausalForestDML` CATE 95% confidence intervals, (iii) Group Average Treatment
Effects (GATES) across CATE quintiles with inverse-probability weighting, and (iv) SHAP
importance. The sensitivity notebooks add the harmonized pooled analysis and the per-dataset
sensitivity analyses (regularized S-learner, GATES for the neurologic outcome, uniform
collinearity filter). Evaluation uses a single stratified 70/30 train/test split by default
(`EVAL_METHOD = 'split'`, consistent with the main analysis); a k-fold cross-fitting option
(`EVAL_METHOD = 'crossfit'`) is also provided in the builders.

## Reproducibility

**Environment.** Python 3.11; exact package versions are pinned in [`deps.txt`](deps.txt)
(key: scikit-learn 1.5.0, XGBoost 3.1.2, econml 0.16.0, statsmodels 0.14.4, PyTorch 2.7.1,
TensorFlow 2.20.0 / Keras 3.12.0, pymc 5.27.0 + pymc_bart 0.11.0, SHAP 0.48.0).

```bash
python -m pip install -r deps.txt   # or: pip install python-docx econml xgboost pymc-bart ...
```

**Pipeline builders.** The `pooled/` notebooks are generated from small builder scripts —
edit the builder, then regenerate (do not hand-edit the `.ipynb`):

```bash
python pooled/_build_pooled_analysis.py
python pooled/_build_per_dataset_analysis.py
```

**Run (on the analysis host with the predictor CSVs present):**

```bash
jupyter nbconvert --to notebook --execute summarized_results/pooledObservationalAnalysis.ipynb \
  --output pooledObservationalAnalysis.ipynb --ExecutePreprocessor.timeout=-1
```

Results are written to `summarized_results/pooled_analysis_results.csv` and
`summarized_results/per_dataset_results.csv`; figures are saved as PNGs alongside the notebooks.

## Data availability

The predictor CSVs are **not** included (data-use agreements). To reproduce:

| Dataset | Access |
|---|---|
| eICU-CRD | PhysioNet (credentialed): https://physionet.org/content/eicu-crd/ |
| MIMIC-IV | PhysioNet (credentialed): https://physionet.org/content/mimiciv/ |
| PMAP (Johns Hopkins) | Institutional data-use agreement; not publicly redistributable |
| HYPERION | From the trial investigators (NCT02057835) on reasonable request |

The cohort-identification and feature-extraction notebooks document how each predictor table
is built from the source databases.

## License

Code released under the MIT License (see `LICENSE`). Clinical data are governed by their
respective data-use agreements and are not covered by this license.

## Contact

Corresponding author: Robert D. Stevens (rsteven1@jh.edu). Code: Michel B. Raskin.
