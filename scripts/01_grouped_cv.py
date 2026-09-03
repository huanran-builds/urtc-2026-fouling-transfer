"""
Study-level batch effects in a literature-curated antibacterial ML dataset.

Data: Chen et al. 2026, Cell Reports Physical Science 7, 103411.
      342 nanoparticle MIC records curated from 65 source papers.
      github.com/YaxiiC/Nanoparticle-Antibacterial-Dataset

The published analysis reports 0.79 +/- 0.02 accuracy (XGBoost) from a
stratified random 80/20 split. Because rows are grouped by source paper,
a random split places rows from the same paper in both train and test.

This script evaluates the same model class three ways:

  1. random             StratifiedKFold, source paper ignored (their approach)
  2. grouped            GroupKFold on Ref, no paper spans train and test
  3. grouped+corrected  same, after within-study standardisation of numeric
                        features (a crude batch correction)

MIC_class uses external clinical breakpoints, not thresholds fitted to this
data (paper p.13): strong MIC <= 10, moderate 10 < MIC <= 100, weak MIC > 100.

Run from the repo root:
    python scripts/01_grouped_cv.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

RANDOM_STATE = 42
N_SPLITS = 5
PATH = "data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv"
OUT = "outputs/tables/grouped_cv_mic.csv"

# ======================================================================
# Load
# ======================================================================
df = pd.read_csv(PATH)

# Ref is a DOI written once at the top of each source paper's block and left
# blank underneath. Forward-fill turns it into a usable study identifier.
df["Ref"] = df["Ref"].ffill()

# The column labelled "zeta potential" holds only 0.0 and 1.0. Table 1 of the
# paper confirms zeta was "categorized as positive or negative", so this is a
# binary charge indicator, not a millivolt measurement.
df = df.rename(columns={"zeta potential": "zeta_binary"})

# ======================================================================
# Features
# ======================================================================
NUM = [c for c in df.columns if c.startswith("MagpieData")] + [
    "size (nm)",
    "zeta_binary",
    "duration",
    "temperature",
]

CAT = [
    "Shape",
    "bacteria",
    "gram stain",
    "motility",
    "Oxygen Requirement",
    "Shape.1",
    "Arrangement",
]

missing = [c for c in NUM + CAT if c not in df.columns]
if missing:
    raise SystemExit(f"Columns not found in the CSV: {missing}")


# ======================================================================
# Crude batch correction
# ======================================================================
def within_study_zscore(frame, cols, group_col="Ref"):
    """Center and scale each numeric feature within its source paper, so
    paper-level offsets are removed before modeling.

    Caveat to state in any write-up: this uses each paper's own mean, which
    would not be available for a genuinely new paper at prediction time. It
    measures how much of the cross-paper gap is explained by additive and
    scale offsets, not a deployable correction.
    """
    out = frame.copy()
    for c in cols:
        grp = out.groupby(group_col)[c]
        mu = grp.transform("mean")
        sd = grp.transform("std")
        sd = sd.where(sd > 0, 1.0)   # no within-paper variance -> center only
        out[c] = (out[c] - mu) / sd
    return out


X_raw = df[NUM + CAT]
X_corr = within_study_zscore(df, NUM)[NUM + CAT]

le = LabelEncoder()
y = pd.Series(le.fit_transform(df["MIC_class"]), index=df.index)
groups = df["Ref"]


# ======================================================================
# Pipeline
# ======================================================================
def build_pipeline():
    """Imputation and encoding sit inside the pipeline so they are fit on
    training folds only. Doing either beforehand would leak test-fold
    information through the median and the category list."""
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median"), NUM),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "imp",
                                        SimpleImputer(
                                            strategy="constant",
                                            fill_value="missing",
                                        ),
                                    ),
                                    (
                                        "oh",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            min_frequency=3,
                                        ),
                                    ),
                                ]
                            ),
                            CAT,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    num_class=3,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


SCORING = ["accuracy", "f1_macro"]

# ======================================================================
# Three evaluations
# ======================================================================
random_cv = cross_validate(
    build_pipeline(),
    X_raw,
    y,
    scoring=SCORING,
    cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
)

grouped_cv = cross_validate(
    build_pipeline(),
    X_raw,
    y,
    groups=groups,
    scoring=SCORING,
    cv=GroupKFold(N_SPLITS),
)

grouped_corr = cross_validate(
    build_pipeline(),
    X_corr,
    y,
    groups=groups,
    scoring=SCORING,
    cv=GroupKFold(N_SPLITS),
)


def grouped_train_majority_scores(target, paper_groups):
    """Score the majority class learned separately in each grouped training fold.

    Unlike the overall class prevalence, this baseline never uses labels from
    the held-out papers to choose its predicted class.  It mirrors the
    fold-trained majority calculation in ``scripts/04_mbc.py``.
    """
    scores = []
    coverage = np.zeros(len(target), dtype=int)
    for fold, (train_index, test_index) in enumerate(
        GroupKFold(N_SPLITS).split(X_raw, target, paper_groups), start=1
    ):
        train_papers = set(paper_groups.iloc[train_index])
        test_papers = set(paper_groups.iloc[test_index])
        assert not train_papers & test_papers, f"Fold {fold} has paper overlap"

        train_majority = target.iloc[train_index].value_counts().idxmax()
        scores.append((target.iloc[test_index] == train_majority).mean())
        coverage[test_index] += 1

    assert np.all(coverage == 1), "Each row must be tested exactly once"
    return np.asarray(scores, dtype=float)

# ======================================================================
# Report
# ======================================================================
overall_majority_baseline = df["MIC_class"].value_counts(normalize=True).max()
grouped_cv_majority = grouped_train_majority_scores(y, groups)
vc = groups.value_counts()

print()
print("Chen et al. 2026, nanoparticle MIC dataset")
print(f"  rows              : {len(df)}")
print(f"  source papers     : {groups.nunique()}")
print(f"  rows per paper    : mean {vc.mean():.1f}, median {vc.median():.0f}, max {vc.max()}")
print(f"  classes           : {dict(df['MIC_class'].value_counts())}")
print(f"  overall majority prevalence : {overall_majority_baseline:.3f}")
print(
    "  grouped train-majority baseline: "
    f"{grouped_cv_majority.mean():.3f} +/- {grouped_cv_majority.std():.3f}"
)
print("  paper reports     : 0.79 +/- 0.02 accuracy (XGBoost, random 80/20)")
print()

rows = []
for metric in SCORING:
    r = random_cv[f"test_{metric}"]
    g = grouped_cv[f"test_{metric}"]
    c = grouped_corr[f"test_{metric}"]
    print(f"  {metric}")
    print(f"    random              {r.mean():.3f} +/- {r.std():.3f}")
    print(f"    grouped             {g.mean():.3f} +/- {g.std():.3f}")
    print(f"    grouped + corrected {c.mean():.3f} +/- {c.std():.3f}")
    print(f"    leakage drop        {r.mean() - g.mean():+.3f}")
    print(f"    correction gain     {c.mean() - g.mean():+.3f}")
    print()
    rows.append(
        {
            "metric": metric,
            "random_mean": round(r.mean(), 4),
            "random_std": round(r.std(), 4),
            "grouped_mean": round(g.mean(), 4),
            "grouped_std": round(g.std(), 4),
            "grouped_corrected_mean": round(c.mean(), 4),
            "grouped_corrected_std": round(c.std(), 4),
            "leakage_drop": round(r.mean() - g.mean(), 4),
            "correction_gain": round(c.mean() - g.mean(), 4),
            # Keep the historical name as the fold-trained, deployable
            # comparator used by downstream result and figure scripts.
            "majority_baseline": round(grouped_cv_majority.mean(), 4),
            "cv_majority_accuracy_mean": round(grouped_cv_majority.mean(), 4),
            "cv_majority_accuracy_std": round(grouped_cv_majority.std(), 4),
            "overall_majority_baseline": round(overall_majority_baseline, 4),
            "accuracy_delta_over_cv_majority": round(
                g.mean() - grouped_cv_majority.mean(), 4
            ),
            "corrected_accuracy_delta_over_cv_majority": round(
                c.mean() - grouped_cv_majority.mean(), 4
            ),
            "n_rows": len(df),
            "n_papers": int(groups.nunique()),
            "model": "XGBClassifier",
        }
    )

print("  random              StratifiedKFold, source paper ignored")
print("  grouped             GroupKFold on Ref")
print("  grouped + corrected GroupKFold on Ref, numeric features z-scored within paper")
print()

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Saved to {OUT}")
