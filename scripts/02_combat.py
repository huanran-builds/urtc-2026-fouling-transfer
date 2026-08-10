"""
Batch correction for study-level effects in a literature-curated
antibacterial ML dataset.

Data: Chen et al. 2026, Cell Reports Physical Science 7, 103411.
      342 nanoparticle MIC records curated from 65 source papers.

Compares four evaluation settings:

  1. random             StratifiedKFold, source paper ignored (their approach)
  2. grouped, raw       GroupKFold on Ref, no correction
  3. grouped, standardized
                        GroupKFold on Ref, each paper's numeric features
                        centered and scaled by that paper's own statistics.
                        Computed inside the fold, uses no labels, and works
                        on a paper never seen in training. Deployable.
  4. grouped, ComBat    GroupKFold on Ref, empirical-Bayes ComBat fit on the
                        training papers only.

CORE LIMITATION, and the actual research problem:
ComBat estimates location and scale parameters per batch. For a paper that
never appeared in training there are no parameters to apply, so ComBat as
normally used cannot harmonize a genuinely new study. Setting 4 therefore
falls back to per-paper standardization on the held-out fold, rescaled onto
the pooled training scale. Setting 3 is the honest deployable comparison.
Setting 4 tests whether ComBat's shrinkage on the training side adds anything.

Run from the repo root:
    python scripts/02_combat.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5
PATH = "data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv"
OUT = "outputs/tables/combat_comparison.csv"

# ======================================================================
# Load
# ======================================================================
df = pd.read_csv(PATH)
df["Ref"] = df["Ref"].ffill()
df = df.rename(columns={"zeta potential": "zeta_binary"})

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
    raise SystemExit(f"Columns not found: {missing}")

le = LabelEncoder()
y = pd.Series(le.fit_transform(df["MIC_class"]), index=df.index)
groups = df["Ref"]

# Batch label rides along with the features so the transformer can see it
# inside each CV fold. It is dropped before it ever reaches the model.
BATCH = "__batch__"
X_all = df[NUM + CAT].copy()
X_all[BATCH] = groups.values


# ======================================================================
# Transformers
# ======================================================================
class DropBatch(BaseEstimator, TransformerMixin):
    """Remove the batch column before modeling."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[BATCH])


class PerBatchStandardizer(BaseEstimator, TransformerMixin):
    """Center and scale each numeric feature within its own batch.

    Statistics come from the rows being transformed, not from training, so
    this works on a batch that never appeared in the training set. It uses
    no labels, so applying it to test rows is not leakage. A batch with one
    row, or zero variance in a column, is centered but not scaled.
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        for c in self.cols:
            grp = out.groupby(BATCH)[c]
            mu = grp.transform("mean")
            sd = grp.transform("std")
            sd = sd.where(sd > 0, 1.0).fillna(1.0)
            out[c] = (out[c] - mu) / sd
        return out.drop(columns=[BATCH])


class ComBatTrainThenStandardize(BaseEstimator, TransformerMixin):
    """ComBat on the training batches, per-batch standardization elsewhere.

    fit:       run ComBat across the training papers, then record the pooled
               mean and standard deviation of the harmonized training data.
    transform: batches seen during fit get their fitted ComBat parameters.
               Unseen batches are standardized by their own statistics and
               mapped onto the pooled training scale, which is the only
               option when a paper contributed no training rows.

    neurocombat_sklearn casts site labels to float, so DOI strings are
    mapped to integer codes before being passed in.
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        from neurocombat_sklearn import CombatModel

        Xc = X[self.cols].copy()
        self.fallback_median_ = Xc.median()
        Xc = Xc.fillna(self.fallback_median_)

        raw = X[BATCH].astype(str)
        self.site_map_ = {s: i for i, s in enumerate(sorted(raw.unique()))}
        sites = raw.map(self.site_map_).values.reshape(-1, 1).astype(float)

        self.model_ = CombatModel()
        harmonized = self.model_.fit_transform(Xc.values, sites)

        self.pooled_mu_ = harmonized.mean(axis=0)
        sd = harmonized.std(axis=0)
        self.pooled_sd_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, X):
        out = X.copy()
        Xc = out[self.cols].fillna(self.fallback_median_)
        result = Xc.values.astype(float).copy()

        raw = out[BATCH].astype(str)
        mask_seen = raw.isin(self.site_map_).values

        if mask_seen.any():
            codes = (
                raw[mask_seen].map(self.site_map_)
                .values.reshape(-1, 1).astype(float)
            )
            result[mask_seen] = self.model_.transform(Xc.values[mask_seen], codes)

        # Unseen papers: standardize by their own stats, then map onto the
        # pooled training scale.
        if (~mask_seen).any():
            tmp = out.loc[~mask_seen].copy()
            for j, c in enumerate(self.cols):
                grp = tmp.groupby(BATCH)[c]
                mu = grp.transform("mean")
                sd = grp.transform("std")
                sd = sd.where(sd > 0, 1.0).fillna(1.0)
                z = ((tmp[c].fillna(self.fallback_median_[c]) - mu) / sd).values
                result[~mask_seen, j] = z * self.pooled_sd_[j] + self.pooled_mu_[j]

        out[self.cols] = result
        return out.drop(columns=[BATCH])


# ======================================================================
# Pipeline
# ======================================================================
def build_pipeline(correction="none"):
    if correction == "none":
        pre = DropBatch()
    elif correction == "standardize":
        pre = PerBatchStandardizer(NUM)
    elif correction == "combat":
        pre = ComBatTrainThenStandardize(NUM)
    else:
        raise ValueError(correction)

    return Pipeline(
        [
            ("batch", pre),
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median"), NUM),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="constant",
                                                          fill_value="missing")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore",
                                                         min_frequency=3)),
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
# Run
# ======================================================================
settings = [
    ("random, no correction", "none",
     StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE), None),
    ("grouped, no correction", "none", GroupKFold(N_SPLITS), groups),
    ("grouped, standardized", "standardize", GroupKFold(N_SPLITS), groups),
    ("grouped, ComBat", "combat", GroupKFold(N_SPLITS), groups),
]

baseline = df["MIC_class"].value_counts(normalize=True).max()

print()
print("Chen et al. 2026, nanoparticle MIC dataset")
print(f"  {len(df)} rows from {groups.nunique()} source papers")
print(f"  majority baseline: {baseline:.3f}")
print("  paper reports    : 0.79 +/- 0.02 (XGBoost, random 80/20)")
print()

rows = []
for label, corr, cv, grp in settings:
    try:
        res = cross_validate(
            build_pipeline(corr), X_all, y,
            groups=grp, scoring=SCORING, cv=cv, error_score="raise",
        )
    except Exception as e:
        print(f"  {label:<24}  FAILED: {type(e).__name__}: {e}")
        rows.append({"setting": label, "accuracy_mean": None,
                     "note": f"{type(e).__name__}: {e}"})
        continue

    acc = res["test_accuracy"]
    f1 = res["test_f1_macro"]
    print(f"  {label:<24}  acc {acc.mean():.3f} +/- {acc.std():.3f}"
          f"   f1 {f1.mean():.3f} +/- {f1.std():.3f}")
    rows.append({
        "setting": label,
        "accuracy_mean": round(acc.mean(), 4),
        "accuracy_std": round(acc.std(), 4),
        "f1_macro_mean": round(f1.mean(), 4),
        "f1_macro_std": round(f1.std(), 4),
        "majority_baseline": round(baseline, 4),
        "n_rows": len(df),
        "n_papers": int(groups.nunique()),
        "note": "",
    })

print()
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Saved to {OUT}")
