"""Pre-specified grouped-CV ablation of composition-derived Magpie features.

This repeats the MIC grouped baseline exactly, except that the 22 columns whose
names begin with ``MagpieData`` are excluded.  All remaining feature columns,
preprocessing, XGBoost settings, GroupKFold splits, and metrics are unchanged.

Run from the repository root:
    python scripts/09_magpie_ablation.py
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SPLITS = 5
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv"
OUTPUT_DIR = ROOT / "outputs/tables"
SUMMARY_OUT = OUTPUT_DIR / "magpie_ablation_summary.csv"
FOLDS_OUT = OUTPUT_DIR / "magpie_ablation_fold_metrics.csv"
OOF_OUT = OUTPUT_DIR / "magpie_ablation_oof_predictions.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
            ]), categorical),
        ])),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, objective="multi:softprob", num_class=3,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
        )),
    ])


def main() -> None:
    frame = pd.read_csv(INPUT).rename(columns={"zeta potential": "zeta_binary"})
    frame["Ref"] = frame["Ref"].ffill()
    magpie = [c for c in frame if c.startswith("MagpieData")]
    numeric = ["size (nm)", "zeta_binary", "duration", "temperature"]
    categorical = ["Shape", "bacteria", "gram stain", "motility", "Oxygen Requirement", "Shape.1", "Arrangement"]
    assert len(magpie) == 22, f"Expected 22 Magpie columns, found {len(magpie)}"
    assert all(c in frame for c in numeric + categorical)
    labels = LabelEncoder().fit(frame["MIC_class"])
    y = pd.Series(labels.transform(frame["MIC_class"]), index=frame.index)
    groups = frame["Ref"].astype(str)
    X_full = frame[magpie + numeric + categorical]
    X_ablated = frame[numeric + categorical]
    assert len(frame) == 342 and groups.nunique() == 65 and set(y.unique()) == {0, 1, 2}

    fold_rows, oof_rows = [], []
    coverage = {"full_features": np.zeros(len(frame), int), "without_magpie": np.zeros(len(frame), int)}
    for fold, (train, test) in enumerate(GroupKFold(N_SPLITS).split(X_full, y, groups), 1):
        train_papers, test_papers = set(groups.iloc[train]), set(groups.iloc[test])
        assert not train_papers & test_papers, f"Fold {fold} has paper overlap"
        for name, X, nums in (("full_features", X_full, magpie + numeric), ("without_magpie", X_ablated, numeric)):
            model = pipeline(nums, categorical)
            model.fit(X.iloc[train], y.iloc[train])
            pred = model.predict(X.iloc[test]).astype(int)
            coverage[name][test] += 1
            fold_rows.append({
                "evaluation": name, "fold": fold, "n_train_rows": len(train), "n_test_rows": len(test),
                "n_train_papers": len(train_papers), "n_test_papers": len(test_papers),
                "n_shared_papers": len(train_papers & test_papers), "accuracy": accuracy_score(y.iloc[test], pred),
                "macro_f1": f1_score(y.iloc[test], pred, average="macro", zero_division=0),
            })
            oof_rows.extend({"evaluation": name, "fold": fold, "row_index": int(i), "Ref": groups.iloc[i],
                             "actual_class": labels.classes_[y.iloc[i]], "predicted_class": labels.classes_[p],
                             "correct": bool(y.iloc[i] == p)} for i, p in zip(test, pred))

    folds, oof = pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)
    assert folds.groupby("evaluation").size().eq(N_SPLITS).all()
    assert folds["n_shared_papers"].eq(0).all() and all(np.all(x == 1) for x in coverage.values())
    baseline = frame["MIC_class"].value_counts(normalize=True).max()
    summary = folds.groupby("evaluation", as_index=False).agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"))
    summary["majority_baseline"] = baseline
    summary["accuracy_minus_majority_baseline"] = summary["accuracy_mean"] - baseline
    summary["n_rows"] = len(frame); summary["n_papers"] = groups.nunique(); summary["n_magpie_features_removed"] = 22
    summary["input_sha256"] = sha256(INPUT); summary["python_version"] = platform.python_version(); summary["sklearn_version"] = sklearn.__version__; summary["xgboost_version"] = xgboost.__version__
    assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()
    folds.to_csv(FOLDS_OUT, index=False); oof.to_csv(OOF_OUT, index=False); summary.to_csv(SUMMARY_OUT, index=False)
    print(summary[["evaluation", "accuracy_mean", "macro_f1_mean", "accuracy_minus_majority_baseline"]].to_string(index=False))


if __name__ == "__main__":
    main()
