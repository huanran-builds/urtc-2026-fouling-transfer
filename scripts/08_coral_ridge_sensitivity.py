"""Fixed-grid, label-safe sensitivity analysis for CORAL ridge regularization.

This is an exploratory robustness check, not hyperparameter selection. The
five ridge values are fixed before inspecting their held-out metrics and all
five are written to the output. In every case target-paper labels are used
only after prediction for scoring.

Run from the repository root:
    python scripts/08_coral_ridge_sensitivity.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
# Fixed a priori log-scale grid; do not select a winner from outer-fold labels.
RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv"
OUTPUT_DIR = ROOT / "outputs/tables"
SUMMARY_OUT = OUTPUT_DIR / "coral_ridge_sensitivity.csv"
FOLDS_OUT = OUTPUT_DIR / "coral_ridge_sensitivity_fold_metrics.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
                    ]
                ),
                cat,
            ),
        ]
    )


def make_classifier() -> XGBClassifier:
    return XGBClassifier(
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
    )


def covariance(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return centered.T @ centered / len(matrix)


def symmetric_matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return (eigenvectors * np.power(eigenvalues, exponent)) @ eigenvectors.T


def coral_align_source_to_target(
    source: np.ndarray, target: np.ndarray, ridge: float
) -> np.ndarray:
    """Align source covariance to target covariance using features only."""
    if len(target) < 2:
        return source.copy()
    n_features = source.shape[1]
    source_covariance = covariance(source) + ridge * np.eye(n_features)
    target_covariance = covariance(target) + ridge * np.eye(n_features)
    return source @ symmetric_matrix_power(source_covariance, -0.5) @ symmetric_matrix_power(
        target_covariance, 0.5
    )


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame["Ref"] = frame["Ref"].ffill()
    frame = frame.rename(columns={"zeta potential": "zeta_binary"})
    num = [column for column in frame.columns if column.startswith("MagpieData")] + [
        "size (nm)", "zeta_binary", "duration", "temperature"
    ]
    cat = [
        "Shape", "bacteria", "gram stain", "motility", "Oxygen Requirement",
        "Shape.1", "Arrangement",
    ]
    assert all(column in frame for column in num + cat)
    labels = LabelEncoder().fit(frame["MIC_class"])
    target = pd.Series(labels.transform(frame["MIC_class"]), index=frame.index)
    features = frame[num + cat]
    groups = frame["Ref"].astype(str)
    baseline = frame["MIC_class"].value_counts(normalize=True).max()
    assert len(frame) == 342 and groups.nunique() == 65 and set(target.unique()) == {0, 1, 2}

    rows: list[dict[str, object]] = []
    coverage = {ridge: np.zeros(len(frame), dtype=int) for ridge in RIDGES}
    for fold, (train_index, test_index) in enumerate(
        GroupKFold(N_SPLITS).split(features, target, groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        assert not train_groups & test_groups, f"Fold {fold} has paper overlap"
        train_target = target.iloc[train_index].to_numpy()
        test_target = target.iloc[test_index].to_numpy()

        preprocessor = make_preprocessor(num, cat)
        train_encoded = np.asarray(preprocessor.fit_transform(features.iloc[train_index]), dtype=float)
        test_encoded = np.asarray(preprocessor.transform(features.iloc[test_index]), dtype=float)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded)
        test_scaled = scaler.transform(test_encoded)

        for ridge in RIDGES:
            predictions = np.empty(len(test_index), dtype=int)
            for paper in sorted(test_groups):
                paper_positions = np.flatnonzero(groups.iloc[test_index].to_numpy() == paper)
                paper_features = test_scaled[paper_positions]
                # No labels are passed to the alignment transform or training call.
                aligned_train = coral_align_source_to_target(train_scaled, paper_features, ridge)
                model = make_classifier()
                model.fit(aligned_train, train_target)
                predictions[paper_positions] = model.predict(paper_features).astype(int)
            coverage[ridge][test_index] += 1
            rows.append(
                {
                    "ridge": ridge,
                    "fold": fold,
                    "n_train_rows": len(train_index),
                    "n_test_rows": len(test_index),
                    "n_train_papers": len(train_groups),
                    "n_test_papers": len(test_groups),
                    "n_shared_papers": len(train_groups & test_groups),
                    "accuracy": accuracy_score(test_target, predictions),
                    "macro_f1": f1_score(test_target, predictions, average="macro", zero_division=0),
                }
            )
            print(f"completed fold {fold}/{N_SPLITS}, ridge {ridge:g}", flush=True)

    fold_table = pd.DataFrame(rows)
    assert fold_table.groupby("ridge").size().eq(N_SPLITS).all()
    assert fold_table["n_shared_papers"].eq(0).all()
    assert all(np.all(values == 1) for values in coverage.values())
    assert np.isfinite(fold_table.select_dtypes(include=np.number).to_numpy()).all()
    summary = (
        fold_table.groupby("ridge", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda values: values.std(ddof=0)),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda values: values.std(ddof=0)),
            n_rows=("n_test_rows", "sum"),
            n_papers=("n_test_papers", "sum"),
        )
    )
    summary["majority_baseline"] = baseline
    summary["accuracy_minus_majority_baseline"] = summary["accuracy_mean"] - baseline
    summary["input_sha256"] = file_sha256(INPUT)

    # Outputs exist only if every fixed-grid run and leakage check passed.
    fold_table.to_csv(FOLDS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    print("CORAL ridge sensitivity (all rows are exploratory; no selected winner)")
    print(summary[["ridge", "accuracy_mean", "macro_f1_mean", "accuracy_minus_majority_baseline"]].to_string(index=False))
    print(f"saved {SUMMARY_OUT.relative_to(ROOT)} and {FOLDS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
