"""Evaluate unlabeled target-paper adaptation with CORAL.

For every GroupKFold split, preprocessing is fitted only on the training
papers. CORAL then uses the *features* of one held-out paper to align the
training feature covariance to that paper's covariance; its MIC labels never
enter preprocessing, CORAL, or model fitting. A separate XGBoost model is fit
for each held-out paper because each paper is its own target domain.

CORAL (CORrelation ALignment) is suitable for ordinary tabular vectors. It
does not make the abundance/compositional assumptions of DEBIAS-M.

Run from the repository root:
    python scripts/05_adaptation.py
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
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
EXPECTED_GROUPED_ACCURACY = 0.3162
REPLICATION_TOLERANCE = 0.02
MAX_ADAPTED_ACCURACY = 0.55
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv"
OUTPUT_DIR = ROOT / "outputs/tables"
SUMMARY_OUT = OUTPUT_DIR / "adaptation_comparison.csv"
FOLDS_OUT = OUTPUT_DIR / "adaptation_fold_metrics.csv"
PREDICTIONS_OUT = OUTPUT_DIR / "adaptation_oof_predictions.csv"


def file_sha256(path: Path) -> str:
    """Return a stable identifier for the raw input used in this run."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Fit all preprocessing on training papers only."""
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
    """The exact classifier settings used by scripts/01_grouped_cv.py."""
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


def within_study_zscore(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Use unlabeled, within-paper feature statistics; singletons center only."""
    out = frame.copy()
    for column in cols:
        grouped = out.groupby("Ref")[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").where(lambda values: values > 0, 1.0)
        out[column] = (out[column] - mean) / std.fillna(1.0)
    return out


def covariance(matrix: np.ndarray) -> np.ndarray:
    """Population covariance, defined as zero when a target paper has one row."""
    if len(matrix) < 2:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return centered.T @ centered / len(matrix)


def symmetric_matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    """Stable power of a symmetric positive-definite covariance matrix."""
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return (eigenvectors * np.power(eigenvalues, exponent)) @ eigenvectors.T


def coral_align_source_to_target(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    ridge: float = 1.0,
) -> np.ndarray:
    """Return source vectors covariance-aligned to an unlabeled target paper.

    This function deliberately accepts features only. It has no target-label
    argument, so labels from the held-out paper cannot influence the CORAL
    transform. With one target row there is no covariance to estimate; the
    identity transform is used rather than inventing a covariance estimate.
    """
    if len(target_features) < 2:
        return source_features.copy()

    n_features = source_features.shape[1]
    source_covariance = covariance(source_features) + ridge * np.eye(n_features)
    target_covariance = covariance(target_features) + ridge * np.eye(n_features)
    whitening = symmetric_matrix_power(source_covariance, -0.5)
    coloring = symmetric_matrix_power(target_covariance, 0.5)
    return source_features @ whitening @ coloring


def add_predictions(
    rows: list[dict[str, object]],
    *,
    evaluation: str,
    fold: int,
    indices: np.ndarray,
    papers: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    class_names: np.ndarray,
) -> None:
    for index, paper, truth, prediction in zip(indices, papers, actual, predicted):
        rows.append(
            {
                "evaluation": evaluation,
                "fold": fold,
                "row_index": int(index),
                "Ref": str(paper),
                "actual_class": str(class_names[truth]),
                "predicted_class": str(class_names[prediction]),
                "correct": bool(truth == prediction),
            }
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
    missing = [column for column in num + cat if column not in frame.columns]
    assert not missing, f"Columns not found in the input: {missing}"

    labels = LabelEncoder().fit(frame["MIC_class"])
    target = pd.Series(labels.transform(frame["MIC_class"]), index=frame.index)
    features = frame[num + cat]
    standardized_features = within_study_zscore(frame, num)[num + cat]
    groups = frame["Ref"].astype(str)
    group_sizes = groups.value_counts()
    single_row_papers = set(group_sizes[group_sizes.eq(1)].index)
    baseline = frame["MIC_class"].value_counts(normalize=True).max()

    assert len(frame) == 342
    assert groups.nunique() == 65
    assert set(target.unique()) == {0, 1, 2}
    assert len(single_row_papers) == 5

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    coverage = {name: np.zeros(len(frame), dtype=int) for name in ("uncorrected", "standardized", "coral")}

    splitter = GroupKFold(N_SPLITS)
    for fold, (train_index, test_index) in enumerate(splitter.split(features, target, groups), start=1):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        assert not train_groups & test_groups, f"Fold {fold} has source-paper overlap"
        assert set(target.iloc[train_index].unique()) == {0, 1, 2}, f"Fold {fold} lost a class"

        train_target = target.iloc[train_index].to_numpy()
        test_target = target.iloc[test_index].to_numpy()

        # Uncorrected baseline: identical preprocessing/model structure to script 01.
        raw_model = Pipeline([("prep", make_preprocessor(num, cat)), ("clf", make_classifier())])
        raw_model.fit(features.iloc[train_index], train_target)
        raw_predictions = raw_model.predict(features.iloc[test_index]).astype(int)

        corrected_model = Pipeline([("prep", make_preprocessor(num, cat)), ("clf", make_classifier())])
        corrected_model.fit(standardized_features.iloc[train_index], train_target)
        standardized_predictions = corrected_model.predict(standardized_features.iloc[test_index]).astype(int)

        # Fit encoding and scaling once from this fold's training papers. Held-out
        # rows are transformed with those fitted objects, then used only as CORAL
        # target features. Their labels remain untouched until scoring below.
        preprocessor = make_preprocessor(num, cat)
        train_encoded = preprocessor.fit_transform(features.iloc[train_index])
        test_encoded = preprocessor.transform(features.iloc[test_index])
        train_encoded = np.asarray(train_encoded, dtype=float)
        test_encoded = np.asarray(test_encoded, dtype=float)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_encoded)
        test_scaled = scaler.transform(test_encoded)
        coral_predictions = np.empty(len(test_index), dtype=int)

        for paper in sorted(test_groups):
            paper_mask = groups.iloc[test_index].to_numpy() == paper
            paper_positions = np.flatnonzero(paper_mask)
            paper_features = test_scaled[paper_positions]
            # The CORAL function receives no y values. This is the key leakage
            # guard: only the held-out paper's input-feature covariance is used.
            aligned_train = coral_align_source_to_target(train_scaled, paper_features)
            coral_model = make_classifier()
            coral_model.fit(aligned_train, train_target)
            coral_predictions[paper_positions] = coral_model.predict(paper_features).astype(int)

        evaluations = {
            "uncorrected": raw_predictions,
            "standardized": standardized_predictions,
            "coral": coral_predictions,
        }
        for evaluation, predictions in evaluations.items():
            coverage[evaluation][test_index] += 1
            fold_rows.append(
                {
                    "evaluation": evaluation,
                    "fold": fold,
                    "n_train_rows": len(train_index),
                    "n_test_rows": len(test_index),
                    "n_train_papers": len(train_groups),
                    "n_test_papers": len(test_groups),
                    "n_shared_papers": len(train_groups & test_groups),
                    "no_paper_overlap": not bool(train_groups & test_groups),
                    "n_single_row_test_papers": len(test_groups & single_row_papers),
                    "accuracy": accuracy_score(test_target, predictions),
                    "macro_f1": f1_score(test_target, predictions, average="macro", zero_division=0),
                }
            )
            add_predictions(
                prediction_rows,
                evaluation=evaluation,
                fold=fold,
                indices=test_index,
                papers=groups.iloc[test_index],
                actual=test_target,
                predicted=predictions,
                class_names=labels.classes_,
            )

    fold_table = pd.DataFrame(fold_rows)
    predictions_table = pd.DataFrame(prediction_rows)
    assert fold_table.groupby("evaluation").size().eq(N_SPLITS).all()
    assert fold_table["no_paper_overlap"].all()
    assert fold_table["n_shared_papers"].eq(0).all()
    assert all(np.all(values == 1) for values in coverage.values())
    assert len(predictions_table) == len(frame) * 3
    assert predictions_table.groupby("evaluation").size().eq(len(frame)).all()

    summary = (
        fold_table.groupby("evaluation", sort=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_rows=("n_test_rows", "sum"),
            n_papers=("n_test_papers", "sum"),
            n_single_row_papers=("n_single_row_test_papers", "sum"),
        )
        .reset_index()
    )
    summary["majority_baseline"] = baseline
    summary["accuracy_minus_majority_baseline"] = summary["accuracy_mean"] - baseline
    summary["input_sha256"] = file_sha256(INPUT)
    summary["python_version"] = platform.python_version()
    summary["numpy_version"] = np.__version__
    summary["pandas_version"] = pd.__version__
    summary["scipy_version"] = scipy.__version__
    summary["sklearn_version"] = sklearn.__version__
    summary["xgboost_version"] = xgboost.__version__

    uncorrected_accuracy = float(summary.loc[summary["evaluation"].eq("uncorrected"), "accuracy_mean"].iloc[0])
    coral_accuracy = float(summary.loc[summary["evaluation"].eq("coral"), "accuracy_mean"].iloc[0])
    assert abs(uncorrected_accuracy - EXPECTED_GROUPED_ACCURACY) <= REPLICATION_TOLERANCE, (
        f"Uncorrected grouped accuracy {uncorrected_accuracy:.3f} is outside the "
        f"+/- {REPLICATION_TOLERANCE:.2f} replication tolerance around {EXPECTED_GROUPED_ACCURACY:.3f}"
    )
    assert coral_accuracy < MAX_ADAPTED_ACCURACY, (
        f"CORAL grouped accuracy {coral_accuracy:.3f} is implausibly high; inspect leakage."
    )
    assert np.isfinite(summary.select_dtypes(include=np.number).to_numpy()).all()

    # Do not create result files until all checks above pass.
    fold_table.to_csv(FOLDS_OUT, index=False)
    predictions_table.to_csv(PREDICTIONS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("Chen et al. 2026, MIC cross-study adaptation")
    print(f"  rows / papers      : {len(frame)} / {groups.nunique()}")
    print(f"  majority baseline  : {baseline:.3f}")
    print(f"  singleton papers   : {len(single_row_papers)} (CORAL uses identity fallback)")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.evaluation:<12} acc {row.accuracy_mean:.3f} +/- {row.accuracy_std:.3f}"
            f"  f1 {row.macro_f1_mean:.3f} +/- {row.macro_f1_std:.3f}"
            f"  delta {row.accuracy_minus_majority_baseline:+.3f}"
        )
    print(f"Saved {SUMMARY_OUT.relative_to(ROOT)}, {FOLDS_OUT.relative_to(ROOT)}, and {PREDICTIONS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
