"""Replicate the source-paper generalization test on the MBC endpoint.

MBC is the minimum bactericidal concentration. Chen et al. released 133 raw
MBC rows from 24 source papers, but did not provide a categorical MBC target.
This script applies the task-specified operational bins:

    strong    MBC <= 10
    moderate  10 < MBC <= 100
    weak      MBC > 100

The same feature set and XGBoost pipeline used by scripts/01_grouped_cv.py are
evaluated with (1) random stratified cross-validation and (2) source-paper
GroupKFold. The raw CSV is never edited.

Run from the repository root:
    python scripts/04_mbc.py
"""

from __future__ import annotations

import hashlib
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
EXPECTED_RAW_ROWS = 133
EXPECTED_PAPERS = 24

ROOT = Path(__file__).resolve().parents[1]
INPUT_RELATIVE = Path(
    "data/external/chen2026_nanoparticles/Nanoparticles_MBC.csv"
)
INPUT = ROOT / INPUT_RELATIVE
OUTPUT_DIR = ROOT / "outputs/tables"
SUMMARY_OUT = OUTPUT_DIR / "04_mbc_summary.csv"
FOLDS_OUT = OUTPUT_DIR / "04_mbc_fold_support.csv"
PREDICTIONS_OUT = OUTPUT_DIR / "04_mbc_oof_predictions.csv"
CLASS_COUNTS_OUT = OUTPUT_DIR / "04_mbc_class_counts.csv"
DROPPED_OUT = OUTPUT_DIR / "04_mbc_dropped_rows.csv"

CLASS_NAMES = ("strong", "moderate", "weak")
# scripts/01_grouped_cv.py uses LabelEncoder, which orders labels
# alphabetically. Preserve that class-code assignment because XGBoost's
# seeded sampling can still depend on the numeric class index.
ENCODED_CLASS_NAMES = tuple(sorted(CLASS_NAMES))
CLASS_TO_INT = {
    name: index for index, name in enumerate(ENCODED_CLASS_NAMES)
}
INT_TO_CLASS = {index: name for name, index in CLASS_TO_INT.items()}


def file_sha256(path: Path) -> str:
    """Return a stable identifier for the exact input bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_mbc_column(columns: pd.Index) -> str:
    """Find the single MBC outcome column despite its mojibaked unit symbol."""
    candidates = [str(column) for column in columns if str(column).startswith("MBC (")]
    if len(candidates) != 1:
        raise SystemExit(
            "Expected exactly one column beginning with 'MBC ('; "
            f"found {candidates!r}"
        )
    return candidates[0]


def classify_mbc(values: pd.Series) -> pd.Series:
    """Apply the task-specified inclusive boundaries to numeric MBC values."""
    classes = np.select(
        [values.le(10), values.le(100)],
        ["strong", "moderate"],
        default="weak",
    )
    return pd.Series(classes, index=values.index, dtype="string")


def build_pipeline(num_features: list[str], cat_features: list[str]) -> Pipeline:
    """Build the MIC baseline model with preprocessing fitted per fold."""
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            SimpleImputer(strategy="median"),
                            num_features,
                        ),
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
                            cat_features,
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


def evaluate_cv(
    *,
    evaluation: str,
    splitter: StratifiedKFold | GroupKFold,
    use_groups: bool,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    num_features: list[str],
    cat_features: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit every fold manually so group separation and support are auditable."""
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    test_coverage = np.zeros(len(frame), dtype=int)
    tested_groups: list[str] = []

    if use_groups:
        splits = splitter.split(features, target, groups)
    else:
        splits = splitter.split(features, target)

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        shared_groups = train_groups & test_groups

        if use_groups:
            assert not shared_groups, (
                f"Fold {fold} leaked source papers across train and test: "
                f"{sorted(shared_groups)!r}"
            )
            tested_groups.extend(str(group) for group in test_groups)

        test_coverage[test_index] += 1
        model = build_pipeline(num_features, cat_features)
        model.fit(features.iloc[train_index], target.iloc[train_index])
        predictions = model.predict(features.iloc[test_index]).astype(int)

        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index]
        if use_groups:
            assert set(y_train.unique()) == set(CLASS_TO_INT.values()), (
                f"Grouped fold {fold} training data does not contain all classes: "
                f"{sorted(y_train.unique())!r}"
            )
        train_majority = int(y_train.value_counts().idxmax())
        majority_predictions = np.full(len(test_index), train_majority, dtype=int)

        fold_row: dict[str, object] = {
            "evaluation": evaluation,
            "fold": fold,
            "n_train_rows": len(train_index),
            "n_test_rows": len(test_index),
            "n_train_papers": len(train_groups),
            "n_test_papers": len(test_groups),
            "n_shared_papers": len(shared_groups),
            "no_paper_overlap": len(shared_groups) == 0,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=list(CLASS_TO_INT.values()),
                average="macro",
                zero_division=0,
            ),
            "train_majority_class": INT_TO_CLASS[train_majority],
            "majority_accuracy": accuracy_score(y_test, majority_predictions),
        }

        for class_name, class_code in CLASS_TO_INT.items():
            fold_row[f"train_{class_name}"] = int((y_train == class_code).sum())
            fold_row[f"test_{class_name}"] = int((y_test == class_code).sum())
        fold_rows.append(fold_row)

        for row_position, prediction in zip(test_index, predictions, strict=True):
            prediction_rows.append(
                {
                    "evaluation": evaluation,
                    "fold": fold,
                    "raw_row_index": int(frame.iloc[row_position]["raw_row_index"]),
                    "No.": frame.iloc[row_position]["No."],
                    "Ref": groups.iloc[row_position],
                    "MBC_ug_per_mL": frame.iloc[row_position]["MBC_ug_per_mL"],
                    "true_class": INT_TO_CLASS[int(target.iloc[row_position])],
                    "predicted_class": INT_TO_CLASS[int(prediction)],
                    "correct": int(target.iloc[row_position]) == int(prediction),
                }
            )

    assert np.all(test_coverage == 1), (
        f"{evaluation} test coverage must equal one for every analyzed row; "
        f"observed counts {np.unique(test_coverage, return_counts=True)!r}"
    )
    if use_groups:
        assert len(tested_groups) == len(set(tested_groups)) == groups.nunique(), (
            "Every source paper must occur in exactly one grouped test fold; "
            f"observed {len(tested_groups)} assignments for "
            f"{len(set(tested_groups))} unique papers"
        )
    return fold_rows, prediction_rows


def main() -> None:
    input_hash = file_sha256(INPUT)
    raw = pd.read_csv(INPUT)

    print("MBC source-paper generalization analysis")
    print(f"  input file        : {INPUT_RELATIVE}")
    print(f"  resolved input    : {INPUT}")
    print(f"  input SHA-256     : {input_hash}")
    print(f"  raw rows          : {len(raw)}")
    print(f"  raw columns       : {list(raw.columns)!r}")
    print(f"  random state      : {RANDOM_STATE}")
    print("  package versions")
    print(f"    Python          : {sys.version.split()[0]}")
    print(f"    platform        : {platform.platform()}")
    print(f"    NumPy           : {np.__version__}")
    print(f"    pandas          : {pd.__version__}")
    print(f"    scikit-learn    : {sklearn.__version__}")
    print(f"    XGBoost         : {xgboost.__version__}")
    print()

    if len(raw) != EXPECTED_RAW_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_RAW_ROWS} raw MBC rows, found {len(raw)}"
        )

    gram_source = "+ / or  -/'"
    if gram_source not in raw.columns:
        raise SystemExit(
            f"Expected Gram column {gram_source!r}; columns are {list(raw.columns)!r}"
        )
    mbc_source = find_mbc_column(raw.columns)

    frame = raw.copy()
    frame.insert(0, "raw_row_index", np.arange(len(frame), dtype=int))
    frame["Ref"] = frame["Ref"].ffill()
    frame["MBC_numeric"] = pd.to_numeric(frame[mbc_source], errors="coerce")

    drop_reasons = pd.Series("", index=frame.index, dtype="string")
    missing_ref = frame["Ref"].isna()
    invalid_mbc = frame["MBC_numeric"].isna()
    nonpositive_mbc = frame["MBC_numeric"].le(0) & ~invalid_mbc
    drop_reasons.loc[missing_ref] += "missing source paper after Ref forward-fill; "
    drop_reasons.loc[invalid_mbc] += "missing or nonnumeric MBC outcome; "
    drop_reasons.loc[nonpositive_mbc] += "nonpositive MBC outcome; "

    dropped = frame.loc[drop_reasons.ne(""), ["raw_row_index", "No.", "Ref"]].copy()
    dropped["reason"] = drop_reasons.loc[drop_reasons.ne("")].str.rstrip("; ")
    frame = frame.loc[drop_reasons.eq("")].copy().reset_index(drop=True)
    frame = frame.rename(
        columns={
            gram_source: "gram stain",
            "zeta potential": "zeta_binary",
            mbc_source: "MBC_original",
            "MBC_numeric": "MBC_ug_per_mL",
        }
    )

    if len(frame) != EXPECTED_RAW_ROWS:
        raise AssertionError(
            f"Expected to analyze all {EXPECTED_RAW_ROWS} rows; "
            f"analyzed {len(frame)} and logged {len(dropped)} dropped rows"
        )
    if frame["Ref"].nunique() != EXPECTED_PAPERS:
        raise AssertionError(
            f"Expected {EXPECTED_PAPERS} source papers after Ref forward-fill; "
            f"found {frame['Ref'].nunique()}"
        )

    boundary_check = classify_mbc(
        pd.Series([10.0, 10.0001, 100.0, 100.0001])
    )
    assert boundary_check.tolist() == ["strong", "moderate", "moderate", "weak"]
    frame["MBC_class"] = classify_mbc(frame["MBC_ug_per_mL"])
    assert frame.loc[frame["MBC_ug_per_mL"].eq(10), "MBC_class"].eq("strong").all()
    assert frame.loc[frame["MBC_ug_per_mL"].eq(100), "MBC_class"].eq("moderate").all()

    num_features = [
        column for column in frame.columns if column.startswith("MagpieData")
    ] + [
        "size (nm)",
        "zeta_binary",
        "duration",
        "temperature",
    ]
    cat_features = [
        "Shape",
        "bacteria",
        "gram stain",
        "motility",
        "Oxygen Requirement",
        "Shape.1",
        "Arrangement",
    ]
    missing_features = [
        column
        for column in num_features + cat_features
        if column not in frame.columns
    ]
    if missing_features:
        raise SystemExit(f"Required feature columns not found: {missing_features!r}")

    features = frame[num_features + cat_features]
    target = frame["MBC_class"].map(CLASS_TO_INT).astype(int)
    groups = frame["Ref"]

    class_counts = (
        frame["MBC_class"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_counts["proportion"] = class_counts["count"] / len(frame)
    class_counts["under_10_rows"] = class_counts["count"] < 10
    assert int(class_counts["count"].sum()) == len(frame)

    random_folds, random_predictions = evaluate_cv(
        evaluation="random_stratified",
        splitter=StratifiedKFold(
            N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        use_groups=False,
        frame=frame,
        features=features,
        target=target,
        groups=groups,
        num_features=num_features,
        cat_features=cat_features,
    )
    grouped_folds, grouped_predictions = evaluate_cv(
        evaluation="grouped_by_ref",
        splitter=GroupKFold(N_SPLITS),
        use_groups=True,
        frame=frame,
        features=features,
        target=target,
        groups=groups,
        num_features=num_features,
        cat_features=cat_features,
    )

    fold_table = pd.DataFrame(random_folds + grouped_folds)
    predictions_table = pd.DataFrame(random_predictions + grouped_predictions)

    overall_majority_baseline = frame["MBC_class"].value_counts(normalize=True).max()
    summary_rows: list[dict[str, object]] = []
    for evaluation in ("random_stratified", "grouped_by_ref"):
        subset = fold_table.loc[fold_table["evaluation"].eq(evaluation)]
        summary_rows.append(
            {
                "evaluation": evaluation,
                "accuracy_mean": subset["accuracy"].mean(),
                "accuracy_std": subset["accuracy"].std(ddof=0),
                "accuracy_delta_over_majority": (
                    subset["accuracy"].mean() - overall_majority_baseline
                ),
                "macro_f1_mean": subset["macro_f1"].mean(),
                "macro_f1_std": subset["macro_f1"].std(ddof=0),
                "cv_majority_accuracy_mean": subset["majority_accuracy"].mean(),
                "cv_majority_accuracy_std": subset["majority_accuracy"].std(ddof=0),
                "overall_majority_baseline": overall_majority_baseline,
                "n_raw_rows": len(raw),
                "n_analyzed_rows": len(frame),
                "n_dropped_rows": len(dropped),
                "n_papers": groups.nunique(),
                "n_splits": N_SPLITS,
                "random_state": RANDOM_STATE,
                "model": "XGBClassifier",
                "n_jobs": 1,
                "input_file": str(INPUT_RELATIVE),
                "resolved_input_file": str(INPUT),
                "input_sha256": input_hash,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "sklearn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
            }
        )
    summary = pd.DataFrame(summary_rows)

    random_accuracy = float(
        summary.loc[
            summary["evaluation"].eq("random_stratified"),
            "accuracy_mean",
        ].iloc[0]
    )
    grouped_accuracy = float(
        summary.loc[
            summary["evaluation"].eq("grouped_by_ref"),
            "accuracy_mean",
        ].iloc[0]
    )
    grouped_delta_over_baseline = grouped_accuracy - overall_majority_baseline
    if not 0.55 <= random_accuracy <= 0.80:
        warnings.warn(
            f"Random accuracy {random_accuracy:.3f} is outside the expected "
            "range 0.55-0.80; reporting the result unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    assert grouped_delta_over_baseline < 0.10, (
        "Grouped accuracy exceeds the overall majority baseline by "
        f"{grouped_delta_over_baseline:.3f}, which is not below 0.10"
    )
    if grouped_accuracy >= random_accuracy:
        warnings.warn(
            f"Grouped accuracy {grouped_accuracy:.3f} is not below random "
            f"accuracy {random_accuracy:.3f}; reporting the result unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    grouped_support = fold_table.loc[fold_table["evaluation"].eq("grouped_by_ref")]
    assert grouped_support["no_paper_overlap"].all()
    assert grouped_support["n_shared_papers"].eq(0).all()

    expected_evaluations = {"random_stratified", "grouped_by_ref"}
    assert set(fold_table["evaluation"]) == expected_evaluations
    assert fold_table.groupby("evaluation").size().eq(N_SPLITS).all()
    assert fold_table[["n_train_rows", "n_test_rows"]].sum(axis=1).eq(len(frame)).all()
    train_support_columns = [f"train_{name}" for name in CLASS_NAMES]
    test_support_columns = [f"test_{name}" for name in CLASS_NAMES]
    assert fold_table[train_support_columns].sum(axis=1).eq(
        fold_table["n_train_rows"]
    ).all()
    assert fold_table[test_support_columns].sum(axis=1).eq(
        fold_table["n_test_rows"]
    ).all()

    metric_columns = ["accuracy", "macro_f1", "majority_accuracy"]
    metric_values = fold_table[metric_columns].to_numpy(dtype=float)
    assert np.isfinite(metric_values).all()
    assert ((metric_values >= 0) & (metric_values <= 1)).all()

    assert len(predictions_table) == len(frame) * len(expected_evaluations)
    assert predictions_table.groupby("evaluation").size().eq(len(frame)).all()
    assert set(predictions_table["evaluation"]) == expected_evaluations

    # Write outputs only after every verification check has passed, so a failed
    # run cannot leave behind files that look like validated results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(DROPPED_OUT, index=False)
    class_counts.to_csv(CLASS_COUNTS_OUT, index=False)
    fold_table.to_csv(FOLDS_OUT, index=False)
    predictions_table.to_csv(PREDICTIONS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"  analyzed rows     : {len(frame)}")
    print(f"  dropped rows      : {len(dropped)}")
    print(f"  source papers     : {groups.nunique()}")
    print(f"  class counts      : {dict(zip(class_counts['class'], class_counts['count']))}")
    small_classes = class_counts.loc[class_counts["under_10_rows"], "class"].tolist()
    print(f"  classes <10 rows  : {small_classes if small_classes else 'none'}")
    print(f"  majority baseline : {overall_majority_baseline:.3f}")
    print()
    for row in summary.itertuples(index=False):
        print(f"  {row.evaluation}")
        print(f"    accuracy         {row.accuracy_mean:.3f} +/- {row.accuracy_std:.3f}")
        print(f"    delta vs baseline {row.accuracy_delta_over_majority:+.3f}")
        print(f"    macro F1         {row.macro_f1_mean:.3f} +/- {row.macro_f1_std:.3f}")
        print(
            "    CV majority      "
            f"{row.cv_majority_accuracy_mean:.3f} +/- "
            f"{row.cv_majority_accuracy_std:.3f}"
        )
    print()
    print("  grouped fold support")
    print(
        grouped_support[
            [
                "fold",
                "n_test_rows",
                "n_test_papers",
                "test_strong",
                "test_moderate",
                "test_weak",
                "accuracy",
                "macro_f1",
                "n_shared_papers",
            ]
        ].to_string(index=False)
    )
    print()
    print("Verification passed:")
    print(f"  raw/analyzed rows : {len(raw)}/{len(frame)}")
    print(f"  papers            : {groups.nunique()}")
    print("  grouped overlap   : zero papers in every fold")
    print(f"  random accuracy   : {random_accuracy:.3f} (expected 0.55-0.80)")
    print(f"  grouped accuracy  : {grouped_accuracy:.3f}")
    print(
        "  grouped delta     : "
        f"{grouped_delta_over_baseline:+.3f} vs majority "
        "(required < +0.100)"
    )
    print("Saved:")
    for path in (
        SUMMARY_OUT,
        FOLDS_OUT,
        PREDICTIONS_OUT,
        CLASS_COUNTS_OUT,
        DROPPED_OUT,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
