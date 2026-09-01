"""Audit species representation and grouped-CV performance for the MIC model.

The model in this script is the uncorrected source-paper GroupKFold baseline
from scripts/01_grouped_cv.py. Species cleanup is deliberately post hoc: the
pipeline receives the original ``bacteria`` strings, while normalized species
names are attached only to out-of-fold (OOF) predictions for auditing.

This distinction matters. Replacing the raw model feature with normalized
names would create a different model rather than analyze the established one.

Run from the repository root:
    python scripts/06_species_analysis.py
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_SPLITS = 5
EXPECTED_ROWS = 342
EXPECTED_PAPERS = 65
CANONICAL_GROUPED_ACCURACY = 0.3162
PLATFORM_TOLERANCE = 0.02

ROOT = Path(__file__).resolve().parents[1]
INPUT_RELATIVE = Path(
    "data/external/chen2026_nanoparticles/"
    "Nanoparticles_MIC_with_class.csv"
)
INPUT = ROOT / INPUT_RELATIVE
OUTPUT_DIR = ROOT / "outputs/tables"

SUMMARY_OUT = OUTPUT_DIR / "06_species_analysis_summary.csv"
MAPPING_OUT = OUTPUT_DIR / "06_species_analysis_name_mapping.csv"
DISTRIBUTION_OUT = OUTPUT_DIR / "06_species_analysis_distribution.csv"
OOF_OUT = OUTPUT_DIR / "06_species_analysis_oof_predictions.csv"
FOLDS_OUT = OUTPUT_DIR / "06_species_analysis_fold_metrics.csv"
SPECIES_PERFORMANCE_OUT = (
    OUTPUT_DIR / "06_species_analysis_species_performance.csv"
)
BUCKET_PERFORMANCE_OUT = (
    OUTPUT_DIR / "06_species_analysis_bucket_performance.csv"
)
ORGANISM_PERFORMANCE_OUT = (
    OUTPUT_DIR / "06_species_analysis_organism_performance.csv"
)
NONBACTERIAL_OUT = OUTPUT_DIR / "06_species_analysis_nonbacterial_rows.csv"

CLASS_DISPLAY_ORDER = ("strong", "moderate", "weak")
BUCKET_ORDER = (">20", "5-20", "<5")

# Conservative, explicit species-level cleanup. Serovars remain separate; the
# script does not silently collapse every Salmonella label into S. enterica.
SPECIES_ALIASES: dict[str, tuple[str, str]] = {
    "A.baumannii": (
        "A. baumannii",
        "spacing correction",
    ),
    "E. fecalis": (
        "E. faecalis",
        "spelling correction",
    ),
    "E. aerogenes": (
        "K. aerogenes",
        "taxonomic synonym",
    ),
    "MRSA": (
        "S. aureus",
        "resistance phenotype collapsed to species",
    ),
    "Salmonella typhimurium": (
        "S. typhimurium",
        "genus abbreviation harmonized",
    ),
    "Salmonella typhi": (
        "S. typhi",
        "genus abbreviation harmonized",
    ),
    "C. parapisilosis": (
        "C. parapsilosis",
        "spelling correction",
    ),
    "L. pneumophilia subsp. pneumophila": (
        "L. pneumophila subsp. pneumophila",
        "species spelling correction; subspecies retained",
    ),
}

# Manually curated for the normalized species present in this frozen dataset;
# review and extend this set before analyzing data containing new organisms.
FUNGAL_SPECIES = {
    "C. albicans",
    "C. tropicalis",
    "C. parapsilosis",
    "C. glabrata",
}

EXPECTED_DOMINANT_COUNTS = {
    "E. coli": 70,
    "S. aureus": 67,
    "P. aeruginosa": 40,
}
EXPECTED_NONBACTERIAL_ROWS = 15


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for the exact input bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_species_label(raw_value: str) -> dict[str, str]:
    """Return an auditable cleanup record for one exact raw label."""
    raw = str(raw_value)
    cleaned = raw.replace("\u00c2\u00a0", " ").replace("\u00a0", " ")
    cleaned = " ".join(cleaned.split())

    rules: list[str] = []
    if "\u00c2\u00a0" in raw:
        rules.append("replace mojibaked nonbreaking space")
    elif "\u00a0" in raw:
        rules.append("replace nonbreaking space")
    if raw != cleaned:
        rules.append("trim/collapse whitespace")

    normalized = cleaned
    if cleaned in SPECIES_ALIASES:
        normalized, alias_rule = SPECIES_ALIASES[cleaned]
        rules.append(alias_rule)

    organism_type = "fungus" if normalized in FUNGAL_SPECIES else "bacterium"
    return {
        "raw_species": raw,
        "raw_species_repr": ascii(raw),
        "cleaned_species": cleaned,
        "normalized_species": normalized,
        "normalization_rule": "; ".join(rules) if rules else "unchanged",
        "organism_type": organism_type,
    }


def frequency_bucket(species_count: int) -> str:
    """Apply the assignment's mutually exclusive frequency buckets."""
    if species_count > 20:
        return ">20"
    if 5 <= species_count <= 20:
        return "5-20"
    return "<5"


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_classes: int,
) -> Pipeline:
    """Reproduce the uncorrected MIC pipeline from script 01."""
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            SimpleImputer(strategy="median"),
                            numeric_features,
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
                            categorical_features,
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
                    num_class=n_classes,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def summarize_oof_subset(
    subset: pd.DataFrame,
    *,
    class_codes: list[int],
) -> dict[str, object]:
    """Summarize pooled OOF predictions without refitting a model."""
    if subset.empty:
        raise AssertionError("Cannot summarize an empty OOF subset")
    true_codes = subset["true_code"].astype(int)
    predicted_codes = subset["predicted_code"].astype(int)
    majority_baseline = float(true_codes.value_counts(normalize=True).max())
    accuracy = float(accuracy_score(true_codes, predicted_codes))
    result: dict[str, object] = {
        "n_rows": len(subset),
        "n_species": subset["normalized_species"].nunique(),
        "n_papers": subset["Ref"].nunique(),
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "accuracy_delta_over_baseline": accuracy - majority_baseline,
        "macro_f1_3class": f1_score(
            true_codes,
            predicted_codes,
            labels=class_codes,
            average="macro",
            zero_division=0,
        ),
    }
    for class_name in CLASS_DISPLAY_ORDER:
        result[f"true_{class_name}"] = int(
            subset["true_class"].eq(class_name).sum()
        )
    return result


def main() -> None:
    input_hash = file_sha256(INPUT)
    raw = pd.read_csv(INPUT)

    print("MIC species representation and grouped-CV analysis")
    print(f"  input file        : {INPUT_RELATIVE}")
    print(f"  resolved input    : {INPUT}")
    print(f"  input SHA-256     : {input_hash}")
    print(f"  raw rows          : {len(raw)}")
    print(f"  random state      : {RANDOM_STATE}")
    print("  package versions")
    print(f"    Python          : {sys.version.split()[0]}")
    print(f"    platform        : {platform.platform()}")
    print(f"    NumPy           : {np.__version__}")
    print(f"    pandas          : {pd.__version__}")
    print(f"    scikit-learn    : {sklearn.__version__}")
    print(f"    XGBoost         : {xgboost.__version__}")
    print()

    assert len(raw) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} MIC rows, found {len(raw)}"
    )
    required_columns = {"No.", "Ref", "bacteria", "MIC_class"}
    missing_required = required_columns - set(raw.columns)
    if missing_required:
        raise SystemExit(f"Missing required columns: {sorted(missing_required)!r}")
    assert raw["No."].notna().all() and raw["No."].is_unique
    assert raw["bacteria"].notna().all()
    assert raw["MIC_class"].notna().all()

    frame = raw.copy()
    frame.insert(0, "raw_row_index", np.arange(len(frame), dtype=int))
    frame["Ref"] = frame["Ref"].ffill()
    assert frame["Ref"].notna().all()
    assert frame["Ref"].nunique() == EXPECTED_PAPERS, (
        f"Expected {EXPECTED_PAPERS} source papers, found {frame['Ref'].nunique()}"
    )

    normalization = pd.DataFrame(
        [normalize_species_label(value) for value in frame["bacteria"]],
        index=frame.index,
    )
    frame = pd.concat([frame, normalization], axis=1)
    assert frame["raw_species"].equals(raw["bacteria"])
    assert frame[
        ["cleaned_species", "normalized_species", "organism_type"]
    ].notna().all().all()
    assert set(frame["organism_type"]) == {"bacterium", "fungus"}

    normalized_counts = frame["normalized_species"].value_counts()
    frame["normalized_species_count"] = frame["normalized_species"].map(
        normalized_counts
    )
    frame["frequency_bucket"] = frame["normalized_species_count"].map(
        frequency_bucket
    )
    boundary_check = {
        value: frequency_bucket(value) for value in (4, 5, 20, 21)
    }
    assert boundary_check == {4: "<5", 5: "5-20", 20: "5-20", 21: ">20"}
    assert frame.loc[
        frame["normalized_species_count"].eq(5), "frequency_bucket"
    ].eq("5-20").all()

    dominant_counts = {
        name: int(normalized_counts.get(name, 0))
        for name in EXPECTED_DOMINANT_COUNTS
    }
    assert dominant_counts == EXPECTED_DOMINANT_COUNTS, (
        f"Dominant normalized species counts changed: {dominant_counts!r}"
    )
    nonbacterial_count = int(frame["organism_type"].ne("bacterium").sum())
    assert nonbacterial_count == EXPECTED_NONBACTERIAL_ROWS, (
        f"Expected {EXPECTED_NONBACTERIAL_ROWS} fungal rows, "
        f"found {nonbacterial_count}"
    )

    # Preserve the baseline's exact model feature contract. Normalized species
    # columns never enter X.
    frame = frame.rename(columns={"zeta potential": "zeta_binary"})
    numeric_features = [
        column for column in frame.columns if column.startswith("MagpieData")
    ] + [
        "size (nm)",
        "zeta_binary",
        "duration",
        "temperature",
    ]
    categorical_features = [
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
        for column in numeric_features + categorical_features
        if column not in frame.columns
    ]
    if missing_features:
        raise SystemExit(f"Required model features missing: {missing_features!r}")
    assert len([c for c in numeric_features if c.startswith("MagpieData")]) == 22
    assert len(numeric_features) == 26
    assert len(categorical_features) == 7

    features = frame[numeric_features + categorical_features].copy()
    assert "Ref" not in features.columns
    assert "MIC_class" not in features.columns
    assert "normalized_species" not in features.columns
    assert features["bacteria"].equals(raw["bacteria"])

    encoder = LabelEncoder()
    target = pd.Series(
        encoder.fit_transform(frame["MIC_class"]),
        index=frame.index,
        dtype=int,
    )
    assert set(encoder.classes_) == set(CLASS_DISPLAY_ORDER)
    class_codes = list(range(len(encoder.classes_)))
    class_to_code = {
        class_name: int(encoder.transform([class_name])[0])
        for class_name in encoder.classes_
    }
    groups = frame["Ref"]

    predicted_codes = np.full(len(frame), -1, dtype=int)
    fold_ids = np.full(len(frame), -1, dtype=int)
    test_coverage = np.zeros(len(frame), dtype=int)
    tested_groups: list[str] = []
    fold_rows: list[dict[str, object]] = []

    splitter = GroupKFold(N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(features, target, groups),
        start=1,
    ):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        shared_groups = train_groups & test_groups
        assert not shared_groups, (
            f"Fold {fold} leaked source papers: {sorted(shared_groups)!r}"
        )
        tested_groups.extend(str(group) for group in test_groups)
        test_coverage[test_index] += 1

        model = build_pipeline(
            numeric_features,
            categorical_features,
            len(encoder.classes_),
        )
        model.fit(features.iloc[train_index], target.iloc[train_index])
        fold_predictions = model.predict(features.iloc[test_index]).astype(int)
        predicted_codes[test_index] = fold_predictions
        fold_ids[test_index] = fold

        y_train = target.iloc[train_index]
        y_test = target.iloc[test_index]
        assert set(y_train.unique()) == set(class_codes)
        fold_row: dict[str, object] = {
            "fold": fold,
            "n_train_rows": len(train_index),
            "n_test_rows": len(test_index),
            "n_train_papers": len(train_groups),
            "n_test_papers": len(test_groups),
            "n_shared_papers": len(shared_groups),
            "no_paper_overlap": not shared_groups,
            "accuracy": accuracy_score(y_test, fold_predictions),
            "macro_f1_3class": f1_score(
                y_test,
                fold_predictions,
                labels=class_codes,
                average="macro",
                zero_division=0,
            ),
        }
        for class_name in CLASS_DISPLAY_ORDER:
            class_code = class_to_code[class_name]
            fold_row[f"train_{class_name}"] = int((y_train == class_code).sum())
            fold_row[f"test_{class_name}"] = int((y_test == class_code).sum())
        fold_rows.append(fold_row)

    assert np.all(test_coverage == 1)
    assert np.all(predicted_codes >= 0)
    assert np.all(fold_ids >= 1)
    assert len(tested_groups) == len(set(tested_groups)) == groups.nunique()

    fold_table = pd.DataFrame(fold_rows)
    assert len(fold_table) == N_SPLITS
    assert fold_table["n_shared_papers"].eq(0).all()
    assert fold_table["no_paper_overlap"].all()
    assert fold_table[["n_train_rows", "n_test_rows"]].sum(axis=1).eq(
        len(frame)
    ).all()
    for prefix, size_column in (
        ("train", "n_train_rows"),
        ("test", "n_test_rows"),
    ):
        support_columns = [
            f"{prefix}_{class_name}" for class_name in CLASS_DISPLAY_ORDER
        ]
        assert fold_table[support_columns].sum(axis=1).eq(
            fold_table[size_column]
        ).all()
    metric_values = fold_table[["accuracy", "macro_f1_3class"]].to_numpy(
        dtype=float
    )
    assert np.isfinite(metric_values).all()
    assert ((metric_values >= 0) & (metric_values <= 1)).all()

    oof = pd.DataFrame(
        {
            "raw_row_index": frame["raw_row_index"],
            "No.": frame["No."],
            "Ref": frame["Ref"],
            "fold": fold_ids,
            "raw_species": frame["raw_species"],
            "raw_species_repr": frame["raw_species_repr"],
            "cleaned_species": frame["cleaned_species"],
            "normalized_species": frame["normalized_species"],
            "normalization_rule": frame["normalization_rule"],
            "organism_type": frame["organism_type"],
            "normalized_species_count": frame["normalized_species_count"],
            "frequency_bucket": frame["frequency_bucket"],
            "true_code": target,
            "predicted_code": predicted_codes,
            "true_class": encoder.inverse_transform(target),
            "predicted_class": encoder.inverse_transform(predicted_codes),
        }
    )
    oof["correct"] = oof["true_code"].eq(oof["predicted_code"])
    assert len(oof) == EXPECTED_ROWS
    assert oof["raw_row_index"].nunique() == EXPECTED_ROWS
    assert oof.groupby("Ref")["fold"].nunique().eq(1).all()
    assert oof["Ref"].nunique() == EXPECTED_PAPERS
    assert oof.groupby("normalized_species")["frequency_bucket"].nunique().eq(
        1
    ).all()
    assert oof.groupby("normalized_species")[
        "normalized_species_count"
    ].nunique().eq(1).all()

    # Exact raw-to-normalized audit, including visible representations for
    # whitespace and mojibake that a spreadsheet may otherwise conceal.
    mapping_columns = [
        "raw_species",
        "raw_species_repr",
        "cleaned_species",
        "normalized_species",
        "normalization_rule",
        "organism_type",
    ]
    name_mapping = (
        frame.groupby(mapping_columns, dropna=False, sort=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["row_count", "raw_species_repr"], ascending=[False, True])
        .reset_index(drop=True)
    )
    assert int(name_mapping["row_count"].sum()) == EXPECTED_ROWS
    assert name_mapping.groupby("raw_species_repr")["normalized_species"].nunique().eq(
        1
    ).all()

    distribution_base = (
        frame.groupby("normalized_species", sort=False)
        .agg(
            row_count=("normalized_species", "size"),
            paper_count=("Ref", "nunique"),
            raw_label_count=("raw_species_repr", "nunique"),
            raw_label_variants=(
                "raw_species_repr",
                lambda values: " | ".join(sorted(set(values))),
            ),
            organism_type=("organism_type", "first"),
            frequency_bucket=("frequency_bucket", "first"),
        )
        .reset_index()
    )
    class_support = pd.crosstab(
        frame["normalized_species"],
        frame["MIC_class"],
    ).reindex(columns=CLASS_DISPLAY_ORDER, fill_value=0)
    class_support = class_support.rename(
        columns={name: f"true_{name}" for name in CLASS_DISPLAY_ORDER}
    ).reset_index()
    distribution = distribution_base.merge(
        class_support,
        on="normalized_species",
        validate="one_to_one",
    )
    distribution["row_proportion"] = distribution["row_count"] / len(frame)
    distribution = distribution.sort_values(
        ["row_count", "normalized_species"],
        ascending=[False, True],
    ).reset_index(drop=True)
    distribution.insert(0, "rank", np.arange(1, len(distribution) + 1))
    assert int(distribution["row_count"].sum()) == EXPECTED_ROWS
    assert distribution["normalized_species"].is_unique
    assert distribution.loc[
        distribution["normalized_species"].isin(EXPECTED_DOMINANT_COUNTS),
        "rank",
    ].tolist() == [1, 2, 3]

    species_rows: list[dict[str, object]] = []
    for distribution_row in distribution.itertuples(index=False):
        subset = oof.loc[
            oof["normalized_species"].eq(distribution_row.normalized_species)
        ]
        result = summarize_oof_subset(subset, class_codes=class_codes)
        result.update(
            {
                "rank": distribution_row.rank,
                "normalized_species": distribution_row.normalized_species,
                "organism_type": distribution_row.organism_type,
                "frequency_bucket": distribution_row.frequency_bucket,
                "raw_label_count": distribution_row.raw_label_count,
            }
        )
        species_rows.append(result)
    species_performance = pd.DataFrame(species_rows).sort_values("rank")
    assert len(species_performance) == len(distribution)
    assert int(species_performance["n_rows"].sum()) == EXPECTED_ROWS

    bucket_rows: list[dict[str, object]] = []
    for population, population_mask in (
        ("all_rows", pd.Series(True, index=oof.index)),
        ("bacteria_only", oof["organism_type"].eq("bacterium")),
    ):
        population_total = int(population_mask.sum())
        for bucket in BUCKET_ORDER:
            subset = oof.loc[
                population_mask & oof["frequency_bucket"].eq(bucket)
            ]
            result = summarize_oof_subset(subset, class_codes=class_codes)
            result.update(
                {
                    "population": population,
                    "frequency_bucket": bucket,
                    "row_proportion_within_population": len(subset)
                    / population_total,
                }
            )
            bucket_rows.append(result)
    bucket_performance = pd.DataFrame(bucket_rows)
    assert bucket_performance.groupby("population")["n_rows"].sum().to_dict() == {
        "all_rows": EXPECTED_ROWS,
        "bacteria_only": EXPECTED_ROWS - EXPECTED_NONBACTERIAL_ROWS,
    }

    organism_rows: list[dict[str, object]] = []
    for organism_type in ("bacterium", "fungus"):
        subset = oof.loc[oof["organism_type"].eq(organism_type)]
        result = summarize_oof_subset(subset, class_codes=class_codes)
        result["organism_type"] = organism_type
        result["row_proportion"] = len(subset) / len(oof)
        organism_rows.append(result)
    organism_performance = pd.DataFrame(organism_rows)
    assert int(organism_performance["n_rows"].sum()) == EXPECTED_ROWS

    nonbacterial_rows = oof.loc[oof["organism_type"].ne("bacterium")].copy()
    assert len(nonbacterial_rows) == EXPECTED_NONBACTERIAL_ROWS
    assert set(nonbacterial_rows["normalized_species"]) == FUNGAL_SPECIES

    fold_accuracy_mean = float(fold_table["accuracy"].mean())
    fold_accuracy_std = float(fold_table["accuracy"].std(ddof=0))
    fold_f1_mean = float(fold_table["macro_f1_3class"].mean())
    fold_f1_std = float(fold_table["macro_f1_3class"].std(ddof=0))
    canonical_difference = fold_accuracy_mean - CANONICAL_GROUPED_ACCURACY
    if abs(canonical_difference) > PLATFORM_TOLERANCE:
        warnings.warn(
            f"Grouped accuracy {fold_accuracy_mean:.4f} differs from canonical "
            f"{CANONICAL_GROUPED_ACCURACY:.4f} by {canonical_difference:+.4f}, "
            f"outside the accepted +/-{PLATFORM_TOLERANCE:.2f} platform "
            "tolerance; reporting the result unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    global_oof = summarize_oof_subset(oof, class_codes=class_codes)
    dominant_combined_count = sum(EXPECTED_DOMINANT_COUNTS.values())

    summary = pd.DataFrame(
        [
            {
                "evaluation": "MIC_grouped_by_ref_species_audit",
                "fold_accuracy_mean": fold_accuracy_mean,
                "fold_accuracy_std": fold_accuracy_std,
                "fold_macro_f1_3class_mean": fold_f1_mean,
                "fold_macro_f1_3class_std": fold_f1_std,
                "pooled_oof_accuracy": global_oof["accuracy"],
                "pooled_oof_macro_f1_3class": global_oof[
                    "macro_f1_3class"
                ],
                "overall_majority_baseline": global_oof["majority_baseline"],
                "pooled_accuracy_delta_over_baseline": global_oof[
                    "accuracy_delta_over_baseline"
                ],
                "canonical_grouped_accuracy": CANONICAL_GROUPED_ACCURACY,
                "difference_from_canonical": canonical_difference,
                "platform_tolerance": PLATFORM_TOLERANCE,
                "n_rows": len(frame),
                "n_papers": groups.nunique(),
                "n_raw_species_labels": frame["raw_species"].nunique(),
                "n_normalized_species": frame["normalized_species"].nunique(),
                "n_bacterial_rows": int(frame["organism_type"].eq("bacterium").sum()),
                "n_fungal_rows": nonbacterial_count,
                "fungal_row_proportion": nonbacterial_count / len(frame),
                "e_coli_count": dominant_counts["E. coli"],
                "s_aureus_count": dominant_counts["S. aureus"],
                "p_aeruginosa_count": dominant_counts["P. aeruginosa"],
                "dominant_three_combined_count": dominant_combined_count,
                "dominant_three_combined_proportion": dominant_combined_count
                / len(frame),
                "n_splits": N_SPLITS,
                "random_state": RANDOM_STATE,
                "model": "XGBClassifier",
                "n_jobs": 1,
                "no_paper_overlap": True,
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
        ]
    )

    tables = {
        SUMMARY_OUT: summary,
        MAPPING_OUT: name_mapping,
        DISTRIBUTION_OUT: distribution,
        OOF_OUT: oof,
        FOLDS_OUT: fold_table,
        SPECIES_PERFORMANCE_OUT: species_performance,
        BUCKET_PERFORMANCE_OUT: bucket_performance,
        ORGANISM_PERFORMANCE_OUT: organism_performance,
        NONBACTERIAL_OUT: nonbacterial_rows,
    }
    for table in tables.values():
        assert not table.columns.duplicated().any()

    # Write only after every data, leakage, metric, and reconciliation check.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, table in tables.items():
        table.to_csv(path, index=False)

    print(f"  analyzed rows     : {len(frame)}")
    print(f"  source papers     : {groups.nunique()}")
    print(f"  raw labels        : {frame['raw_species'].nunique()}")
    print(f"  normalized species: {frame['normalized_species'].nunique()}")
    print(f"  fungal rows       : {nonbacterial_count} ({nonbacterial_count / len(frame):.3%})")
    print(f"  dominant counts   : {dominant_counts}")
    print(
        "  dominant share    : "
        f"{dominant_combined_count}/{len(frame)} "
        f"({dominant_combined_count / len(frame):.3%})"
    )
    print()
    print("  grouped model")
    print(f"    accuracy         {fold_accuracy_mean:.3f} +/- {fold_accuracy_std:.3f}")
    print(f"    macro F1 (3 cls) {fold_f1_mean:.3f} +/- {fold_f1_std:.3f}")
    print(
        "    canonical delta "
        f"{canonical_difference:+.3f} "
        f"(accepted +/-{PLATFORM_TOLERANCE:.2f})"
    )
    print()
    print("  all-row frequency buckets")
    print(
        bucket_performance.loc[
            bucket_performance["population"].eq("all_rows"),
            [
                "frequency_bucket",
                "n_rows",
                "n_species",
                "accuracy",
                "majority_baseline",
                "accuracy_delta_over_baseline",
                "macro_f1_3class",
            ],
        ].to_string(index=False)
    )
    print()
    print("Verification passed:")
    print(f"  OOF coverage      : {len(oof)}/{len(frame)} rows exactly once")
    print("  grouped overlap   : zero papers in every fold")
    print("  subgroup models   : none; all strata use one OOF prediction set")
    print("Saved:")
    for path in tables:
        print(f"  {path}")


if __name__ == "__main__":
    main()
