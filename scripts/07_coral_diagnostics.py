"""Describe where CORAL changes unseen-paper performance.

This script uses the already-verified out-of-fold predictions from
scripts/05_adaptation.py. It performs no model fitting and does not create a
new headline score. It answers the diagnostic questions: which papers improved
or worsened under CORAL, and are changes associated with paper size?

Run from the repository root after scripts/05_adaptation.py:
    python scripts/07_coral_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/tables/adaptation_oof_predictions.csv"
OUTPUT_DIR = ROOT / "outputs/tables"
PAPER_OUT = OUTPUT_DIR / "coral_per_paper_diagnostics.csv"
SUMMARY_OUT = OUTPUT_DIR / "coral_diagnostics_summary.csv"
CLASS_ORDER = ["moderate", "strong", "weak"]


def size_bucket(n_rows: int) -> str:
    if n_rows == 1:
        return "1 row (CORAL identity fallback)"
    if n_rows <= 4:
        return "2-4 rows"
    return "5+ rows"


def main() -> None:
    predictions = pd.read_csv(INPUT)
    required = {
        "evaluation", "fold", "row_index", "Ref", "actual_class",
        "predicted_class", "correct",
    }
    assert required <= set(predictions.columns), "Unexpected adaptation prediction schema"
    assert set(predictions["evaluation"]) == {"uncorrected", "standardized", "coral"}
    assert predictions.groupby("evaluation")["row_index"].nunique().eq(342).all()

    records: list[dict[str, object]] = []
    for paper, paper_rows in predictions.groupby("Ref", sort=True):
        evaluations = {name: values for name, values in paper_rows.groupby("evaluation")}
        assert set(evaluations) == {"uncorrected", "standardized", "coral"}
        reference = evaluations["uncorrected"].sort_values("row_index")
        n_rows = len(reference)
        actual = reference["actual_class"].to_numpy()
        row: dict[str, object] = {
            "Ref": paper,
            "fold": int(reference["fold"].iloc[0]),
            "n_rows": n_rows,
            "size_bucket": size_bucket(n_rows),
            "n_observed_classes": int(pd.Series(actual).nunique()),
            "actual_moderate": int((actual == "moderate").sum()),
            "actual_strong": int((actual == "strong").sum()),
            "actual_weak": int((actual == "weak").sum()),
        }
        for evaluation, values in evaluations.items():
            values = values.sort_values("row_index")
            assert values["row_index"].tolist() == reference["row_index"].tolist()
            predicted = values["predicted_class"].to_numpy()
            row[f"{evaluation}_accuracy"] = float(values["correct"].mean())
            row[f"{evaluation}_macro_f1"] = f1_score(
                actual, predicted, labels=CLASS_ORDER, average="macro", zero_division=0
            )
        row["coral_minus_uncorrected_accuracy"] = (
            row["coral_accuracy"] - row["uncorrected_accuracy"]
        )
        row["coral_minus_standardized_accuracy"] = (
            row["coral_accuracy"] - row["standardized_accuracy"]
        )
        row["coral_minus_uncorrected_macro_f1"] = (
            row["coral_macro_f1"] - row["uncorrected_macro_f1"]
        )
        row["coral_minus_standardized_macro_f1"] = (
            row["coral_macro_f1"] - row["standardized_macro_f1"]
        )
        records.append(row)

    paper_table = pd.DataFrame(records).sort_values(["fold", "Ref"]).reset_index(drop=True)
    assert len(paper_table) == 65
    assert paper_table["n_rows"].sum() == 342
    assert paper_table["fold"].value_counts().eq(13).all()
    assert paper_table.loc[paper_table["n_rows"].eq(1), "size_bucket"].eq(
        "1 row (CORAL identity fallback)"
    ).all()

    bucket_summary = (
        paper_table.groupby("size_bucket", sort=False)
        .agg(
            n_papers=("Ref", "size"),
            n_rows=("n_rows", "sum"),
            coral_accuracy_mean=("coral_accuracy", "mean"),
            coral_macro_f1_mean=("coral_macro_f1", "mean"),
            coral_gain_vs_uncorrected_accuracy_mean=("coral_minus_uncorrected_accuracy", "mean"),
            coral_gain_vs_standardized_accuracy_mean=("coral_minus_standardized_accuracy", "mean"),
        )
        .reset_index()
    )
    rho_acc, p_acc = spearmanr(
        paper_table["n_rows"], paper_table["coral_minus_uncorrected_accuracy"]
    )
    rho_f1, p_f1 = spearmanr(
        paper_table["n_rows"], paper_table["coral_minus_uncorrected_macro_f1"]
    )
    overall = pd.DataFrame(
        [
            {
                "section": "overall",
                "measure": "papers improved / unchanged / worsened in CORAL accuracy vs uncorrected",
                "value": f"{int((paper_table['coral_minus_uncorrected_accuracy'] > 0).sum())} / "
                f"{int((paper_table['coral_minus_uncorrected_accuracy'] == 0).sum())} / "
                f"{int((paper_table['coral_minus_uncorrected_accuracy'] < 0).sum())}",
            },
            {
                "section": "overall",
                "measure": "Spearman rho: paper rows vs CORAL accuracy gain over uncorrected",
                "value": rho_acc,
                "p_value": p_acc,
            },
            {
                "section": "overall",
                "measure": "Spearman rho: paper rows vs CORAL macro-F1 gain over uncorrected",
                "value": rho_f1,
                "p_value": p_f1,
            },
        ]
    )
    bucket_output = bucket_summary.assign(section="paper-size bucket", measure=lambda x: x["size_bucket"])
    summary = pd.concat([overall, bucket_output], ignore_index=True, sort=False)
    assert np.isfinite(paper_table.select_dtypes(include=np.number).to_numpy()).all()

    PAPER_OUT.write_text("")  # Outputs are created only after all assertions pass.
    paper_table.to_csv(PAPER_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("CORAL per-paper diagnostic")
    print(f"  papers improved / unchanged / worsened vs uncorrected: {overall.iloc[0]['value']}")
    print(f"  size vs accuracy gain Spearman rho: {rho_acc:.3f} (p={p_acc:.3f})")
    print(f"  saved {PAPER_OUT.relative_to(ROOT)} and {SUMMARY_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
