"""Generate poster-ready figures from committed canonical result CSVs.

Run from the repository root:
    python scripts/10_poster_figures.py

The script reads only CSVs committed with the canonical analyses and writes
three 450-dpi PNGs to outputs/figures/. It never reruns a model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
RESULTS = ROOT / "results"
FIGURES = ROOT / "outputs" / "figures"
DPI = 450

NAVY = "#1F4E79"
TEAL = "#1B998B"
GOLD = "#E9B44C"
SLATE = "#5B6770"
LIGHT_GRAY = "#E8EDF1"


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)


def endpoint_figure() -> None:
    """Random versus paper-grouped accuracy for MIC and MBC."""
    mic = pd.read_csv(TABLES / "grouped_cv_mic.csv")
    mbc = pd.read_csv(TABLES / "04_mbc_summary.csv")
    mic_accuracy = mic.loc[mic["metric"].eq("accuracy")].iloc[0]
    mbc_random = mbc.loc[mbc["evaluation"].eq("random_stratified")].iloc[0]
    mbc_grouped = mbc.loc[mbc["evaluation"].eq("grouped_by_ref")].iloc[0]

    endpoints = {
        "MIC\n342 rows, 65 papers": {
            "values": [mic_accuracy["random_mean"], mic_accuracy["grouped_mean"]],
            "errors": [mic_accuracy["random_std"], mic_accuracy["grouped_std"]],
            "baseline": mic_accuracy["majority_baseline"],
        },
        "MBC\n133 rows, 24 papers": {
            "values": [mbc_random["accuracy_mean"], mbc_grouped["accuracy_mean"]],
            "errors": [mbc_random["accuracy_std"], mbc_grouped["accuracy_std"]],
            "baseline": mbc_grouped["cv_majority_accuracy_mean"],
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), sharey=True)
    labels = ["Random\nrow split", "Grouped\nby paper"]
    colors = [NAVY, TEAL]
    for ax, (endpoint, data) in zip(axes, endpoints.items()):
        x = np.arange(2)
        ax.bar(
            x,
            data["values"],
            yerr=data["errors"],
            capsize=5,
            color=colors,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.axhline(data["baseline"], color=GOLD, linestyle="--", linewidth=2)
        for xpos, value in zip(x, data["values"]):
            ax.text(xpos, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=11, weight="bold")
        ax.text(1.43, data["baseline"] + 0.012, f"majority\n{data['baseline']:.3f}", color=SLATE, fontsize=8.5, ha="right")
        ax.set_title(endpoint, fontsize=13, weight="bold", pad=10)
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_ylim(0, 0.9)
        style_axes(ax)
    axes[0].set_ylabel("Five-fold CV accuracy (mean ± fold SD)", fontsize=11)
    fig.suptitle("Random splits overstate cross-publication performance", fontsize=15, weight="bold", y=1.02)
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=NAVY), plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Line2D([0], [0], color=GOLD, linestyle="--", linewidth=2)],
        ["Random row split", "Paper-grouped split", "Majority-class baseline"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )
    save(fig, "figure_1_endpoint_random_vs_grouped.png")


def species_figure() -> None:
    """Grouped out-of-fold accuracy by species frequency stratum."""
    species = pd.read_csv(TABLES / "06_species_analysis_bucket_performance.csv")
    species = species.loc[species["population"].eq("all_rows")].copy()
    order = [">20", "5-20", "<5"]
    species["frequency_bucket"] = pd.Categorical(species["frequency_bucket"], categories=order, ordered=True)
    species = species.sort_values("frequency_bucket")
    labels = [
        f"{row.frequency_bucket} rows/species\n{row.n_species} species, n={row.n_rows}"
        for row in species.itertuples(index=False)
    ]
    x = np.arange(len(species))

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    model = ax.bar(x, species["accuracy"], 0.56, color=TEAL, label="Grouped out-of-fold accuracy")
    for bar in model:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.014, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 0.58)
    ax.set_title("Grouped OOF accuracy by species-frequency stratum", fontsize=15, weight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax)
    ax.text(
        0.01,
        -0.22,
        "Single paper-grouped MIC model; strata are evaluated from its out-of-fold predictions.",
        transform=ax.transAxes,
        fontsize=9,
        color=SLATE,
    )
    save(fig, "figure_2_species_frequency_strata.png")


def magpie_figure() -> None:
    """How often the 22 Magpie descriptors are constant within a paper."""
    constancy = pd.read_csv(RESULTS / "magpie_constancy.csv")
    ablation = pd.read_csv(RESULTS / "magpie_ablation_final.csv").set_index("config")
    all_constant = int(constancy["all_constant"].sum())
    total = len(constancy)
    variable = total - all_constant
    fraction = all_constant / total
    full = ablation.loc["full_features"]
    no_magpie = ablation.loc["no_magpie"]
    accuracy_gain = no_magpie["grouped_accuracy"] - full["grouped_accuracy"]
    macro_f1_gain = no_magpie["grouped_macro_f1"] - full["grouped_macro_f1"]

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.barh([0], [all_constant], color=NAVY, height=0.52, label="All 22 descriptors constant")
    ax.barh([0], [variable], left=[all_constant], color=GOLD, height=0.52, label="At least one descriptor varies")
    ax.text(all_constant / 2, 0, f"{all_constant} papers\n({fraction:.1%})", ha="center", va="center", color="white", fontsize=14, weight="bold")
    ax.text(all_constant + variable / 2, 0, f"{variable}\npapers", ha="center", va="center", color="#303030", fontsize=12, weight="bold")
    ax.set_xlim(0, total)
    ax.set_xticks(np.arange(0, total + 1, 10))
    ax.set_xlabel("Source papers (n = 65)", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Magpie composition descriptors are often paper-constant", fontsize=15, weight="bold", pad=12)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0,
        -0.72,
        "Within 55 of 65 source papers, all 22 composition-derived Magpie descriptors are identical across rows.",
        fontsize=10,
        color=SLATE,
    )
    ax.text(
        0,
        -0.99,
        f"Removing them raised grouped accuracy by {accuracy_gain:+.4f} and macro F1 by {macro_f1_gain:+.4f}; "
        f"accuracy remained below the {no_magpie['majority_baseline']:.4f} fold-trained majority baseline.",
        fontsize=9.5,
        color=SLATE,
    )
    save(fig, "figure_3_magpie_within_paper_constancy.png")


def main() -> None:
    endpoint_figure()
    species_figure()
    magpie_figure()
    print(f"Saved three {DPI}-dpi figures to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
