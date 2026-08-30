import pandas as pd
import re
from pathlib import Path

# ============================================================
# 1. LOAD DATA
# ============================================================

folder = Path(__file__).parent

df = pd.read_csv(
    folder / "Nanoparticles_MIC_with_class.csv"
)

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. FILL SOURCE PAPER REFERENCES DOWN
# ============================================================
#
# IMPORTANT:
# The Ref column is only populated on the first row of each
# paper's block. The blank rows underneath belong to that paper.
#
# Therefore, forward-fill Ref BEFORE doing any paper-level
# grouping or variation analysis.
# ============================================================

missing_ref_before = df["Ref"].isna().sum()

df["Ref"] = df["Ref"].ffill()

missing_ref_after = df["Ref"].isna().sum()

print("\n" + "=" * 70)
print("SOURCE PAPER REFERENCE FILL-DOWN")
print("=" * 70)

print("Missing Ref values before ffill:", missing_ref_before)
print("Missing Ref values after ffill:", missing_ref_after)

if missing_ref_after > 0:
    print(
        "\nWARNING: Some rows still have missing Ref values "
        "after forward-fill."
    )


# ============================================================
# 3. IDENTIFY SOURCE PAPERS
# ============================================================

print("\n" + "=" * 70)
print("SOURCE PAPER GROUPING")
print("=" * 70)

num_papers = df["Ref"].nunique()

rows_per_paper = df.groupby("Ref").size()

print("Unique source papers:", num_papers)
print("Minimum rows/paper:", rows_per_paper.min())
print("Maximum rows/paper:", rows_per_paper.max())
print("Mean rows/paper:", round(rows_per_paper.mean(), 2))
print("Median rows/paper:", round(rows_per_paper.median(), 2))

print("\nRows per paper:")
print(rows_per_paper.to_string())

# Expected structure from the dataset audit:
#
# 342 rows
# 65 papers
# mean approximately 5.3 rows/paper
# maximum 18 rows/paper


# ============================================================
# 4. IDENTIFY MAGPIE FEATURES
# ============================================================

magpie_cols = [
    c for c in df.columns
    if c.startswith("MagpieData")
]

print("\n" + "=" * 70)
print("MAGPIE FEATURES")
print("=" * 70)

print("Number of Magpie columns:", len(magpie_cols))

for i, col in enumerate(magpie_cols, start=1):
    print(f"{i}. {col}")

if len(magpie_cols) != 22:
    print(
        "\nWARNING: Expected 22 Magpie columns, "
        f"but found {len(magpie_cols)}."
    )
else:
    print("\nConfirmed: 22 Magpie columns.")


# ============================================================
# 5. SHOW ALL DATASET COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ALL DATASET COLUMNS")
print("=" * 70)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")


# ============================================================
# 6. NORMALIZE REFERENCES FOR DISPLAY / DUPLICATE CHECKING
# ============================================================

def clean_ref(value):
    """
    Normalize a reference.

    Handles:
    - DOI URLs
    - Markdown links
    - 'DOI:' prefixes
    - whitespace
    - capitalization
    """

    if pd.isna(value):
        return "MISSING_REFERENCE"

    x = str(value).strip()

    # Remove Markdown backslashes
    x = x.replace("\\", "")

    # Extract DOI if present
    match = re.search(
        r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+',
        x,
        flags=re.IGNORECASE
    )

    if match:
        doi = match.group(0)

        # Remove punctuation accidentally captured at the end
        doi = doi.rstrip(").,;]}>")

        return doi.lower()

    # Otherwise normalize whitespace
    x = re.sub(r"\s+", " ", x).strip()

    return x


df["Ref_clean"] = df["Ref"].apply(clean_ref)


# ============================================================
# 7. REFERENCE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFERENCE SUMMARY")
print("=" * 70)

raw_unique = df["Ref"].nunique(dropna=False)
clean_unique = df["Ref_clean"].nunique(dropna=False)

print("Unique filled Ref values:", raw_unique)
print("Unique cleaned references:", clean_unique)

print("\nCleaned references and row counts:")

cleaned_counts = df["Ref_clean"].value_counts()

print(cleaned_counts.to_string())


# ============================================================
# 8. ROWS PER CLEANED PAPER
# ============================================================

print("\n" + "=" * 70)
print("ROWS PER CLEANED PAPER")
print("=" * 70)

print("Number of cleaned papers:", len(cleaned_counts))
print("Minimum rows/paper:", cleaned_counts.min())
print("Maximum rows/paper:", cleaned_counts.max())
print("Mean rows/paper:", round(cleaned_counts.mean(), 2))
print("Median rows/paper:", round(cleaned_counts.median(), 2))

print("\nSummary:")
print(cleaned_counts.describe())


# ============================================================
# 9. WITHIN-PAPER MAGPIE VARIATION
# ============================================================
#
# For each paper and each Magpie feature, calculate the number
# of unique values.
#
# If unique values == 1:
#     feature is constant within that paper.
#
# If unique values > 1:
#     feature varies within that paper.
# ============================================================

print("\n" + "=" * 70)
print("WITHIN-PAPER MAGPIE VARIATION")
print("=" * 70)

variation = (
    df.groupby("Ref_clean")[magpie_cols]
    .nunique(dropna=False)
)

constant_fraction = (variation == 1).mean()

print("\nFraction of papers where each Magpie feature is constant:")

print(
    constant_fraction
    .sort_values(ascending=False)
    .to_string()
)


# ============================================================
# 10. DETAILED MAGPIE FEATURE VARIATION
# ============================================================

print("\n" + "=" * 70)
print("MAGPIE FEATURE VARIATION SUMMARY")
print("=" * 70)

for col in magpie_cols:

    values = variation[col]

    print(
        f"{col}: "
        f"mean unique values/paper = {values.mean():.2f}, "
        f"median = {values.median():.0f}, "
        f"constant papers = {(values == 1).sum()}/{len(values)}, "
        f"varying papers = {(values > 1).sum()}/{len(values)}"
    )


# ============================================================
# 11. THE MAIN QUESTION:
#     HOW MANY PAPERS HAVE ALL 22 MAGPIE FEATURES CONSTANT?
# ============================================================
#
# This is the statistic atxtp specifically asked for.
#
# For every paper:
#
#     all 22 constant = every Magpie column has exactly
#                       one unique value
#
# This measures whether the Magpie block effectively behaves
# like a paper/material identifier.
# ============================================================

all_magpie_constant_per_paper = (
    variation[magpie_cols] == 1
).all(axis=1)

num_all_constant = all_magpie_constant_per_paper.sum()
num_varying = (~all_magpie_constant_per_paper).sum()

fraction_all_constant = (
    num_all_constant / len(all_magpie_constant_per_paper)
)

print("\n" + "=" * 70)
print("ALL-22 MAGPIE CONSTANCY BY PAPER")
print("=" * 70)

print(
    "Papers where ALL 22 Magpie features are constant:",
    f"{num_all_constant}/{len(all_magpie_constant_per_paper)}"
)

print(
    "Fraction of papers with all 22 constant:",
    f"{fraction_all_constant:.4f}"
)

print(
    "Percentage of papers with all 22 constant:",
    f"{fraction_all_constant * 100:.2f}%"
)

print(
    "Papers where at least one Magpie feature varies:",
    f"{num_varying}/{len(all_magpie_constant_per_paper)}"
)


# ============================================================
# 12. LIST PAPERS WITH MAGPIE VARIATION
# ============================================================

print("\n" + "=" * 70)
print("PAPERS WITH MAGPIE VARIATION")
print("=" * 70)

varying_papers = variation.loc[
    ~all_magpie_constant_per_paper
]

if len(varying_papers) == 0:

    print("NONE")

else:

    print(
        f"{len(varying_papers)} papers have at least "
        "one varying Magpie feature.\n"
    )

    for paper in varying_papers.index:

        print("\n" + "-" * 70)
        print("Paper:", paper)

        print("\nMagpie unique-value counts:")

        paper_variation = variation.loc[paper]

        print(
            paper_variation[
                paper_variation > 1
            ].to_string()
        )


# ============================================================
# 13. IDENTIFY PAPERS WITH MULTIPLE MATERIALS / FORMULAS
# ============================================================
#
# This is important because a paper testing multiple materials
# is where we EXPECT Magpie descriptors to change.
# ============================================================

print("\n" + "=" * 70)
print("MATERIAL / FORMULA DIVERSITY BY PAPER")
print("=" * 70)

for material_col in ["Material 1", "Formula"]:

    if material_col not in df.columns:
        continue

    counts = (
        df.groupby("Ref_clean")[material_col]
        .nunique(dropna=True)
    )

    print(f"\nColumn: {material_col}")

    print(
        "Mean unique values/paper:",
        round(counts.mean(), 2)
    )

    print(
        "Median unique values/paper:",
        round(counts.median(), 2)
    )

    print(
        "Minimum:",
        counts.min()
    )

    print(
        "Maximum:",
        counts.max()
    )

    print(
        "Papers with >1 unique value:",
        (counts > 1).sum()
    )


# ============================================================
# 14. DIRECTLY COMPARE MATERIAL DIVERSITY
#     WITH MAGPIE VARIATION
# ============================================================

print("\n" + "=" * 70)
print("MATERIAL DIVERSITY VS MAGPIE VARIATION")
print("=" * 70)

if "Formula" in df.columns:

    formula_counts = (
        df.groupby("Ref_clean")["Formula"]
        .nunique(dropna=True)
    )

    comparison = pd.DataFrame({
        "unique_formulas": formula_counts,
        "all_22_magpie_constant": all_magpie_constant_per_paper
    })

    print("\nPapers with multiple formulas:")

    print(
        comparison[
            comparison["unique_formulas"] > 1
        ].sort_values(
            "unique_formulas",
            ascending=False
        ).to_string()
    )

    print("\nCross-tabulation:")

    print(
        pd.crosstab(
            comparison["unique_formulas"] > 1,
            comparison["all_22_magpie_constant"],
            rownames=["Multiple formulas?"],
            colnames=["All 22 Magpie constant?"]
        )
    )


# ============================================================
# 15. INSPECT MULTI-MATERIAL PAPERS
# ============================================================
#
# For every paper containing multiple formulas, print:
# - paper reference
# - number of rows
# - formulas
# - whether all Magpie descriptors are constant
#
# This is the strongest sanity check for the hypothesis.
# ============================================================

print("\n" + "=" * 70)
print("MULTI-FORMULA PAPERS: DIRECT INSPECTION")
print("=" * 70)

if "Formula" in df.columns:

    formula_counts = (
        df.groupby("Ref_clean")["Formula"]
        .nunique(dropna=True)
    )

    multi_formula_papers = formula_counts[
        formula_counts > 1
    ].index

    print(
        "Number of papers with >1 formula:",
        len(multi_formula_papers)
    )

    for paper in multi_formula_papers:

        paper_df = df[
            df["Ref_clean"] == paper
        ]

        formulas = (
            paper_df["Formula"]
            .dropna()
            .unique()
            .tolist()
        )

        print("\n" + "-" * 70)
        print("Paper:", paper)
        print("Rows:", len(paper_df))
        print("Number of formulas:", len(formulas))
        print("Formulas:", formulas)

        print(
            "All 22 Magpie constant:",
            bool(all_magpie_constant_per_paper.loc[paper])
        )

        # Show Magpie unique-value counts for this paper
        print("\nMagpie variation:")

        print(
            variation.loc[paper]
            .to_string()
        )


# ============================================================
# 16. MAGPIE VARIATION BY FORMULA
# ============================================================
#
# A formula should map to one set of Magpie descriptors.
# If this is true, it supports the interpretation that Magpie
# descriptors are essentially encoding material composition.
# ============================================================

print("\n" + "=" * 70)
print("MAGPIE VARIATION BY FORMULA")
print("=" * 70)

if "Formula" in df.columns:

    print(
        "Unique formulas:",
        df["Formula"].nunique()
    )

    formula_variation = (
        df.groupby("Formula")[magpie_cols]
        .nunique(dropna=False)
    )

    print("\nFraction of formulas where each Magpie feature is constant:")

    for col in magpie_cols:

        fraction_constant = (
            formula_variation[col] == 1
        ).mean()

        print(
            f"{col}: "
            f"{fraction_constant:.4f}"
        )


# ============================================================
# 17. MATERIAL / COMPOSITION COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("POSSIBLE MATERIAL / COMPOSITION COLUMNS")
print("=" * 70)

keywords = [
    "material",
    "composition",
    "element",
    "formula",
    "particle",
    "nanoparticle",
    "chemical"
]

possible_columns = []

for col in df.columns:

    if any(
        keyword in col.lower()
        for keyword in keywords
    ):
        possible_columns.append(col)

for col in possible_columns:
    print(col)

if not possible_columns:
    print("No obvious material/composition column found.")


# ============================================================
# 18. MATERIAL DIVERSITY
# ============================================================

print("\n" + "=" * 70)
print("MATERIAL / FORMULA DIVERSITY")
print("=" * 70)

for material_col in ["Material 1", "Formula"]:

    if material_col not in df.columns:
        continue

    material_counts = (
        df.groupby("Ref_clean")[material_col]
        .nunique(dropna=True)
    )

    print(f"\nColumn: {material_col}")

    print(
        "Mean unique values per paper:",
        round(material_counts.mean(), 2)
    )

    print(
        "Median unique values per paper:",
        round(material_counts.median(), 2)
    )

    print(
        "Minimum:",
        material_counts.min()
    )

    print(
        "Maximum:",
        material_counts.max()
    )

    print(
        "Papers with multiple values:",
        (material_counts > 1).sum()
    )


# ============================================================
# 19. BOUSLA-STYLE SANITY CHECK
# ============================================================
#
# If the Bouasla paper is present, find it by DOI/title-related
# text and print its rows.
#
# Replace the search string if your copy uses a different DOI.
# ============================================================

print("\n" + "=" * 70)
print("BOUASLA SANITY CHECK")
print("=" * 70)

bouasla_mask = df["Ref_clean"].astype(str).str.contains(
    "10.1016",
    case=False,
    na=False
)

# Print candidate papers with multiple formulas.
# This avoids hard-coding a DOI we have not verified here.

if "Formula" in df.columns:

    formula_counts = (
        df.groupby("Ref_clean")["Formula"]
        .nunique(dropna=True)
    )

    candidate_papers = formula_counts[
        formula_counts > 1
    ].index

    print(
        "Candidate multi-material papers to inspect:",
        len(candidate_papers)
    )

    for paper in candidate_papers:

        paper_df = df[
            df["Ref_clean"] == paper
        ]

        print("\nPaper:", paper)
        print(
            "Rows:",
            len(paper_df)
        )
        print(
            "Formulas:",
            paper_df["Formula"]
            .dropna()
            .unique()
            .tolist()
        )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Dataset rows:", len(df))
print("Dataset columns:", len(df.columns))
print("Magpie features:", len(magpie_cols))

print(
    "Source papers after ffill:",
    df["Ref"].nunique()
)

print(
    "Mean rows per paper:",
    round(rows_per_paper.mean(), 2)
)

print(
    "Maximum rows per paper:",
    rows_per_paper.max()
)

print(
    "Papers with all 22 Magpie features constant:",
    f"{num_all_constant}/{len(all_magpie_constant_per_paper)}"
)

print(
    "Fraction with all 22 Magpie features constant:",
    f"{fraction_all_constant:.4f}"
)

print(
    "Percentage with all 22 Magpie features constant:",
    f"{fraction_all_constant * 100:.2f}%"
)

print(
    "Papers with at least one varying Magpie feature:",
    f"{num_varying}/{len(all_magpie_constant_per_paper)}"
)

print(
    "Papers with multiple formulas:",
    (
        (formula_counts > 1).sum()
        if "Formula" in df.columns
        else "N/A"
    )
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)