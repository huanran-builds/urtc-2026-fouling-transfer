"""Freeze Edward's dataset-audit results into reproducible outputs.

Run from the repository root:

    python scripts/07_dataset_audit.py

The script writes ``results/magpie_constancy.csv`` and asserts every
structural value required by the close-out task sheet.  It does not fit a
model or perform a new analysis.
"""

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MIC_PATH = ROOT / "data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv"
MBC_PATH = ROOT / "data/external/chen2026_nanoparticles/Nanoparticles_MBC.csv"
ROUGHNESS_PATH = ROOT / "data/external/chen2024_nanostructured.csv"
OUTPUT_PATH = ROOT / "results/magpie_constancy.csv"

BOUASLA_DOI = "10.1002/cbdv.202400724"
ABSTRACT_ELIGIBLE_STUDIES = 98
KNOWN_REF_CORRECTIONS = {
    # The released CSV drops the final letter from this Scientific Reports DOI.
    "10.1038/s41598-021-85584-": "10.1038/s41598-021-85584-w",
}


def normalize_ref(value):
    """Return a lowercase DOI when possible, otherwise a normalized identifier."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    doi_match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        text,
        flags=re.IGNORECASE,
    )
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;").lower()
        return KNOWN_REF_CORRECTIONS.get(doi, doi)

    # One released reference is an RSC article URL whose article code is its
    # DOI suffix.  Convert it so every output row carries a DOI as requested.
    rsc_match = re.search(
        r"pubs\.rsc\.org/.*/articlelanding/\d{4}/[a-z]+/([a-z0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if rsc_match:
        return f"10.1039/{rsc_match.group(1).lower()}"

    return text.lower()


def load_endpoint(path):
    frame = pd.read_csv(path)
    frame["Ref"] = frame["Ref"].ffill()
    assert frame["Ref"].notna().all(), f"Unfilled Ref values remain in {path}"
    frame["ref"] = frame["Ref"].map(normalize_ref)
    assert frame["ref"].notna().all(), f"Unusable Ref values remain in {path}"
    return frame


def build_magpie_table(mic):
    magpie_columns = [
        column for column in mic.columns if column.startswith("MagpieData")
    ]
    assert len(magpie_columns) == 22, (
        f"Expected 22 Magpie descriptors, found {len(magpie_columns)}"
    )
    assert mic["Formula"].notna().all(), "Formula contains missing values"
    assert not mic[magpie_columns].isna().any().any(), (
        "At least one Magpie descriptor is missing"
    )

    rows = []
    for ref, group in mic.groupby("ref", sort=True):
        constant = group[magpie_columns].nunique(dropna=False).eq(1)
        rows.append(
            {
                "ref": ref,
                "n_rows": len(group),
                "n_unique_formulas": group["Formula"].nunique(dropna=False),
                "n_constant_magpie": int(constant.sum()),
                "all_constant": bool(constant.all()),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "ref",
            "n_rows",
            "n_unique_formulas",
            "n_constant_magpie",
            "all_constant",
        ],
    )

    assert len(result) == 65, f"Expected 65 references, found {len(result)}"
    assert result["n_rows"].sum() == 342, "Not all 342 MIC rows were counted"
    assert result["all_constant"].sum() == 55, (
        "Expected 55 references with all 22 descriptors constant"
    )
    assert result.loc[
        result["n_unique_formulas"].eq(1), "all_constant"
    ].all(), "A single-formula reference has varying Magpie descriptors"
    assert result.loc[
        ~result["all_constant"], "n_unique_formulas"
    ].gt(1).all(), "A varying Magpie reference does not contain multiple formulas"

    return result, magpie_columns


def verify_roughness_recount():
    roughness = pd.read_csv(ROUGHNESS_PATH).dropna(how="all")
    assert len(roughness) == 291, (
        f"Expected 291 nonblank Chen 2024 rows, found {len(roughness)}"
    )

    has_ra = roughness["Ra_nm"].notna()
    has_rq = roughness["Rq_nm"].notna()
    has_either = has_ra | has_rq

    counts = {
        "ra_only": int((has_ra & ~has_rq).sum()),
        "rq_only": int((~has_ra & has_rq).sum()),
        "both": int((has_ra & has_rq).sum()),
        "neither": int((~has_ra & ~has_rq).sum()),
        "present": int(has_either.sum()),
        "missing": int((~has_either).sum()),
    }
    assert counts == {
        "ra_only": 39,
        "rq_only": 29,
        "both": 22,
        "neither": 201,
        "present": 90,
        "missing": 201,
    }, f"Unexpected roughness recount: {counts}"

    # These are the four jointly available core variables used in the
    # close-out recount.  Spacing is not part of this four-variable subset.
    core_columns = ["diameter_nm", "height_nm", "aspect_ratio", "WCA_deg"]
    complete_core_and_roughness = (
        roughness[core_columns].notna().all(axis=1) & has_either
    )
    complete_rows = int(complete_core_and_roughness.sum())
    complete_refs = int(
        roughness.loc[complete_core_and_roughness, "ref"].nunique()
    )
    assert complete_rows == 70, (
        f"Expected 70 core-plus-roughness rows, found {complete_rows}"
    )
    assert complete_refs == 12, (
        f"Expected 12 contributing references, found {complete_refs}"
    )

    return counts, core_columns, complete_rows, complete_refs


def main():
    mic = load_endpoint(MIC_PATH)
    mbc = load_endpoint(MBC_PATH)

    assert len(mic) == 342
    assert len(mbc) == 133

    mic_refs = set(mic["ref"])
    mbc_refs = set(mbc["ref"])
    assert len(mic_refs) == 65
    assert len(mbc_refs) == 24
    assert mbc_refs <= mic_refs, "MBC contains a reference absent from MIC"

    study_gap = ABSTRACT_ELIGIBLE_STUDIES - len(mic_refs)
    assert study_gap == 33

    result, magpie_columns = build_magpie_table(mic)

    group_sizes = mic.groupby("ref").size()
    assert round(group_sizes.mean(), 1) == 5.3
    assert group_sizes.median() == 4
    assert group_sizes.max() == 18

    bouasla = mic[mic["ref"].eq(BOUASLA_DOI)]
    bouasla_formula_counts = bouasla.groupby("Formula").size().to_dict()
    assert len(bouasla) == 18
    assert bouasla_formula_counts == {"ZnO": 9, "ZnS": 9}

    roughness_counts, core_columns, complete_rows, complete_refs = (
        verify_roughness_recount()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("Edward dataset audit: PASS")
    print(f"MIC: {len(mic)} rows, {len(mic_refs)} references")
    print(f"MBC: {len(mbc)} rows, {len(mbc_refs)} references; complete MIC subset")
    print(f"Unaccounted eligible studies: {study_gap}")
    print(f"Magpie descriptors examined: {len(magpie_columns)}")
    print(
        "MIC rows/reference: "
        f"mean {group_sizes.mean():.1f}, median {group_sizes.median():.0f}, "
        f"max {group_sizes.max()}"
    )
    print(f"All 22 Magpie descriptors constant: {result['all_constant'].sum()}/65")
    print(f"Bouasla DOI rows: {len(bouasla)} ({bouasla_formula_counts})")
    print(
        "Roughness recount: "
        f"Ra only {roughness_counts['ra_only']}, "
        f"Rq only {roughness_counts['rq_only']}, "
        f"both {roughness_counts['both']}, "
        f"neither {roughness_counts['neither']}"
    )
    print(
        f"Core columns {core_columns} plus roughness: "
        f"{complete_rows} rows across {complete_refs} references"
    )
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
