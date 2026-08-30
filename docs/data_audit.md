2026 paper:
98 peer-reviewed studies met the eligibility criteria.

Released MIC dataset:
342 rows, 65 unique normalized Ref values.

Released MBC dataset:
133 rows, 24 unique normalized Ref values.

Overlap:
24 references.

Combined:
65 unique normalized references.

Discrepancy:
98 - 65 = 33 studies.

Conclusion:
The published statement of 98 eligible studies cannot be
reconciled with the 65 unique source references represented
in the released MIC/MBC files using the available Ref fields.
The paper does not appear to explain this discrepancy in the
sections searched.

1b. ZnS and ZnO-TOP reference check — PASS
DOI: 10.1002/cbdv.202400724

The MIC dataset contains 18 rows associated with this DOI after
forward-filling the Ref column. We checked the original paper,
Bouasla et al. (2024), Chemistry & Biodiversity.

The paper contains separate MIC tables for both nanoparticle
materials:

Table 3: ZnO-TOP NPs
Table 4: ZnS NPs

Each table reports MIC values for the same 10 microorganisms:
9 bacterial strains and 1 fungus (C. albicans). The released dataset
contains 18 rows for this DOI, split evenly between the two materials,
but its nine rows per material contain 8 bacteria and C. albicans.
Therefore, the dataset does not reproduce all 10 table organisms and
the earlier shorthand "9 bacterial strains × 2" was inaccurate.

The paper nevertheless establishes that ZnS and ZnO-TOP were tested
within the same study. The shared DOI is therefore supported, and the
forward-fill of Ref did not merge two different studies.

Evidence: Bouasla et al. (2024), DOI 10.1002/cbdv.202400724,
Tables 3–4, pages 3–4.

1c. Chen et al. 2024 missingness claim — verified against source

Chen et al. 2024 (10.1186/s12951-024-02974-8 — a distinct paper on nanostructured surfaces) justifies dropping surface roughness by claiming the data had "approximately 90% or more missing values and was therefore excluded." Their own Table S5 does not support that figure: roughness is present in 90 of 291 rows, i.e. 69.1% missing, not ~90%.

This dataset's independently recomputed count matches Table S5 exactly (90/291 present, 201/291 missing), so the discrepancy is in Chen et al.'s stated figure, not in this audit's numbers. Section 4 below gives the fuller picture: even though the ~90% figure is wrong, the roughness data that is present is inconsistently reported (Ra-only/Rq-only/both split 39/29/22), which is a better-grounded reason to exclude the column than the missingness rate alone.
One addition worth making: only 12 unique references contribute the 70 rows with all four core features plus roughness. This is the subset in which the potential value of including roughness is actually testable, but 12 papers is too small to support a well-powered cross-paper comparison. Therefore, the audit cannot determine whether including roughness would have improved model performance. The defensible conclusion is not that roughness would not have helped, but that the available data are insufficient to answer that question reliably.
