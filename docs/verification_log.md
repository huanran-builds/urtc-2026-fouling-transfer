# Z2/Z3 verification log — 2026-09-02

## Verdict

**FAIL pending correction of `results/canonical_results.csv`.**

The fresh-environment recomputation and both input-integrity checks pass. The
canonical table does not yet pass the hostile traceability read: four standard
deviations disagree with the committed outputs produced by their named scripts,
and four dataset-count rows lack script, commit, and platform provenance.

No Windows-regenerated output is proposed for commit. ARM macOS remains the
project's canonical reporting platform.

## Frozen state and environment

- Expected analysis snapshot from the handoff: `2600365de5334813198f6b1101e830d299e0c977`
- Audited `main`: `3bee08030382078de6292a4dbd8fee6e2231d354`
- Reason for the extra commit: `3bee080` adds the final canonical-results table
  that Z3 requires; `2600365` is its parent and remains the snapshot cited by
  the populated table rows.
- Fresh clone: separate from all prior working copies
- Python: 3.13.14
- NumPy: 2.4.6
- pandas: 3.0.5
- SciPy: 1.18.0
- scikit-learn: 1.9.0
- XGBoost: 3.4.0
- `pip check`: no broken requirements
- Root `requirements.txt` SHA-256:
  `96c034219c0fc25c542cf7393e81f24ce8feb474f64540c41fb4ffeb0a17a1ce`

## Z2 — data-integrity and recomputation checks

The two input hashes exactly match Ezra's pre-merge reference hashes:

| Input | SHA-256 | Result |
| --- | --- | --- |
| `Nanoparticles_MIC_with_class.csv` | `d7bb60ab6bb6388c59f76de3d724462749a0ff8aefefc44385bed6c20807464c` | PASS |
| `Nanoparticles_MBC.csv` | `a5d7a86865d67e406f31ad1c618cf4940467aa3acf785a2d81e6a7b250ca96c6` | PASS |

Fresh Windows reruns used the unmodified scripts on `main`. Cross-platform
acceptance uses the project's predeclared absolute tolerance of 0.02 per
metric.

| Quantity | Windows recomputation | Canonical ARM | Difference (Windows − ARM) | Result |
| --- | ---: | ---: | ---: | --- |
| MIC grouped accuracy | 0.3044 | 0.3162 | −0.0118 | PASS |
| MIC majority baseline | 0.4181 | 0.4181 | 0.0000 | PASS |
| MBC grouped accuracy | 0.4900 | 0.5051 | −0.0151 | PASS |

The MIC rerun read 342 rows from 65 source papers. The MBC rerun read and
analyzed all 133 rows from 24 source papers, dropped zero rows, and confirmed
zero source-paper overlap in every grouped fold.

## Z3 — hostile read of all 28 canonical rows

Global structural checks:

- Exactly 28 data rows: PASS.
- All 28 metric identifiers are nonblank and unique: PASS.
- Every populated script path exists both at audited `main` and snapshot
  `2600365`: PASS.
- Populated SHA `2600365` resolves to a commit and is an ancestor of audited
  `main`: PASS.
- Four rows have no script/SHA/platform to check: FAIL.
- One estimate is represented twice under different identifiers: review
  required. `mic_coral_accuracy` and `coral_ridge_sweep_accuracy_max` are both
  the ridge-1.0 CORAL accuracy of 0.3987. The latter is a derived range endpoint,
  not an independent result.

The row-level check compares the canonical table with the committed artifact
produced by the named script. “PASS” means the reported value and any populated
standard deviation match after four-decimal rounding.

| # | Metric | Named source artifact | Result |
| ---: | --- | --- | --- |
| 1 | `mic_random_accuracy` | `outputs/tables/grouped_cv_mic.csv` | PASS |
| 2 | `mic_random_macro_f1` | `outputs/tables/grouped_cv_mic.csv` | PASS |
| 3 | `mic_grouped_accuracy` | `outputs/tables/grouped_cv_mic.csv` | PASS |
| 4 | `mic_grouped_macro_f1` | `outputs/tables/grouped_cv_mic.csv` | PASS |
| 5 | `mic_majority_baseline` | `outputs/tables/grouped_cv_mic.csv` | PASS |
| 6 | `mic_standardized_accuracy` | `outputs/tables/adaptation_comparison.csv` | PASS |
| 7 | `mic_standardized_macro_f1` | `outputs/tables/adaptation_comparison.csv` | PASS |
| 8 | `mic_coral_accuracy` | `outputs/tables/adaptation_comparison.csv` | **FAIL: std 0.1794; source rounds to 0.1789** |
| 9 | `mic_coral_macro_f1` | `outputs/tables/adaptation_comparison.csv` | **FAIL: std 0.1440; source rounds to 0.1439** |
| 10 | `mic_no_magpie_accuracy` | `outputs/tables/magpie_ablation_summary.csv` | **FAIL: std 0.1258; source is 0.1126** |
| 11 | `mic_no_magpie_macro_f1` | `outputs/tables/magpie_ablation_summary.csv` | **FAIL: std 0.1372; source is 0.1227** |
| 12 | `magpie_ablation_delta_accuracy` | Difference of paired grouped-CV means | PASS: 0.0439898 → 0.0440 |
| 13 | `mbc_random_accuracy` | `outputs/tables/04_mbc_summary.csv` | PASS |
| 14 | `mbc_random_macro_f1` | `outputs/tables/04_mbc_summary.csv` | PASS |
| 15 | `mbc_grouped_accuracy` | `outputs/tables/04_mbc_summary.csv` | PASS |
| 16 | `mbc_grouped_macro_f1` | `outputs/tables/04_mbc_summary.csv` | PASS |
| 17 | `mbc_majority_baseline` | `outputs/tables/04_mbc_summary.csv` | PASS |
| 18 | `species_delta_gt20` | `outputs/tables/06_species_analysis_bucket_performance.csv` | PASS |
| 19 | `species_delta_5to20` | `outputs/tables/06_species_analysis_bucket_performance.csv` | PASS |
| 20 | `species_delta_lt5` | `outputs/tables/06_species_analysis_bucket_performance.csv` | PASS |
| 21 | `coral_ridge_sweep_accuracy_min` | `outputs/tables/coral_ridge_sensitivity.csv` | PASS |
| 22 | `coral_ridge_sweep_accuracy_max` | `outputs/tables/coral_ridge_sensitivity.csv` | PASS; duplicate underlying ridge-1 estimate noted above |
| 23 | `combat_status` | `outputs/tables/combat_comparison.csv` | PASS as prose status; warning: blank machine-readable value |
| 24 | `magpie_constancy_papers` | `results/magpie_constancy.csv` | PASS: 55 of 65 references |
| 25 | `mic_n_rows` | No script/SHA/platform named | **FAIL provenance; count 342 is correct** |
| 26 | `mic_n_papers` | No script/SHA/platform named | **FAIL provenance; count 65 is correct** |
| 27 | `mbc_n_rows` | No script/SHA/platform named | **FAIL provenance; count 133 is correct** |
| 28 | `mbc_n_papers` | No script/SHA/platform named | **FAIL provenance; count 24 is correct** |

### Corrections required before Z3 can pass

1. Replace the four stale/misrounded standard deviations:
   - CORAL accuracy: `0.1789`
   - CORAL macro-F1: `0.1439`
   - no-Magpie accuracy: `0.1126`
   - no-Magpie macro-F1: `0.1227`
2. Add provenance to all four count rows. MIC counts are produced by
   `scripts/01_grouped_cv.py`; MBC counts are produced by `scripts/04_mbc.py`.
   Record an explicit commit and platform designation.
3. Make the CORAL sweep-maximum row explicitly a derived sensitivity-range
   endpoint, or remove it, so the same ridge-1 estimate cannot be interpreted
   as two independent results.
4. Prefer a categorical status field/value for `combat_status` if this table
   is intended for machine consumption.

## CORAL interpretation

The ±0.02 threshold is an engineering acceptance rule for cross-platform
reproduction. It is not a confidence interval, standard error, equivalence
margin, or scientific uncertainty bound. Therefore, the fact that the CORAL
point estimate is 0.019 below the majority baseline does not establish parity
with the baseline.

Recommended wording:

> Using each held-out publication's features but not its labels, CORAL raised
> mean grouped accuracy from 0.316 to 0.399, but remained below the 0.418
> majority-class baseline and varied markedly across folds (0.087–0.618;
> SD 0.179).

This reports the observed improvement without implying that CORAL reliably
recovered cross-publication generalization. The fold SD describes dispersion
across five folds, not uncertainty in the mean.

## Adjacent documentation and output warnings

- `README.md` still says a human-owned canonical rerun is required, although
  that rerun landed in `2600365`.
- `docs/baseline_reconciliation.md` likewise retains future-tense “rerun
  pending” language. Its retired numbers are clearly labeled as noncanonical,
  but the status text is stale.
- `scripts/07_coral_diagnostics.py` outputs predate the final canonical
  adaptation OOF predictions. Do not use its per-paper diagnostic counts or
  correlations until a human-owned rerun refreshes those artifacts.

## Gate status

- Z2: **PASS**
- Z3: **FAIL pending the corrections above**
- Overall verification gate: **FAIL**

After the canonical table is corrected, repeat the row-level comparison and
update this log rather than changing the present failure record silently.
