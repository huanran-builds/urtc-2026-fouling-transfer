# Cross-Study Generalization in Literature-Curated Nanoparticle Antibacterial Prediction

IEEE MIT Undergraduate Research Technology Conference (URTC) 2026
Track 01, Technology OF Engineering | Subtrack: Nanotechnology

## What this is

Machine learning datasets in materials science are often assembled by
extracting results from many published papers into one table. Rows from the
same source paper share a lab, a protocol, and an assay, so a random
train/test split places interdependent rows on both sides and inflates
reported accuracy.

We test whether unlabeled target-study adaptation can improve prediction for
a source publication the model has never seen. The held-out publication
contributes input features but no labels.

## Data

Chen et al. (2026), Cell Reports Physical Science 7, 103411.
342 nanoparticle MIC records curated from 65 source papers, plus 133 MBC
records from 24 papers. Openly released at
[YaxiiC/Nanoparticle-Antibacterial-Dataset](https://github.com/YaxiiC/Nanoparticle-Antibacterial-Dataset).

## Canonical results (MIC, XGBoost, 5-fold; ARM macOS)

| Evaluation | Accuracy | Macro F1 |
| --- | --- | --- |
| Random split (their approach) | 0.728 | 0.708 |
| Grouped by source paper | 0.316 | 0.273 |
| Grouped + within-paper standardization | 0.345 | 0.306 |
| Majority-class baseline | 0.418 | |

A single shuffled-label control yielded 0.315 random and 0.406 grouped
accuracy, versus a 0.418 majority-class baseline. Repeated permutations
would be required for an inferential leakage test.

These headline values are the designated canonical ARM macOS results. A
human-owned rerun is required to replace the noncanonical generated files
introduced by PR #6; values produced there under scikit-learn 1.6.1 and
XGBoost 3.2.0 must not be cited.

## Related work

See [`docs/literature.md`](docs/literature.md) for the canonical related-work
review and [`docs/related_work_paragraph.md`](docs/related_work_paragraph.md)
for the 147-word poster-ready synthesis.

## Reproduce

```text
pip install -r requirements.txt
python scripts/01_grouped_cv.py  # random vs grouped vs standardized
python scripts/02_combat.py      # ComBat compatibility check
python scripts/03_permutation.py # shuffled-label control
```

Modeling scripts set explicit random seeds locally where applicable, and
XGBoost runs with `n_jobs=1`. Generated tables are written to
`outputs/tables/`.

## Authors

Henry (first author), Ezra, Stephen, Edward, Matthew

## Limitations

Literature-derived convenience sample subject to publication bias. All
relationships reported are associational. The MBC analysis reuses a subset of
the MIC source-publication corpus and is not an independent external
replication.

## Reproducibility note

The canonical reporting environment is Python 3.13 on ARM macOS using the
package versions pinned in the root `requirements.txt`, including XGBoost
3.4.0, scikit-learn 1.9.0, NumPy 2.4.6, and pandas 3.0.5. XGBoost results may
differ slightly across CPU architectures even with `random_state=42` and
`n_jobs=1`; cross-platform verification therefore uses an absolute tolerance
of 0.02 per metric. Development runs produced under other package versions
are noncanonical and must not be cited.
