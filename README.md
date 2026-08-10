# Cross-Study Generalization in Literature-Curated Nanoparticle Antibacterial Prediction

IEEE MIT Undergraduate Research Technology Conference (URTC) 2026
Track 01, Technology OF Engineering | Subtrack: Nanotechnology

## What this is

Machine learning datasets in materials science are often assembled by
extracting results from many published papers into one table. Rows from the
same source paper share a lab, a protocol, and an assay, so a random
train/test split places interdependent rows on both sides and inflates
reported accuracy.

We test whether semi-supervised batch correction can recover predictive
performance on a source paper the model has never seen, using that paper's
input features but never its labels.

## Data

Chen et al. (2026), Cell Reports Physical Science 7, 103411.
342 nanoparticle MIC records curated from 65 source papers, plus 133 MBC
records from 24 papers. Openly released at
github.com/YaxiiC/Nanoparticle-Antibacterial-Dataset

## Current results (MIC, XGBoost, 5-fold)

| Evaluation | Accuracy | Macro F1 |
| --- | --- | --- |
| Random split (their approach) | 0.728 | 0.708 |
| Grouped by source paper | 0.316 | 0.273 |
| Grouped + within-paper standardization | 0.345 | 0.306 |
| Majority-class baseline | 0.418 | |

Permutation control with shuffled labels gives 0.315 random and 0.406
grouped, both near the majority baseline, confirming no leakage beyond
study identity.

## Related work

- Li et al. (2026), Green Chemistry 28:9804. Showed random splits inflate
  accuracy on literature-aggregated chemistry data. Did not attempt correction.
- Xu et al. (2026), npj Computational Materials 12:191. Applied DOI-level
  batch control and benchmarked ComBat on literature-extracted catalysis data,
  for causal inference rather than unseen-study prediction.
- Austin et al. (2025), Nature Microbiology 10:897. DEBIAS-M, unlabeled
  target-study adaptation on pooled microbiome datasets.

## Reproduce

pip install -r requirements.txt
python scripts/01_grouped_cv.py # random vs grouped vs standardized
python scripts/02_combat.py # ComBat comparison
python scripts/03_permutation.py # shuffled-label control


Seed fixed at 42 in `config.py`. Outputs land in `outputs/`.

## Authors

Henry (first author), Ezra, Stephen, Edward, Matthew

## Limitations

Literature-derived convenience sample subject to publication bias. All
relationships reported are associational. Files in `data/processed/` are
generated and must not be edited by hand.
