# Cross-Domain Transferability of Nanoscale Surface Descriptors

IEEE MIT Undergraduate Research Technology Conference (URTC) 2026

Track 01, Technology OF Engineering | Subtrack: Nanotechnology

## What this is

A harmonized dataset and transfer benchmark testing whether nanoscale surface

descriptors that predict resistance to one kind of biological fouling also

predict resistance to others, across three domains:

- D1  protein adsorption and blood contact

- D2  clinical bacterial attachment and biofilm

- D3  environmental and engineered water biofilm

Outcomes are harmonized as percent fouling reduction against each source

study's own internal control, which makes records comparable across

incompatible assay units.

## Authors

Henry (first author), Ezra, Stephen, Edward

## How to reproduce

pip install -r requirements.txt
python scripts/01_harmonize.py # raw extraction -> harmonized CSVs
python scripts/02_merge.py # harmonized CSVs -> data/processed/master.csv
python scripts/03_within_domain.py # within-domain models
python scripts/04_transfer.py # LODO + pairwise transfer matrix
python scripts/05_shap_invariance.py # descriptor invariance
python scripts/06_coverage.py # descriptor reporting audit
python scripts/07_figures.py # all figures

Random seed fixed at 42 in `config.py`. All outputs land in `outputs/`.

## Data provenance

Every row is extracted from published literature. Screening logs in

`data/screening/` record every paper considered, including rejections and the

reason for each. `data/processed/master.csv` is generated and must never be

edited by hand.

## Limitations

Literature-derived convenience sample subject to publication bias. All

relationships reported are associational.

