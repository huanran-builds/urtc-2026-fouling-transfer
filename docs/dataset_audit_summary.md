# Dataset Audit Summary

- **Study count discrepancy.** `scripts/07_dataset_audit.py`, using `data/external/chen2026_nanoparticles/Nanoparticles_MIC_with_class.csv`, finds 65 released source references against the 98 eligible studies reported in the 2026 abstract, leaving 33 studies unaccounted for; the paper does not explain the gap.
- **MBC subset relation.** `scripts/07_dataset_audit.py` normalizes and compares the `Ref` fields in `Nanoparticles_MIC_with_class.csv` and `Nanoparticles_MBC.csv`, confirming that all 24 MBC references are contained within the 65 MIC references.
- **Roughness recount.** `scripts/07_dataset_audit.py`, using `data/external/chen2024_nanostructured.csv`, finds roughness in 90 of 291 rows (69.1% missing): Ra only 39, Rq only 29, both 22, and neither 201; 70 rows contain diameter, height, aspect ratio, WCA, and at least one roughness measure, across 12 references. Only 22 rows report both Ra and Rq, so the available roughness block lacks a single consistently reported measure; that inconsistency is a stronger reason for excluding it than the paper's unsupported approximately-90%-missing claim.
- **Forward-fill verification.** `scripts/07_dataset_audit.py` finds 18 rows under DOI `10.1002/cbdv.202400724`, split evenly between ZnS and ZnO formulas; the Wiley article confirms that one study tested both ZnS and ZnO-TOP, so forward-filling did not merge two studies. The released dataset has nine organisms per material, one of which is *Candida albicans* (a fungus), so the close-out sheet's wording "nine bacterial strains x two materials" is not literally correct.

## Source checks

- Chen et al. (2026), *A supervised learning pipeline for decoding nanoparticle antibacterial activity*, Cell Reports Physical Science 7, 103411. DOI: `10.1016/j.xcrp.2026.103411`.
- Bouasla et al. (2024), *Antimicrobial Activity of ZnS and ZnO-TOP Nanoparticles Againts Pathogenic Bacteria*, Chemistry & Biodiversity 21(12), e202400724. DOI: `10.1002/cbdv.202400724`.
- Chen et al. (2024), *A supervised machine learning tool to predict the bactericidal efficiency of nanostructured surface*, Journal of Nanobiotechnology 22, 748. DOI: `10.1186/s12951-024-02974-8`.
