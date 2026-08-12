# Materials Feature Audit

The dataset contains 342 observations from 65 source papers and 22 Magpie descriptors. The Magpie features are composition-derived materials-informatics descriptors calculated from elemental composition using matminer. They describe properties such as mean and range of electronegativity, melting temperature, atomic radius, atomic volume, covalent radius, thermal conductivity, density, fusion enthalpy, periodic row, and periodic column.

These descriptors are primarily compositional proxies rather than mechanism-proximal nanoparticle descriptors. For example, the Magpie descriptors for ZnO encode information about the elemental composition of zinc and oxygen, but do not directly describe nanoparticle size, morphology, exposed crystal facets, surface charge, or synthesis conditions. In contrast, experimentally measured features such as particle size, shape, zeta potential, and synthesis method are more directly connected to mechanisms of nanoparticle-bacteria interaction.

The source-paper grouping required forward-filling the `Ref` column because references are recorded once at the beginning of each paper block. After forward-filling, there are 342 observations across 65 papers, with a mean of 5.26 observations per paper.

The within-paper Magpie analysis shows strong material-identity clustering. In 55 of 65 papers (84.62%), all 22 Magpie descriptors are constant across every observation. The remaining 10 papers all contain multiple material formulas and all show Magpie variation. Conversely, all 55 papers containing a single formula have constant values for all 22 Magpie descriptors. At the formula level, all 22 descriptors are constant across each of the 26 unique formulas.

This suggests that the Magpie block primarily encodes material composition, while material composition is often nearly constant within a source paper. Therefore, the Magpie descriptors may provide a model with a proxy for source-paper identity rather than independently generalizable mechanistic information.

I predict that removing the 22 Magpie descriptors will improve cross-paper generalization. This is a testable hypothesis that can be evaluated by comparing model performance with and without the Magpie feature block under cross-paper validation.

### Bouasla multi-material verification

The Bouasla source paper provides an explicit example of within-paper
material variation. Table 3 reports zone-of-inhibition and MIC results
for ZnO-TOP nanoparticles, while Table 4 reports corresponding results
for ZnS nanoparticles. Thus, the paper experimentally evaluates two
distinct nanoparticle compositions.

The corresponding dataset block contains 18 observations for this
source paper and includes the two formulas ZnS and ZnO. The source
tables confirm that these observations represent distinct materials,
with ZnO-TOP being the material name used in the paper. This provides
an independent check that the observed Magpie variation within this
paper is associated with genuine material/compositional variation
rather than an artifact of the reference grouping.
