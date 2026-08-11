# Method compatibility: DEBIAS-M and CORAL

DEBIAS-M is not appropriate for the full nanoparticle feature table. Its
published implementation expects a numeric matrix with an integer batch ID in
the first column. It treats all remaining columns as non-negative, compositional
abundance features: after multiplying them by learned batch-specific weights,
it L1-normalizes every row so the feature values sum to one. It also fits its
own linear classifier rather than serving as a general preprocessing transform.

Those structural assumptions fit microbial taxon-count or relative-abundance
data. They do not fit a heterogeneous materials table containing continuous
measurements on unrelated scales (for example size, duration, temperature, and
elemental descriptors) plus one-hot encoded categorical variables (particle
shape, bacterial species, and organism traits). L1-normalizing that combined
table would make each value a fraction of an arbitrary total and would change
the scientific meaning of both measurements and categorical indicators. A
numeric-only surrogate would also not be comparable with the established
full-feature XGBoost baseline.

This is a method-structure mismatch, not an implementation error or an
empirical failure of DEBIAS-M. It is documented alongside the existing ComBat
limitation: methods developed for genomics may require feature structures that
are absent from literature-extracted materials datasets.

CORAL (CORrelation ALignment) is the appropriate tabular comparison. Within
each grouped cross-validation fold, features are imputed and encoded from the
training papers only. CORAL uses the covariance of an unlabeled held-out
paper's encoded features to align the training covariance to that target
covariance. The held-out labels are not passed to preprocessing, CORAL, or
model fitting. The classifier is then trained on the aligned source features
and predicts the held-out paper. Papers with one row have no estimable
covariance, so the implementation explicitly uses the identity transform for
them rather than pretending that a covariance estimate exists.
