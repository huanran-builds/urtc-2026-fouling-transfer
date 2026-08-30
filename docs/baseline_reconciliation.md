# Baseline reconciliation: MIC Magpie ablation

## Purpose

The initial Magpie ablation reported full-feature grouped accuracy of 0.3044,
whereas the project’s canonical `scripts/01_grouped_cv.py` result is 0.3162.
The absolute difference is 0.0118, within the project’s documented
cross-platform tolerance of plus or minus 0.02.  This note records the
configuration comparison and establishes one final paired ablation result.

## Configuration comparison before rerun

1. **Feature list passed to the model.** Both scripts construct the
   full-feature arm as 22 `MagpieData*` columns, the four numeric columns
   `size (nm)`, `zeta_binary`, `duration`, and `temperature`, and seven
   categorical columns. This is 26 numeric plus 7 categorical input columns.
   The ablated arm removes exactly the 22 Magpie columns, leaving 4 numeric
   plus the same 7 categorical columns.
2. **Source-paper identifiers.** Both scripts forward-fill `Ref` before
   constructing groups. The MIC dataset therefore has 342 rows, no missing
   group identifiers, and 65 source papers.
3. **Splitter.** Both use `GroupKFold(n_splits=5)` with forward-filled `Ref`
   as the grouping variable.
4. **Model.** Both use `XGBClassifier` with `n_estimators=300`,
   `max_depth=4`, `learning_rate=0.1`, `subsample=0.9`,
   `colsample_bytree=0.9`, `objective="multi:softprob"`, `num_class=3`,
   `random_state=42`, `n_jobs=1`, and `verbosity=0`.
5. **Preprocessing.** Both fit median numeric imputation and categorical
   constant imputation plus one-hot encoding inside each training fold. The
   full-feature ablation arm is uncorrected, matching the canonical grouped
   baseline; neither uses within-paper standardization.
6. **Row filtering.** Neither script filters MIC rows before fitting. Both
   use all 342 rows.

## Hypothesis

No configuration difference among the six pre-specified checks explains the
0.0118 accuracy gap. I predict that a matched rerun will remain within 0.02
of the canonical 0.3162 result, and that the residual difference reflects
cross-platform or library-version variation in XGBoost rather than a changed
feature set, grouping, split, or model specification.
