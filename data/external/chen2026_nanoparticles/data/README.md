# Data files

| File | Used by |
|------|---------|
| `single_features_with_class.csv` | Main training set (RF, XGBoost, LightGBM, SVM, …) |
| `single_MBC_features.csv` | Older ElasticNet / kNN / Lasso / RF_shap scripts |
| `single_features.csv` | Older mlp / RF_outlier scripts |
| `enumerate_ag_ecoli.csv` | `enumerate_xgboost.py`, 3D plots |
| `enumerate_ag.csv` | Optional enumeration set |

All paths are resolved via `paths.py` at the repository root.
