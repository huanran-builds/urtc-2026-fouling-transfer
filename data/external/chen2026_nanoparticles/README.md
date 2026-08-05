# MIC Class Prediction (Nanoparticle Features)

Machine-learning pipelines to classify minimum inhibitory concentration (MIC) strength (`strong` / `moderate` / `weak`) from nanoparticle composition and morphology features.

## Repository layout

```

├── data/                    # Training & enumeration CSVs (see below)
├── outputs/                 # Generated figures (gitignored by default)
├── paths.py                 # Shared data/output paths
├── requirements.txt
└── scripts/
    ├── models/              # Classifier training (RF, XGBoost, LightGBM, …)
    ├── shap/                # SHAP explainability plots
    ├── analysis/            # Outlier checks, candidate enumeration
    └── visualization/       # 3D MIC-class scatter plots
```



## Run examples

From the repo root

```bash
# Random forest + SHAP
python scripts/models/RF.py

# XGBoost (Oct 2024 tuning pipeline)
python scripts/models/XGBoost1004.py

# PyTorch MLP (file RF1004.py — not random forest)
python scripts/models/RF1004.py

# 3D visualization
python scripts/visualization/plot_3d_mic_classes.py enumerate_ag_ecoli.csv -o outputs/mic_3d.png
```
