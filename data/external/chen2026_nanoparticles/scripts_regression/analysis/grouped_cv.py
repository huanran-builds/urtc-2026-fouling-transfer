"""
Grouped cross-validation experiments -- reviewer response.

Addresses reviewer concern about R^2 inflation from Ag-dominant dataset.

Experiments
-----------
  Exp 1  GroupKFold  (5-fold, groups = Formula)
  Exp 2  Leave-One-Material-Out  (materials with n >= MIN_SAMPLES_LOMO)
  Exp 3  Leave-Ag-Out  (train on non-Ag, test on Ag)
  Exp 4  Material-Family GroupKFold  (families: pure_metal / metal_oxide /
         chalcogenide / alloy / carbon)
  Exp 5  Downsampled-Ag standard CV  (Ag capped to match 2nd-largest material,
         repeated over multiple seeds)

All targets are log1p-transformed during training; metrics are reported on the
original scale after expm1 inverse.

Datasets:  MIC  (single_features.csv)   &   MBC  (single_MBC_features.csv)
Models:    RF, XGBoost, XGB_Huber, SVM, KNN, Lasso, ElasticNet

Results saved to  outputs/results_grouped_cv/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths import MIC_DATA, MBC_DATA, GROUPED_CV_DIR

import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = GROUPED_CV_DIR

MIN_SAMPLES_LOMO = 5
N_SPLITS_GKF = 5
INNER_CV = 5
DOWNSAMPLE_SEEDS = [0, 1, 2, 3, 7, 42, 99, 123, 456, 2025]
OUTLIER_MEDIAN_FACTOR = 50

DATASETS = {
    "MIC": {"file": MIC_DATA, "target": "MIC (µg/mL)"},
    "MBC": {"file": MBC_DATA, "target": "MBC (µg/mL)"},
}

ID_COLS = ["No.", "Ref", "Material 1", "Formula", "bacteria", "MIC_class"]

MODELS = {
    "RF": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            objective="reg:squarederror", random_state=42,
            n_jobs=-1, verbosity=0,
        ),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
    },
    "SVM": {
        "estimator": SVR(),
        "param_grid": {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 1],
            "gamma": ["scale", "auto"],
        },
    },
    "KNN": {
        "estimator": KNeighborsRegressor(),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    },
    "Lasso": {
        "estimator": Lasso(max_iter=10000),
        "param_grid": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        },
    },
    "ElasticNet": {
        "estimator": ElasticNet(max_iter=100000),
        "param_grid": {
            "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0.1, 0.25, 0.5, 0.75, 1.0],
        },
    },
    "XGB_Huber": {
        "estimator": XGBRegressor(
            objective="reg:pseudohubererror", huber_slope=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
    },
}

FAMILY_MAP = {
    "Ag": "pure_metal", "Au": "pure_metal", "Cu": "pure_metal",
    "Se": "pure_metal", "Co": "pure_metal", "Zn": "pure_metal",
    "Te": "pure_metal",
    "ZnO": "metal_oxide", "TiO2": "metal_oxide", "CeO2": "metal_oxide",
    "CuO": "metal_oxide", "MgO": "metal_oxide", "AlO": "metal_oxide",
    "FeO": "metal_oxide", "Ag2O2": "metal_oxide", "Cu2O": "metal_oxide",
    "MgOCe": "metal_oxide", "MgFe2O4": "metal_oxide",
    "ZnS": "chalcogenide", "MoS2": "chalcogenide",
    "CuSe": "chalcogenide", "AgSe": "chalcogenide",
    "AuPt": "alloy", "AuPtCu": "alloy", "AgCu": "alloy",
    "C": "carbon",
}


def formula_to_family(formula):
    return FAMILY_MAP.get(formula, "other")


def detect_extreme_materials(y, groups, factor=OUTLIER_MEDIAN_FACTOR):
    overall_median = np.median(y)
    threshold = factor * overall_median
    unique_mats = np.unique(groups)
    return [mat for mat in unique_mats if np.median(y[groups == mat]) > threshold]


def filter_dataset(X, y, groups, excluded_materials):
    mask = ~np.isin(groups, excluded_materials)
    return (
        X.loc[mask].reset_index(drop=True),
        y.loc[mask].reset_index(drop=True),
        groups[mask],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(cfg):
    df = pd.read_csv(cfg["file"], encoding="latin1")
    df.columns = df.columns.str.strip()
    groups = df["Formula"].astype(str).str.strip().values
    y = pd.to_numeric(df[cfg["target"]], errors="coerce")
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    groups = groups[valid.values]
    drop = [c for c in ID_COLS + [cfg["target"]] if c in df.columns]
    X = df.drop(columns=drop, errors="ignore").reset_index(drop=True)
    if "size (nm)" in X.columns:
        X["size (nm)"] = pd.to_numeric(X["size (nm)"], errors="coerce")
    return X, y, groups


def build_transformers(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    cat_pipe.fit(X[cat_cols])
    return cat_cols, num_cols, cat_pipe


def transform_split(X_train, X_test, cat_cols, num_cols, cat_pipe):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    num_pipe.fit(X_train[num_cols])
    Xtr = hstack([num_pipe.transform(X_train[num_cols]),
                  cat_pipe.transform(X_train[cat_cols])])
    Xte = hstack([num_pipe.transform(X_test[num_cols]),
                  cat_pipe.transform(X_test[cat_cols])])
    return Xtr, Xte


def compute_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def fit_and_predict(Xtr, y_train_orig, Xte, model_cfg):
    y_train_log = np.log1p(y_train_orig)
    est = clone(model_cfg["estimator"])
    grid = GridSearchCV(
        est, model_cfg["param_grid"],
        cv=min(INNER_CV, len(y_train_orig)),
        scoring="r2", n_jobs=-1, verbose=0,
    )
    grid.fit(Xtr, y_train_log)
    preds_log = grid.best_estimator_.predict(Xte)
    preds_orig = np.maximum(np.expm1(preds_log), 0.0)
    return preds_orig, grid.best_params_


# ── Experiments ───────────────────────────────────────────────────────────────

def exp_group_kfold(X, y, groups, cat_cols, num_cols, cat_pipe,
                    model_name, model_cfg, ds_name):
    n_groups = len(np.unique(groups))
    n_splits = min(N_SPLITS_GKF, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        test_mats = sorted(set(groups[te_idx]))
        Xtr, Xte = transform_split(X_tr, X_te, cat_cols, num_cols, cat_pipe)
        preds, params = fit_and_predict(Xtr, y_tr.values, Xte, model_cfg)
        m = compute_metrics(y_te.values, preds)
        rows.append({
            "dataset": ds_name, "model": model_name, "fold": fold,
            "n_train": len(tr_idx), "n_test": len(te_idx),
            "test_materials": "; ".join(test_mats),
            **m, "best_params": str(params),
        })
        print(f"      Fold {fold}: R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
              f"RMSE={m['RMSE']:.2f}  (n_test={len(te_idx)})")
    return rows


def exp_lomo(X, y, groups, cat_cols, num_cols, cat_pipe,
             model_name, model_cfg, ds_name):
    unique, counts = np.unique(groups, return_counts=True)
    eligible = sorted(
        unique[counts >= MIN_SAMPLES_LOMO],
        key=lambda m: -dict(zip(unique, counts))[m],
    )
    rows = []
    for mat in eligible:
        te_idx = np.where(groups == mat)[0]
        tr_idx = np.where(groups != mat)[0]
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        Xtr, Xte = transform_split(X_tr, X_te, cat_cols, num_cols, cat_pipe)
        preds, params = fit_and_predict(Xtr, y_tr.values, Xte, model_cfg)
        m = compute_metrics(y_te.values, preds)
        rows.append({
            "dataset": ds_name, "model": model_name,
            "held_out_material": mat,
            "n_train": len(tr_idx), "n_test": len(te_idx),
            **m, "best_params": str(params),
        })
        print(f"      {mat:>8s} (n={len(te_idx):>3d}): "
              f"R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}")
    return rows


def exp_leave_ag_out(X, y, groups, cat_cols, num_cols, cat_pipe,
                     model_name, model_cfg, ds_name):
    ag_idx = np.where(groups == "Ag")[0]
    non_ag_idx = np.where(groups != "Ag")[0]
    if len(ag_idx) == 0:
        print("      No Ag samples -- skipped.")
        return []
    X_tr, X_te = X.iloc[non_ag_idx], X.iloc[ag_idx]
    y_tr, y_te = y.iloc[non_ag_idx], y.iloc[ag_idx]
    Xtr, Xte = transform_split(X_tr, X_te, cat_cols, num_cols, cat_pipe)
    preds, params = fit_and_predict(Xtr, y_tr.values, Xte, model_cfg)
    m = compute_metrics(y_te.values, preds)
    print(f"      Train(non-Ag)={len(non_ag_idx)}  Test(Ag)={len(ag_idx)}: "
          f"R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}")
    return [{
        "dataset": ds_name, "model": model_name,
        "n_train_nonAg": len(non_ag_idx), "n_test_Ag": len(ag_idx),
        **m, "best_params": str(params),
    }]


def exp_family_group_kfold(X, y, groups, cat_cols, num_cols, cat_pipe,
                           model_name, model_cfg, ds_name):
    families = np.array([formula_to_family(g) for g in groups])
    unique_fam = np.unique(families)
    n_fam = len(unique_fam)
    if n_fam < 2:
        print(f"      Only {n_fam} family -- skipped.")
        return []
    n_splits = min(N_SPLITS_GKF, n_fam)
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, families), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        test_fams = sorted(set(families[te_idx]))
        test_mats = sorted(set(groups[te_idx]))
        Xtr, Xte = transform_split(X_tr, X_te, cat_cols, num_cols, cat_pipe)
        preds, params = fit_and_predict(Xtr, y_tr.values, Xte, model_cfg)
        m = compute_metrics(y_te.values, preds)
        rows.append({
            "dataset": ds_name, "model": model_name, "fold": fold,
            "n_train": len(tr_idx), "n_test": len(te_idx),
            "test_families": "; ".join(test_fams),
            "test_materials": "; ".join(test_mats),
            **m, "best_params": str(params),
        })
        print(f"      Fold {fold}: R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
              f"RMSE={m['RMSE']:.2f}  (n={len(te_idx)}, fam={test_fams})")
    return rows


def exp_downsample_ag(X, y, groups, cat_cols, num_cols, cat_pipe,
                      model_name, model_cfg, ds_name):
    unique, counts = np.unique(groups, return_counts=True)
    count_dict = dict(zip(unique, counts))
    sorted_counts = sorted(counts, reverse=True)
    if "Ag" not in count_dict or len(sorted_counts) < 2:
        print("      Cannot downsample -- skipped.")
        return []
    target_n = int(sorted_counts[1])
    ag_idx = np.where(groups == "Ag")[0]
    non_ag_idx = np.where(groups != "Ag")[0]
    print(f"      Ag: {len(ag_idx)} -> {target_n}  |  "
          f"non-Ag: {len(non_ag_idx)}  |  "
          f"total after downsample: {target_n + len(non_ag_idx)}")
    rows = []
    for seed in DOWNSAMPLE_SEEDS:
        rng = np.random.RandomState(seed)
        sampled_ag = rng.choice(ag_idx, size=target_n, replace=False)
        keep_idx = np.sort(np.concatenate([non_ag_idx, sampled_ag]))
        X_sub = X.iloc[keep_idx].reset_index(drop=True)
        y_sub = y.iloc[keep_idx].reset_index(drop=True)
        try:
            strat_bins = pd.qcut(y_sub, q=5, labels=False, duplicates="drop")
            stratify = strat_bins
        except ValueError:
            stratify = None
        tr_idx, te_idx = train_test_split(
            np.arange(len(y_sub)), test_size=0.2, random_state=seed,
            stratify=stratify,
        )
        X_tr, X_te = X_sub.iloc[tr_idx], X_sub.iloc[te_idx]
        y_tr, y_te = y_sub.iloc[tr_idx], y_sub.iloc[te_idx]
        Xtr, Xte = transform_split(X_tr, X_te, cat_cols, num_cols, cat_pipe)
        preds, params = fit_and_predict(Xtr, y_tr.values, Xte, model_cfg)
        m = compute_metrics(y_te.values, preds)
        rows.append({
            "dataset": ds_name, "model": model_name, "seed": seed,
            "n_total": len(y_sub), "n_train": len(tr_idx),
            "n_test": len(te_idx), "ag_kept": target_n,
            **m, "best_params": str(params),
        })
        print(f"      seed={seed}: R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
              f"RMSE={m['RMSE']:.2f}")
    r2s = [r["R2"] for r in rows if not np.isnan(r["R2"])]
    if r2s:
        print(f"      => Mean R2={np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
    return rows


# ── Runner & Saver ────────────────────────────────────────────────────────────

def run_experiment_suite(X, y, groups, cat_cols, num_cols, cat_pipe,
                         ds_name, models):
    all_gkf, all_lomo, all_lag = [], [], []
    all_fam, all_ds = [], []
    for model_name, model_cfg in models.items():
        print(f"\n  -- {model_name} " + "-" * (55 - len(model_name)))
        print(f"    [Exp 1] GroupKFold (k={N_SPLITS_GKF}, by Formula):")
        rows = exp_group_kfold(X, y, groups, cat_cols, num_cols, cat_pipe,
                               model_name, model_cfg, ds_name)
        all_gkf.extend(rows)
        r2s = [r["R2"] for r in rows if not np.isnan(r["R2"])]
        if r2s:
            print(f"      => Mean R2={np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

        print(f"    [Exp 2] LOMO (n>={MIN_SAMPLES_LOMO}):")
        all_lomo.extend(exp_lomo(X, y, groups, cat_cols, num_cols, cat_pipe,
                                 model_name, model_cfg, ds_name))

        print(f"    [Exp 3] Leave-Ag-Out:")
        all_lag.extend(exp_leave_ag_out(X, y, groups, cat_cols, num_cols, cat_pipe,
                                        model_name, model_cfg, ds_name))

        print(f"    [Exp 4] Family-GroupKFold:")
        rows = exp_family_group_kfold(X, y, groups, cat_cols, num_cols, cat_pipe,
                                      model_name, model_cfg, ds_name)
        all_fam.extend(rows)
        r2s = [r["R2"] for r in rows if not np.isnan(r["R2"])]
        if r2s:
            print(f"      => Mean R2={np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

        print(f"    [Exp 5] Downsample-Ag CV:")
        all_ds.extend(exp_downsample_ag(X, y, groups, cat_cols, num_cols, cat_pipe,
                                        model_name, model_cfg, ds_name))

    return {"gkf": all_gkf, "lomo": all_lomo, "lag": all_lag,
            "fam": all_fam, "ds": all_ds}


def save_and_print_results(results_by_ds, suffix=""):
    all_gkf = sum([r["gkf"] for r in results_by_ds.values()], [])
    all_lomo = sum([r["lomo"] for r in results_by_ds.values()], [])
    all_lag = sum([r["lag"] for r in results_by_ds.values()], [])
    all_fam = sum([r["fam"] for r in results_by_ds.values()], [])
    all_ds = sum([r["ds"] for r in results_by_ds.values()], [])

    gkf_df = pd.DataFrame(all_gkf)
    lomo_df = pd.DataFrame(all_lomo)
    lag_df = pd.DataFrame(all_lag)
    fam_df = pd.DataFrame(all_fam)
    ds_df = pd.DataFrame(all_ds)

    sfx = f"_{suffix}" if suffix else ""
    gkf_df.to_csv(os.path.join(OUTPUT_DIR, f"exp1_group_kfold{sfx}.csv"), index=False)
    lomo_df.to_csv(os.path.join(OUTPUT_DIR, f"exp2_lomo{sfx}.csv"), index=False)
    lag_df.to_csv(os.path.join(OUTPUT_DIR, f"exp3_leave_ag_out{sfx}.csv"), index=False)
    fam_df.to_csv(os.path.join(OUTPUT_DIR, f"exp4_family_group_kfold{sfx}.csv"), index=False)
    ds_df.to_csv(os.path.join(OUTPUT_DIR, f"exp5_downsample_ag{sfx}.csv"), index=False)

    if gkf_df.empty:
        print("  (no results to summarise)")
        return pd.DataFrame()

    summary_rows = []
    for (ds, mdl), grp in gkf_df.groupby(["dataset", "model"]):
        row = {"dataset": ds, "model": mdl}
        row["Exp1_GKF_R2_mean"] = grp["R2"].mean()
        row["Exp1_GKF_R2_std"] = grp["R2"].std()
        lag_row = lag_df[(lag_df["dataset"] == ds) & (lag_df["model"] == mdl)]
        row["Exp3_LeaveAgOut_R2"] = lag_row["R2"].values[0] if len(lag_row) else np.nan
        fam_row = fam_df[(fam_df["dataset"] == ds) & (fam_df["model"] == mdl)]
        if len(fam_row):
            row["Exp4_FamGKF_R2_mean"] = fam_row["R2"].mean()
            row["Exp4_FamGKF_R2_std"] = fam_row["R2"].std()
        else:
            row["Exp4_FamGKF_R2_mean"] = np.nan
            row["Exp4_FamGKF_R2_std"] = np.nan
        ds_row = ds_df[(ds_df["dataset"] == ds) & (ds_df["model"] == mdl)]
        if len(ds_row):
            row["Exp5_Downsample_R2_mean"] = ds_row["R2"].mean()
            row["Exp5_Downsample_R2_std"] = ds_row["R2"].std()
        else:
            row["Exp5_Downsample_R2_mean"] = np.nan
            row["Exp5_Downsample_R2_std"] = np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"summary{sfx}.csv"), index=False)

    label = f" [{suffix.upper()}]" if suffix else ""
    print(f"\n{'=' * 72}")
    print(f"  FINAL SUMMARY{label}  "
          f"(all metrics on original scale, trained with log1p)")
    print(f"{'=' * 72}")

    print(f"\n[Exp 1] GroupKFold by Formula -- Mean R2:")
    print(gkf_df.groupby(["dataset", "model"])["R2"].agg(["mean", "std"]).round(4).to_string())

    print(f"\n[Exp 2] LOMO (n >= {MIN_SAMPLES_LOMO}):")
    if not lomo_df.empty:
        cols2 = ["dataset", "model", "held_out_material", "n_test", "R2", "MAE", "RMSE"]
        print(lomo_df[cols2].round(4).to_string(index=False))

    print("\n[Exp 3] Leave-Ag-Out:")
    if not lag_df.empty:
        print(lag_df.round(4).to_string(index=False))

    print("\n[Exp 4] Family-GroupKFold -- Mean R2:")
    if not fam_df.empty:
        print(fam_df.groupby(["dataset", "model"])["R2"].agg(["mean", "std"]).round(4).to_string())

    print("\n[Exp 5] Downsample-Ag -- Mean R2:")
    if not ds_df.empty:
        print(ds_df.groupby(["dataset", "model"])["R2"].agg(["mean", "std"]).round(4).to_string())

    print(f"\n-- Condensed summary (R2 only){label} --")
    print(summary_df.round(4).to_string(index=False))
    return summary_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # Pass 1: Unfiltered
    print("\n" + "#" * 72)
    print("#  PASS 1 -- FULL DATASET (unfiltered)")
    print("#" * 72)

    results_unfilt = {}
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'=' * 72}")
        print(f"  DATASET: {ds_name}")
        print(f"{'=' * 72}")
        X, y, groups = load_dataset(ds_cfg)
        cat_cols, num_cols, cat_pipe = build_transformers(X)
        ug, uc = np.unique(groups, return_counts=True)
        mat_dist = pd.Series(dict(zip(ug, uc))).sort_values(ascending=False)
        print(f"  Samples={len(y)}  Features={X.shape[1]}  Unique formulas={len(ug)}")
        print(f"  Top materials: {mat_dist.head(10).to_dict()}")
        families = np.array([formula_to_family(g) for g in groups])
        fam_u, fam_c = np.unique(families, return_counts=True)
        print(f"  Material families: {dict(zip(fam_u, fam_c))}\n")
        results_unfilt[ds_name] = run_experiment_suite(
            X, y, groups, cat_cols, num_cols, cat_pipe, ds_name, MODELS)

    save_and_print_results(results_unfilt, suffix="")

    # Pass 2: Filtered
    print("\n\n" + "#" * 72)
    print("#  PASS 2 -- FILTERED (extreme-outlier materials removed)")
    print("#" * 72)

    results_filt = {}
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'=' * 72}")
        print(f"  DATASET: {ds_name} [FILTERED]")
        print(f"{'=' * 72}")
        X, y, groups = load_dataset(ds_cfg)
        excluded = detect_extreme_materials(y.values, groups)
        if excluded:
            n_before = len(y)
            X, y, groups = filter_dataset(X, y, groups, excluded)
            print(f"  ** Removed {n_before - len(y)} samples from "
                  f"extreme-outlier material(s): {excluded}")
        else:
            print(f"  ** No extreme-outlier materials detected.")
        cat_cols, num_cols, cat_pipe = build_transformers(X)
        ug, uc = np.unique(groups, return_counts=True)
        mat_dist = pd.Series(dict(zip(ug, uc))).sort_values(ascending=False)
        print(f"  Samples={len(y)}  Features={X.shape[1]}  Unique formulas={len(ug)}")
        print(f"  Top materials: {mat_dist.head(10).to_dict()}")
        families = np.array([formula_to_family(g) for g in groups])
        fam_u, fam_c = np.unique(families, return_counts=True)
        print(f"  Material families: {dict(zip(fam_u, fam_c))}\n")
        results_filt[ds_name] = run_experiment_suite(
            X, y, groups, cat_cols, num_cols, cat_pipe, ds_name, MODELS)

    save_and_print_results(results_filt, suffix="filtered")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print(f"Results saved to {OUTPUT_DIR}/")
