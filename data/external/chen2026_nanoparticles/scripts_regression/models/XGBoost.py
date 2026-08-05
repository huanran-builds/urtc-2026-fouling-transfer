"""
XGBoost regression for MIC prediction with classification-style evaluation.

Trains an XGBoost regressor on log1p(MIC), then bins predictions into
strong/moderate/weak classes for additional classification metrics.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths import MIC_WITH_CLASS_DATA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from xgboost import XGBRegressor
import warnings


def mic_to_bin_label(v):
    if v <= 10:
        return "strong"
    elif v <= 100:
        return "moderate"
    else:
        return "weak"


def distance_to_interval(x, a, b):
    if np.isinf(b):
        return 0.0 if x >= a else (a - x)
    if a <= x <= b:
        return 0.0
    return a - x if x < a else x - b


def scores_from_pred(preds):
    """
    Convert numeric predictions to 3 scores (one-vs-rest) using
    inverse distance to intervals: [0,10], (10,100], (100, inf).
    """
    eps = 1e-9
    s_strong   = 1.0 / (1.0 + np.array([distance_to_interval(p, 0.0, 10.0)   for p in preds]))
    s_moderate = 1.0 / (1.0 + np.array([distance_to_interval(p, 10.0, 100.0) for p in preds]))
    s_weak     = 1.0 / (1.0 + np.array([distance_to_interval(p, 100.0, np.inf) for p in preds]))
    scores = np.vstack([s_strong, s_moderate, s_weak]).T + eps
    return scores


print("Loading data...")
df = pd.read_csv(MIC_WITH_CLASS_DATA, encoding="latin1")
df.columns = df.columns.str.strip()

cols_to_drop = []
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
print("Initial data shape:", df.shape)

X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)', 'MIC_class'], errors='ignore')
y = np.log1p(pd.to_numeric(df['MIC (µg/mL)'], errors='coerce'))

df = df.loc[y.notna() & df['MIC_class'].notna()].copy()
X = X.loc[df.index]
y = y.loc[df.index]

df['MIC_class'] = df['MIC_class'].astype(str).str.strip().str.lower()

if 'size (nm)' in X.columns:
    X['size (nm)'] = pd.to_numeric(X['size (nm)'], errors='coerce')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features.")

cat_tf = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                   ('onehot',  OneHotEncoder(handle_unknown='ignore'))])
cat_tf.fit(X[categorical_cols])
cat_ohe = cat_tf.named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

seeds = [3, 60, 2020]
results = []

for seed in seeds:
    print(f"\nSeed {seed}")

    X_train, X_test, y_train, y_test, cls_train_true, cls_test_true = train_test_split(
        X, y, df['MIC_class'], test_size=0.20, random_state=seed, stratify=df['MIC_class']
    )

    num_tf = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                       ('scaler',  StandardScaler())])
    num_tf.fit(X_train[numerical_cols])

    X_train_num = num_tf.transform(X_train[numerical_cols])
    X_test_num  = num_tf.transform(X_test[numerical_cols])
    X_train_cat = cat_tf.transform(X_train[categorical_cols])
    X_test_cat  = cat_tf.transform(X_test[categorical_cols])

    Xtr = hstack([X_train_num, X_train_cat])
    Xte = hstack([X_test_num,  X_test_cat])

    model = XGBRegressor(objective='reg:squarederror', random_state=seed, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid.fit(Xtr, y_train)
    best = grid.best_estimator_

    y_pred_test_log = best.predict(Xte)
    y_pred_test = np.expm1(y_pred_test_log)
    y_test_original = np.expm1(y_test)
    r2   = r2_score(y_test_original, y_pred_test)
    mae  = mean_absolute_error(y_test_original, y_pred_test)
    mse  = mean_squared_error(y_test_original, y_pred_test)
    rmse = np.sqrt(mse)

    cls_pred_test = pd.Series(y_pred_test).apply(mic_to_bin_label).values
    acc = accuracy_score(cls_test_true, cls_pred_test)
    pr, rc, f1, _ = precision_recall_fscore_support(
        cls_test_true, cls_pred_test, average='macro', zero_division=0
    )

    print("\nClassification report (test):")
    print(classification_report(cls_test_true, cls_pred_test, digits=4, zero_division=0))

    labels = ["strong", "moderate", "weak"]
    cm = confusion_matrix(cls_test_true, cls_pred_test, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print("Confusion matrix:")
    print(cm_df.to_string())

    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(cls_test_true)
    if y_true_bin.ndim == 1:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

    scores = scores_from_pred(y_pred_test)
    class_order = ["strong", "moderate", "weak"]
    cols = [np.where(lb.classes_ == c)[0][0] for c in class_order]
    y_true_bin_ordered = y_true_bin[:, cols]

    try:
        auc_macro = roc_auc_score(y_true_bin_ordered, scores, average='macro', multi_class='ovr')
    except Exception as e:
        warnings.warn(f"Could not compute ROC-AUC: {e}")
        auc_macro = np.nan

    results.append({
        "seed": seed, "R2": r2, "MAE": mae, "RMSE": rmse,
        "Accuracy": acc, "Precision_macro": pr, "Recall_macro": rc,
        "F1_macro": f1, "ROC_AUC_macro": auc_macro,
        "Best_Params": grid.best_params_
    })

res_df = pd.DataFrame(results)
print("\nResults per seed:")
print(res_df.to_string(index=False))

if len(results) > 1:
    print("\nSummary (mean +/- std):")
    for k in ["R2", "MAE", "RMSE", "Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "ROC_AUC_macro"]:
        print(f"{k}: {res_df[k].mean():.4f} +/- {res_df[k].std():.4f}")
