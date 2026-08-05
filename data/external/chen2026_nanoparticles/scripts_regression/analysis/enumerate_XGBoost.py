"""
XGBoost regression for MBC + prediction on enumeration dataset.

Trains on MBC data, then predicts on an enumeration CSV (enumerate_ag.csv)
and appends a 'MBC_pred' column.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths import MBC_DATA, ENUMERATE_AG, OUTPUT_DIR

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
df = pd.read_csv(MBC_DATA, encoding='latin1')
df.columns = df.columns.str.strip()

cols_to_drop = []
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print("Initial data shape:", df.shape)

# ── Outlier removal ──────────────────────────────────────────────────────────
print("Removing extreme (outlier) data points...")
num_cols_all = df.select_dtypes(include=[np.number]).columns
Q1 = df[num_cols_all].quantile(0.05)
Q3 = df[num_cols_all].quantile(0.95)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before_rows = df.shape[0]
mask = ~((df[num_cols_all] < lower_bound) | (df[num_cols_all] > upper_bound)).any(axis=1)
df = df[mask]
after_rows = df.shape[0]
print(f"Removed {before_rows - after_rows} outlier rows. Data shape is now: {df.shape}")

X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MBC (µg/mL)'], errors='ignore')
y = df['MBC (µg/mL)']

if 'size (nm)' in X.columns:
    X['size (nm)'] = pd.to_numeric(X['size (nm)'], errors='coerce')

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
categorical_transformer.fit(X[categorical_cols])

cat_ohe = categorical_transformer.named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

results = []
importances_all = []
seeds = [42]

print(f"\nTotal features after transformation: {len(feature_names)}")

for seed in seeds:
    print(f"\nRunning with random seed {seed}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    numerical_transformer.fit(X_train[numerical_cols])

    X_train_num = numerical_transformer.transform(X_train[numerical_cols])
    X_test_num = numerical_transformer.transform(X_test[numerical_cols])
    X_train_cat = categorical_transformer.transform(X_train[categorical_cols])
    X_test_cat = categorical_transformer.transform(X_test[categorical_cols])

    X_train_transformed = hstack([X_train_num, X_train_cat])
    X_test_transformed = hstack([X_test_num, X_test_cat])

    model = XGBRegressor(objective='reg:squarederror', random_state=seed)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_transformed, y_train)

    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test_transformed)
    y_train_pred = best_model.predict(X_train_transformed)

    r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    importance_dict = best_model.get_booster().get_score(importance_type='weight')
    importance_vector = np.zeros(len(feature_names))
    for i, fname in enumerate(feature_names):
        importance_vector[i] = importance_dict.get(f"f{i}", 0)
    total = np.sum(importance_vector)
    normalized_importance = importance_vector / total if total != 0 else importance_vector
    importances_all.append(normalized_importance)

    results.append({
        'seed': seed, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse,
        'Best Params': grid_search.best_params_
    })

    sns.set(style="whitegrid", context="notebook", font_scale=1.0)
    plt.figure(figsize=(8, 6))
    plt.title("Actual MBC Value vs. Predicted MBC Values")
    plt.scatter(y_test, y_test_pred, label='Testing Set', alpha=0.7)
    plt.scatter(y_train, y_train_pred, label='Training Set', alpha=0.5)

    x_max = max(float(np.nanmax(y)), float(np.nanmax(y_test_pred))) * 1.05 if len(y_test_pred) else 120
    x_vals = np.linspace(0, max(120, x_max), 200)
    plt.plot(x_vals, x_vals, linestyle='--', label='Ideal')
    plt.fill_between(x_vals, x_vals * 0.8, x_vals * 1.2, alpha=0.1, label='+/-20% Relative Error')
    plt.fill_between(x_vals, x_vals * 0.5, x_vals * 1.5, alpha=0.1, label='+/-50% Relative Error')
    plt.xlim(0, max(120, x_max))
    plt.ylim(0, max(120, x_max))
    plt.xlabel("Actual MBC (ug/mL)")
    plt.ylabel("Predicted MBC (ug/mL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results)
print("\nRaw Results:")
print(results_df[['seed', 'R2', 'MAE', 'MSE', 'RMSE']].to_string(index=False))

importances_all = np.array(importances_all)
mean_importance = importances_all.mean(axis=0)
std_importance = importances_all.std(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean Importance': mean_importance,
    'Std Dev': std_importance
}).sort_values(by='Mean Importance', ascending=False)
print("\nFeature Importances from XGBoost (sorted):")
print(importance_df.to_string(index=False, float_format="%.6f"))

# ── Predict on enumeration dataset ───────────────────────────────────────────
if os.path.exists(ENUMERATE_AG):
    print("\nLoading enumeration dataset for prediction...")
    new_df = pd.read_csv(ENUMERATE_AG, encoding='latin1')
    new_df.columns = new_df.columns.str.strip()
    new_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    X_new = new_df.drop(
        columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)', 'MIC_class'],
        errors='ignore'
    )
    if 'size (nm)' in X_new.columns:
        X_new['size (nm)'] = pd.to_numeric(X_new['size (nm)'], errors='coerce')

    X_new_num = X_new.reindex(columns=numerical_cols)
    X_new_cat = X_new.reindex(columns=categorical_cols)

    X_new_num_t = numerical_transformer.transform(X_new_num)
    X_new_cat_t = categorical_transformer.transform(X_new_cat)
    X_new_transformed = hstack([X_new_num_t, X_new_cat_t])

    y_new_pred = best_model.predict(X_new_transformed)
    new_df['MBC_pred (µg/mL)'] = y_new_pred

    out_path = os.path.join(OUTPUT_DIR, "enumerate_ag_with_MBC_pred.csv")
    new_df.to_csv(out_path, index=False)
    print(f"Predictions saved to: {out_path}")
else:
    print(f"\nEnumeration dataset not found at: {ENUMERATE_AG}")
