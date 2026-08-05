"""SVM regression for MBC prediction (log1p-transformed target)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths import MBC_DATA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from scipy.sparse import hstack

print("Loading data...")
df = pd.read_csv(MBC_DATA, encoding='latin1')
df.columns = df.columns.str.strip()

cols_to_drop = [
    'MagpieData range Number',
    'MagpieData range Row',
    'MagpieData range Column',
    'duration'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} features.")
print("Initial data shape:", df.shape)

X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MBC (µg/mL)'])
y = np.log1p(df['MBC (µg/mL)'])

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
seeds = [42, 2, 20, 4]

print(f"\nTotal features after transformation: {len(feature_names)}")

for seed in seeds:
    print(f"\nRunning with random seed {seed}...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

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

    model = SVR()
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],
        'epsilon': [0.001, 0.01, 0.1, 1, 10],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_transformed, y_train)

    best_model = grid_search.best_estimator_
    y_pred_log = best_model.predict(X_test_transformed)
    y_pred = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test)

    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)

    result = permutation_importance(
        best_model, X_test_transformed.toarray(), y_test,
        n_repeats=10, random_state=seed, scoring='r2', n_jobs=-1
    )
    importances = result.importances_mean
    normalized_importance = importances / np.sum(np.abs(importances))
    importances_all.append(normalized_importance)

    results.append({
        'seed': seed, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse,
        'Best Params': grid_search.best_params_
    })

results_df = pd.DataFrame(results)
print("\nRaw Results:")
print(results_df[['seed', 'R2', 'MAE', 'MSE', 'RMSE']].to_string(index=False))

summary = results_df[['R2', 'MAE', 'MSE', 'RMSE']].describe().loc[['mean', 'std']]
print("\nSummary (mean +/- std):")
for col in ['R2', 'MAE', 'MSE', 'RMSE']:
    print(f"{col}: {summary.loc['mean', col]:.4f} +/- {summary.loc['std', col]:.4f}")

print("\nBest Parameters from Each Seed:")
for i, row in results_df.iterrows():
    print(f"Seed {row['seed']}: {row['Best Params']}")

importances_all = np.array(importances_all)
mean_importance = importances_all.mean(axis=0)
std_importance = importances_all.std(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean Importance': mean_importance,
    'Std Dev': std_importance
}).sort_values(by='Mean Importance', ascending=False)
print("\nFeature Importances from Permutation Importance (sorted):")
print(importance_df.to_string(index=False, float_format="%.6f"))
