"""Random Forest regression for MIC prediction (log1p-transformed target)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths import MIC_DATA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
df = pd.read_csv(MIC_DATA, encoding='latin1')
df.columns = df.columns.str.strip()

cols_to_drop = [
    'MagpieData range Number',
    'MagpieData range Row',
    'MagpieData range Column',
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} standard deviation features.")
print("Initial data shape:", df.shape)

X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)'])
y = np.log1p(df['MIC (µg/mL)'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns.")

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

results = []
importances_all = []
seeds = [2025, 3, 4, 42]

print("Splitting data into train and test sets...")

for seed in seeds:
    print(f"\nRunning with random seed {seed}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=seed))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test)

    r2 = r2_score(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)

    results.append({
        'seed': seed, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse,
        'Best Params': grid_search.best_params_
    })

    X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train)
    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
    y_train_pred = np.expm1(best_model.named_steps['regressor'].predict(X_train_transformed))
    y_test_pred = np.expm1(best_model.named_steps['regressor'].predict(X_test_transformed))
    y_train_original = np.expm1(y_train)

    cat_ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)

    importances = best_model.named_steps['regressor'].feature_importances_
    if len(importances) == len(feature_names):
        importances_all.append(importances)

    sns.set(style="whitegrid", context="notebook", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    plt.title("Actual MIC Value vs. Predicted MIC Values")
    plt.scatter(y_test_original, y_test_pred, color='blue', label='Testing Set', alpha=0.7)
    plt.scatter(y_train_original, y_train_pred, color='orange', label='Training Set', alpha=0.5)
    x_vals = np.linspace(0, 120, 100)
    plt.plot(x_vals, x_vals, 'r--', label='Ideal')
    plt.fill_between(x_vals, x_vals * 0.8, x_vals * 1.2, color='orange', alpha=0.1, label='±20% Relative Error')
    plt.fill_between(x_vals, x_vals * 0.5, x_vals * 1.5, color='purple', alpha=0.1, label='±50% Relative Error')
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.xlabel("Actual MIC (µg/mL)")
    plt.ylabel("Predicted MIC (µg/mL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

if importances_all:
    importances_all = np.array(importances_all)
    mean_importance = importances_all.mean(axis=0)
    std_importance = importances_all.std(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Importance': mean_importance,
        'Std Dev': std_importance
    }).sort_values(by='Mean Importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False, float_format="%.6f"))
else:
    print("\nNo valid feature importances collected.")
