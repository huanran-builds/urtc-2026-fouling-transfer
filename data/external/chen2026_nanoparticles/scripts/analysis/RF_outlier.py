import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# === Load data ===
print("Loading data...")
df = pd.read_csv(str(LEGACY_FEATURES_CSV), encoding='latin1')
df.columns = df.columns.str.strip()

# Drop unused or problematic columns (customize if needed)
cols_to_drop = [
    #'MagpieData range Number',
    #'MagpieData range Electronegativity',
    #'MagpieData range MeltingT',
    #'MagpieData range AtomicRadius',
    #'MagpieData range AtomicVolume',
    #'MagpieData mean AtomicVolume',
    #'MagpieData range CovalentRadius',
    #'MagpieData range ThermalConductivity',
    #'MagpieData mean ThermalConductivity',
    #'MagpieData range Density',
    #'MagpieData range FusionEnthalpy',
    #'MagpieData mean FusionEnthalpy',
    #'MagpieData range Row',
    #'MagpieData range Column'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} standard deviation features.")
print("Initial data shape:", df.shape)

# Prepare X and y
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)'])
y = df['MIC (µg/mL)']

# === Remove outliers from target using IQR ===
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = (y >= lower_bound) & (y <= upper_bound)
X = X[mask]
y = y[mask]
print(f"Removed {len(y[~mask])} outliers based on IQR.")

# === Optional: Plot target distribution with bounds ===
plt.figure(figsize=(10, 5))
plt.hist(y, bins=50, alpha=0.7)
plt.axvline(lower_bound, color='red', linestyle='dashed', label='Lower Bound')
plt.axvline(upper_bound, color='red', linestyle='dashed', label='Upper Bound')
plt.title('MIC (µg/mL) Distribution with Outlier Bounds')
plt.xlabel('MIC (µg/mL)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

# Preprocessing
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

# Grid search parameters
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

results = []
importances_all = []
seeds = [42, 2025, 3, 4]

# Fit preprocessor separately to get feature names later
preprocessor.fit(X)
cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

print("Splitting data into train and test sets...")

for seed in seeds:
    print(f"\n🚀 Running with random seed {seed}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=seed))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Store feature importance
    importances = best_model.named_steps['regressor'].feature_importances_
    importances_all.append(importances)

    results.append({
        'seed': seed,
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Best Params': grid_search.best_params_
    })

# === Results ===
results_df = pd.DataFrame(results)

print("\n📊 Raw Results:")
print(results_df[['seed', 'R2', 'MAE', 'MSE', 'RMSE']].to_string(index=False))

summary = results_df[['R2', 'MAE', 'MSE', 'RMSE']].describe().loc[['mean', 'std']]
print("\n📈 Summary (mean ± std):")
for col in ['R2', 'MAE', 'MSE', 'RMSE']:
    mean = summary.loc['mean', col]
    std = summary.loc['std', col]
    print(f"{col}: {mean:.4f} ± {std:.4f}")

print("\n📌 Best Parameters from Each Seed:")
for i, row in results_df.iterrows():
    print(f"Seed {row['seed']}: {row['Best Params']}")

# === Feature Importance Analysis ===
importances_all = np.array([imp for imp in importances_all if len(imp) == len(feature_names)])

mean_importance = importances_all.mean(axis=0)
std_importance = importances_all.std(axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean Importance': mean_importance,
    'Std Dev': std_importance
}).sort_values(by='Mean Importance', ascending=False)

print("\n🌟 All Feature Importances (sorted):")
print(importance_df.to_string(index=False, float_format="%.6f"))