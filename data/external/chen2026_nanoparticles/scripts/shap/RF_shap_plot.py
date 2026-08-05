import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier   # <<< changed here
import shap
import matplotlib.pyplot as plt
import os
from shap import Explanation

# --- Load data ---
print("Loading data...")
df = pd.read_csv(str(TRAINING_CSV), encoding='latin1')
df.columns = df.columns.str.strip()

cols_to_drop = [
    'MagpieData range Number',
    'MagpieData range Electronegativity',
    'MagpieData range MeltingT',
    'MagpieData mean MeltingT',
    'MagpieData range AtomicRadius',
    'MagpieData range AtomicVolume',
    'MagpieData mean AtomicVolume',
    'MagpieData range CovalentRadius',
    'MagpieData range ThermalConductivity',
    'MagpieData mean ThermalConductivity',
    'MagpieData range Density',
    'MagpieData range FusionEnthalpy',
    'MagpieData mean FusionEnthalpy',
    'MagpieData range Row',
    'MagpieData range Column',
    #'duration',
    #'Shape_Polyhedral',
    #'Shape_nanosheets',
    #'Shape_quantum dots',
    #'Shape.1_Oval'
]

df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} standard deviation features.")
print("Initial data shape:", df.shape)

# --- OUTLIER REMOVAL BLOCK ---
print("Removing extreme (outlier) data points...")
num_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before_rows = df.shape[0]
mask = ~((df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)).any(axis=1)
df = df[mask]
after_rows = df.shape[0]
print(f"Removed {before_rows - after_rows} outlier rows. Data shape is now: {df.shape}")

# --- Prepare X and y ---
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)','MIC_class'])
y = df['MIC_class']

# --- Optional: Encode MIC_class labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

# --- Identify feature types ---
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

# --- Preprocessing ---
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

# --- Random Forest parameter grid ---
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

results = []
importances_all = []
seeds = [0]

# Fit preprocessor to get feature names
preprocessor.fit(X)
cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

print("Splitting data into train and test sets...")

for seed in seeds:
    print(f"\n🚀 Running with random seed {seed}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(    # <<< changed here
            random_state=seed,
            n_jobs=-1
        ))
    ])
    X['original_index'] = X.index

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.22, random_state=seed, stratify=y_encoded
    )
    test_indices = X_test['original_index'].values
    X_test = X_test.drop(columns=['original_index'])

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']

    importances = best_model.named_steps['classifier'].feature_importances_
    importances_all.append(importances)

    results.append({
        'seed': seed,
        'Accuracy': accuracy,
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1_macro': f1_macro,
        'Best Params': grid_search.best_params_
    })

    print(f"\n📋 Classification Report for Seed {seed}:\n")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# === Results ===
results_df = pd.DataFrame(results)
print("\n📊 Raw Results per Seed:")
print(results_df[['seed', 'Accuracy', 'Precision', 'Recall', 'F1_macro']].to_string(index=False))

summary = results_df[['Accuracy', 'Precision', 'Recall', 'F1_macro']].describe().loc[['mean', 'std']]
print("\n📈 Summary (mean ± std):")
for col in ['Accuracy', 'Precision', 'Recall', 'F1_macro']:
    mean = summary.loc['mean', col]
    std = summary.loc['std', col]
    print(f"{col}: {mean:.4f} ± {std:.4f}")

print("\n📌 Best Parameters from Each Seed:")
for i, row in results_df.iterrows():
    print(f"Seed {row['seed']}: {row['Best Params']}")

# === Feature Importance Analysis ===
importances_all = np.array([imp for imp in importances_all if imp.shape[0] == len(feature_names)])
if len(importances_all) == 0:
    print("\n⚠️ No valid feature importances to average. Possibly due to inconsistent preprocessing.")
else:
    mean_importance = importances_all.mean(axis=0)
    std_importance = importances_all.std(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Importance': mean_importance,
        'Std Dev': std_importance
    }).sort_values(by='Mean Importance', ascending=False)

    print("\n🌟 Top Feature Importances (Mean ± Std over all seeds):")
    for i, row in importance_df.head(30).iterrows():
        print(f"{row['Feature']:<40} {row['Mean Importance']:.6f} ± {row['Std Dev']:.6f}")


print("Label mapping:", label_mapping)

# === SHAP Analysis ===
rf_model = best_model.named_steps['classifier']
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)   # <<< RF uses TreeExplainer
shap_values = explainer(X_test_transformed)

print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, features=X_test_transformed, feature_names=feature_names)

print("📊 Generating SHAP beeswarm plot...")
shap.summary_plot(shap_values, features=X_test_transformed, feature_names=feature_names)

# === SHAP Waterfall Plots per Sample and Class + CSV dump ===
output_folder = str(OUTPUT_DIR / "shap_waterfall_plots_all_classes_RF")
os.makedirs(output_folder, exist_ok=True)

print("🧬 Saving SHAP waterfall plots + feature tables for each test instance and each class...")

pre = best_model.named_steps['preprocessor']
num_features = numerical_cols
cat_features = pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
feature_names = list(num_features) + list(cat_features)

num_classes = shap_values.values.shape[2]

for i in range(len(X_test)):
    original_idx = test_indices[i]
    x_row_t = np.array(X_test_transformed[i]).ravel()

    for class_idx in range(num_classes):
        shap_row = shap_values.values[i, :, class_idx]

        if not (len(feature_names) == len(x_row_t) == len(shap_row)):
            print(f"⚠️ Length mismatch at sample {i}, class {class_idx}: "
                  f"{len(feature_names)} names vs {len(x_row_t)} values vs {len(shap_row)} shap")
            continue

        df_row = pd.DataFrame({
            "feature": feature_names,
            "value_transformed": x_row_t,
            "shap_value": shap_row,
            "abs_shap": np.abs(shap_row),
        }).sort_values("abs_shap", ascending=False)

        csv_path = f"{output_folder}/features_index_{original_idx}_class_{class_idx}.csv"
        df_row.drop(columns=["abs_shap"]).to_csv(csv_path, index=False)

        print(f"\n🔎 Feature values for original index {original_idx}, class {class_idx}")
        print(df_row.drop(columns=["abs_shap"]).head(15).to_string(index=False))

        explanation = Explanation(
            values=shap_row,
            base_values=shap_values.base_values[i, class_idx],
            data=x_row_t,
            feature_names=feature_names
        )

        plt.figure(figsize=(16, 7))
        shap.plots.waterfall(explanation, show=False)

        filename = f"{output_folder}/shap_waterfall_index_{original_idx}_class_{class_idx}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"✅ Saved plot: {filename}")
        print(f"✅ Saved features CSV: {csv_path}")