import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from collections import Counter

# Load data
print("Loading data...")
df = pd.read_csv(str(TRAINING_CSV), encoding='latin1')
df.columns = df.columns.str.strip()

cols_to_drop = [
    'MagpieData range Number',
    'MagpieData range MeltingT',
    'MagpieData range Row',
    'MagpieData range Column',
]

df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} standard deviation features.")
print("Initial data shape:", df.shape)

# --- OUTLIER REMOVAL BLOCK (add here!) ---
print("Removing extreme (outlier) data points...")
num_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[num_cols].quantile(0.15)
Q3 = df[num_cols].quantile(0.85)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before_rows = df.shape[0]
outlier_counts = ((df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)).sum(axis=1)
mask = outlier_counts < 2  # Removes row only if it is outlier in 2 or more features
df = df[mask]
after_rows = df.shape[0]
print(f"Removed {before_rows - after_rows} outlier rows. Data shape is now: {df.shape}")

# Prepare X and y
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)','MIC_class'])
y = df['MIC_class']

# === Print class counts (string labels) ===
print("\nClass counts (original labels):")
print(y.value_counts())

le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

# === Print class counts (encoded labels) ===
print("\nClass counts (encoded labels):")
unique, counts = np.unique(y_encoded, return_counts=True)
for label, count in zip(unique, counts):
    print(f"{le.inverse_transform([label])[0]} (encoded {label}): {count}")

# === Class weights calculation ===
class_sample_count = Counter(y_encoded)
n_classes = len(class_sample_count)
n_samples = len(y_encoded)
print("\nClass weights (for use as class_weight):")
class_weights = {}
for label, count in class_sample_count.items():
    weight = n_samples / (n_classes * count)
    class_weights[label] = weight
    print(f"{le.inverse_transform([label])[0]} (encoded {label}): {weight:.3f}")

print("\nClass weights dict for scikit-learn:")
print({label: weight for label, weight in class_weights.items()})

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

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [ 15, 20],  # -1 means no limit for LGBM
    'classifier__learning_rate': [ 0.05, 0.1],
    'classifier__num_leaves': [50, 100],    # LGBM-specific
    'classifier__class_weight': [class_weights, 'balanced', None]
}

results = []
importances_all = []
seeds = [2025,3,4,0,20]

preprocessor.fit(X)
cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)

print("Splitting data into train and test sets...")

for seed in seeds:
    print(f"\n🚀 Running with random seed {seed}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=seed, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )

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

    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']

    # Store feature importances
    importances = best_model.named_steps['classifier'].feature_importances_
    importances_all.append(importances)

    # Store run results
    results.append({
        'seed': seed,
        'Accuracy': accuracy,
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1_macro': f1_macro,
        'Best Params': grid_search.best_params_
    })

    # Print classification report
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
    for i, row in importance_df.head(20).iterrows():  # Show top 20
        print(f"{row['Feature']:<40} {row['Mean Importance']:.6f} ± {row['Std Dev']:.6f}")