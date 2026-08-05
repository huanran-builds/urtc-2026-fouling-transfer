import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier   # <--- CHANGED
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
print("Loading data...")
df = pd.read_csv(str(TRAINING_CSV), encoding='latin1')
df.columns = df.columns.str.strip()
# Prepare X and y

cols_to_drop = [
    'MagpieData range Number',
    #'MagpieData range Electronegativity',
    'MagpieData range MeltingT',
    #'MagpieData range AtomicRadius',
    #'MagpieData range AtomicVolume',
    #'MagpieData mean AtomicVolume',
    #'MagpieData range CovalentRadius',
    #'MagpieData range ThermalConductivity',
    #'MagpieData mean ThermalConductivity',
    #'MagpieData range Density',
    #'MagpieData range FusionEnthalpy',
    #'MagpieData mean FusionEnthalpy',
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

# Prepare X and y
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)','MIC_class'])
y = df['MIC_class']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

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
    'classifier__n_estimators': [200, 300],
    'classifier__max_depth': [None, 10, 15, 20],
    'classifier__min_samples_split': [2, 5, 7],
    'classifier__min_samples_leaf': [1, 2, 3],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__bootstrap': [True, False],
    'classifier__class_weight': ['balanced', None]
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
        ('classifier', ExtraTreesClassifier(random_state=seed))    # <--- CHANGED
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=7, shuffle=True, random_state=seed),
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
    for i, row in importance_df.head(20).iterrows():
        print(f"{row['Feature']:<40} {row['Mean Importance']:.6f} ± {row['Std Dev']:.6f}")