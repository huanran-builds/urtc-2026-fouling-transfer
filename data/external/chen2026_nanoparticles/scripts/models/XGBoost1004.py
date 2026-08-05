import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform
import xgboost as xgb

# --- Load data ---
print("Loading data...")
df = pd.read_csv(str(TRAINING_CSV), encoding='latin1')
df.columns = df.columns.str.strip()

cols_to_drop = [
    'MagpieData range Number',
    'MagpieData range Electronegativity',
    'MagpieData range MeltingT',
    #'MagpieData mean MeltingT',
    'MagpieData range AtomicRadius',
    'MagpieData range AtomicVolume',
    #'MagpieData mean AtomicVolume',
    'MagpieData range CovalentRadius',
    'MagpieData range ThermalConductivity',
    #'MagpieData mean ThermalConductivity',
    'MagpieData range Density',
    'MagpieData range FusionEnthalpy',
    #'MagpieData mean FusionEnthalpy',
    'MagpieData range Row',
    'MagpieData range Column',
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} standard deviation features.")
print("Initial data shape:", df.shape)

# --- OUTLIER REMOVAL BLOCK ---
print("Removing extreme (outlier) data points...")
num_cols = df.select_dtypes(include=[np.number]).columns
Q1 = df[num_cols].quantile(0.05)
Q3 = df[num_cols].quantile(0.95)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before_rows = df.shape[0]
mask = ~((df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)).any(axis=1)
df = df[mask]
after_rows = df.shape[0]
print(f"Removed {before_rows - after_rows} outlier rows. Data shape is now: {df.shape}")

# --- Prepare X and y ---
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria', 'MIC (µg/mL)', 'MIC_class'], errors='ignore')
y = df['MIC_class']

# --- Encode MIC_class labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)

# --- Identify feature types ---
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

# --- Preprocessing (NOT pre-fitted; stays inside the pipeline to avoid leakage) ---
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

# --- XGBoost parameter distributions for RandomizedSearchCV ---
param_distributions = {
    'classifier__n_estimators': randint(300, 1200),
    'classifier__max_depth': randint(3, 10),
    'classifier__learning_rate': uniform(0.01, 0.25),
    'classifier__subsample': uniform(0.6, 0.4),          # 0.6–1.0
    'classifier__colsample_bytree': uniform(0.6, 0.4),   # 0.6–1.0
    'classifier__min_child_weight': randint(1, 8),
    'classifier__gamma': uniform(0.0, 5.0),
    'classifier__reg_alpha': uniform(0.0, 1.0),
    'classifier__reg_lambda': uniform(0.5, 2.0)
}

results = []
seeds = [179, 80, 2024, 80, 80, 80, 80, 80, 80, 80, 80]

print("Splitting data into train and test sets per seed and running randomized search...")

for seed in seeds:
    print(f"\n🚀 Running with random seed {seed}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=seed,
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_jobs=-1
        ))
    ])

    X_with_idx = X.copy()
    X_with_idx['original_index'] = X.index

    X_train, X_test, y_train, y_test = train_test_split(
        X_with_idx, y_encoded, test_size=0.22, random_state=seed, stratify=y_encoded
    )
    test_indices = X_test['original_index'].values
    X_train = X_train.drop(columns=['original_index'])
    X_test = X_test.drop(columns=['original_index'])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rand_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=60,                    # increase for a broader search if needed
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=seed,
        refit=True
    )

    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']

    # Per-seed feature importances (optional text output)
    try:
        importances = best_model.named_steps['classifier'].feature_importances_

        # Build feature names from the fitted preprocessor (post-fit to avoid leakage)
        pre = best_model.named_steps['preprocessor']
        cat_ohe = pre.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_ohe.get_feature_names_out(categorical_cols)
        feature_names = np.array(numerical_cols + list(cat_feature_names))

        if importances.shape[0] == len(feature_names):
            top = np.argsort(importances)[::-1][:30]
            print("\n🌟 Top Feature Importances (this seed):")
            for i in top:
                print(f"{feature_names[i]:<40} {importances[i]:.6f}")
        else:
            print(f"\n⚠️ Skipping feature-importance printout (mismatch: {importances.shape[0]} vs {len(feature_names)})")
    except Exception as e:
        print(f"\n⚠️ Could not compute feature importances: {e}")

    results.append({
        'seed': seed,
        'Accuracy': accuracy,
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1_macro': f1_macro,
        'Best Params': rand_search.best_params_
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
for _, row in results_df.iterrows():
    print(f"Seed {row['seed']}: {row['Best Params']}")