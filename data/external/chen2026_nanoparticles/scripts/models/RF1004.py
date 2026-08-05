# RF1004.py  (PyTorch MLP with stronger regularization to reduce overfitting)

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

import os
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 80
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# -----------------------------
# Device setup
# -----------------------------
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

# -----------------------------
# Robust CSV loader
# -----------------------------
print("Loading data...")
csv_path = str(TRAINING_CSV)

try:
    df = pd.read_csv(csv_path, encoding='latin1')
except Exception as e:
    print(f"⚠️ Default CSV load failed: {e}\nRetrying with python engine...")
    df = pd.read_csv(csv_path, encoding='latin1', engine='python')

df.columns = df.columns.str.strip()

def soft_numeric_convert(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() > 0:
            return converted
    return series

df = df.apply(soft_numeric_convert, axis=0)

# -----------------------------
# Drop unnecessary columns
# -----------------------------
cols_to_drop = [
    'MagpieData range Number',
    'MagpieData range Electronegativity',
    'MagpieData range MeltingT',
    'MagpieData range AtomicRadius',
    'MagpieData range AtomicVolume',
    'MagpieData range CovalentRadius',
    'MagpieData range ThermalConductivity',
    'MagpieData range Density',
    'MagpieData range FusionEnthalpy',
    'MagpieData range Row',
    'MagpieData range Column',
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped {len(cols_to_drop)} standard deviation features.")
print("Initial data shape:", df.shape)

# -----------------------------
# Prepare X and y (NO outlier removal)
# -----------------------------
X = df.drop(columns=['No.', 'Ref', 'Material 1', 'Formula', 'bacteria',
                     'MIC (µg/mL)', 'MIC_class'], errors='ignore')
y = df['MIC_class']
if y.isnull().any():
    raise ValueError("Target column 'MIC_class' contains NaNs; please clean or impute.")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
print(f"Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns.")

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.22, random_state=SEED, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
)

# -----------------------------
# Rare-category collapsing (reduces OHE noise/overfit)
# Applied based on TRAIN counts, then mapped to val/test
# -----------------------------
def collapse_rare_categories(X_df, fit_counts=None, min_count=5):
    X_df = X_df.copy()
    cat_cols = X_df.select_dtypes(include=['object']).columns
    mapping = {}
    if fit_counts is None:
        for c in cat_cols:
            vc = X_df[c].value_counts(dropna=False)
            keep = set(vc[vc >= min_count].index)
            mapping[c] = keep
        # apply to X_df
        for c in cat_cols:
            keep = mapping[c]
            X_df[c] = X_df[c].where(X_df[c].isin(keep), other="RARE")
        return X_df, mapping
    else:
        # apply using provided mapping
        for c in fit_counts:
            keep = fit_counts[c]
            if c in X_df.columns:
                X_df[c] = X_df[c].where(X_df[c].isin(keep), other="RARE")
        return X_df

X_train_rc, rare_keep = collapse_rare_categories(X_train, fit_counts=None, min_count=5)
X_val_rc   = collapse_rare_categories(X_val, fit_counts=rare_keep)
X_test_rc  = collapse_rare_categories(X_test, fit_counts=rare_keep)

# -----------------------------
# Preprocessing (fit ONLY on train)
# -----------------------------
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numerical_cols),
    ('cat', categorical_pipe, categorical_cols)
])

X_train_pp = preprocessor.fit_transform(X_train_rc)
X_val_pp   = preprocessor.transform(X_val_rc)
X_test_pp  = preprocessor.transform(X_test_rc)

def to_dense(m):
    return m.toarray() if sparse.issparse(m) else m

X_train_pp = to_dense(X_train_pp).astype('float32')
X_val_pp   = to_dense(X_val_pp).astype('float32')
X_test_pp  = to_dense(X_test_pp).astype('float32')

input_dim = X_train_pp.shape[1]
print(f"Final input dim after preprocessing: {input_dim}")

# -----------------------------
# Class weights (use kwargs for new sklearn)
# -----------------------------
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
print("Class weights:", {i: float(w) for i, w in enumerate(class_weights_np)})

# -----------------------------
# Torch Dataset / DataLoader
# -----------------------------
class NumpyDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y.astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

BATCH_SIZE = 128  # slightly larger batch helps regularization
train_loader = DataLoader(NumpyDataset(X_train_pp, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(NumpyDataset(X_val_pp, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(NumpyDataset(X_test_pp, y_test), batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# MLP model (smaller + more dropout + input noise)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=[256, 128], dropout=0.5, input_dropout=0.05):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        layers_ = []
        prev = input_dim
        for h in hidden:
            layers_ += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers_ += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers_)
    def forward(self, x):
        x = self.input_dropout(x)
        return self.net(x)

model = MLP(input_dim, num_classes, hidden=[256, 128], dropout=0.5, input_dropout=0.05).to(device)

# -----------------------------
# Loss / Optimizer / Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=6, verbose=True, min_lr=1e-5
)

# -----------------------------
# Training loop (Earlier stopping)
# -----------------------------
EPOCHS = 200
best_val_loss = float('inf')
patience, patience_counter = 60, 0
best_state = None

def run_epoch(dl, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)
    scheduler.step(val_loss)
    print(f"Epoch {epoch:03d} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss, patience_counter = val_loss, 0
        best_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

if best_state:
    model.load_state_dict(best_state)

# -----------------------------
# Test evaluation
# -----------------------------
model.eval()
preds, targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.numpy())

y_pred = np.concatenate(preds)
y_true = np.concatenate(targets)

acc = accuracy_score(y_true, y_pred)
print(f"\n=== Test Accuracy: {acc:.4f} ===")
print("=== Classification Report (Test) ===")
print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4, zero_division=0))