#!/usr/bin/env python3
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import TRAINING_CSV, ENUMERATE_AG_ECOLI_CSV, LEGACY_MBC_CSV, LEGACY_FEATURES_CSV, OUTPUT_DIR

"""
3D scatter: Shape x Synthesis Method x Size (nm), coloured by MIC_class.

Optional flags
--------------
--training-csv    Path to training CSV (needed for --overlay and --ad)
--overlay         Option A – overlay training samples as hollow markers
--ad              Option C – mark out-of-AD points with triangle markers
--filter-formula  Filter training data to a single Formula (e.g. "Ag")
--ad-features     Comma-separated feature names for AD distance
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_training(path, filter_formula=None):
    """Load training CSV (latin-1), strip column names & string values."""
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    if filter_formula:
        df = df[df["Formula"] == filter_formula].copy()
        print(f"Filtered training data to Formula='{filter_formula}': "
              f"{len(df)} samples")
    return df


def _compute_ad(df_train, df_enum, k=5, percentile=95, ad_features=None):
    """Return (enum_distances, threshold) using k-NN in shared feature space."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if ad_features:
        feat_cols = [c for c in ad_features
                     if c in df_train.columns and c in df_enum.columns]
    else:
        id_cols = {"No.", "Ref", "Material 1", "Formula", "bacteria"}
        shared = [c for c in df_train.columns if c in df_enum.columns]
        skip = id_cols | {
            c for c in shared
            if any(t in c for t in ("MIC", "MBC", "pred", "class"))}
        feat_cols = [c for c in shared if c not in skip]

    cat_cols = df_train[feat_cols].select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feat_cols if c not in cat_cols]

    ct = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False)),
        ]), cat_cols),
    ])

    X_tr = ct.fit_transform(df_train[feat_cols])
    X_en = ct.transform(df_enum[feat_cols])

    nn = NearestNeighbors(n_neighbors=min(k, len(X_tr)))
    nn.fit(X_tr)

    d_en, _ = nn.kneighbors(X_en)
    d_tr, _ = nn.kneighbors(X_tr)

    mean_en = d_en.mean(axis=1)
    mean_tr = d_tr.mean(axis=1)
    thr = np.percentile(mean_tr, percentile)

    in_ad = mean_en <= thr
    print(f"AD features: {feat_cols}")
    print(f"AD threshold (p{percentile} of training k-NN dist): {thr:.3f}")
    print(f"Enumerated points inside AD: {in_ad.sum()}/{len(in_ad)} "
          f"({in_ad.mean()*100:.1f}%)")
    return mean_en, thr


# ── main ─────────────────────────────────────────────────────────────────────

def main(input_csv, output_path=None, point_size=10, jitter=0.08,
         training_csv=None, overlay=False, ad=False,
         ad_k=5, ad_percentile=95, filter_formula=None,
         ad_features=None):

    # --- Font: Times New Roman ---
    font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    tnr_font = fm.FontProperties(fname=font_path)
    tnr_name = tnr_font.get_name()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  [tnr_name],
        "axes.titlesize":  20,
        "axes.labelsize":  18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.unicode_minus": False,
    })
    sns.set_theme(style="whitegrid", context="notebook",
                  rc={"font.family": "serif", "font.serif": [tnr_name]})

    # --- Load enumeration data ---
    df = pd.read_csv(input_csv)
    req_cols = ["size (nm)", "Shape", "Synthesis Method", "MIC_class"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Categorical axes (X=Shape, Y=Synthesis Method, Z=Size) ---
    shapes  = pd.Categorical(df["Shape"]).categories.tolist()
    methods = pd.Categorical(df["Synthesis Method"]).categories.tolist()
    shape_to_pos  = {s: i for i, s in enumerate(shapes)}
    method_to_pos = {m: i for i, m in enumerate(methods)}

    x = df["Shape"].map(shape_to_pos).astype(float).values
    y = df["Synthesis Method"].map(method_to_pos).astype(float).values
    z = df["size (nm)"].astype(float).values

    if jitter and jitter > 0:
        rng = np.random.default_rng(42)
        x = x + rng.uniform(-jitter, jitter, size=x.size)
        y = y + rng.uniform(-jitter, jitter, size=y.size)

    # --- MIC_class colours ---
    color_map = {"strong": "red", "moderate": "lightblue", "weak": "gray"}
    colors = df["MIC_class"].astype(str).str.lower().map(color_map).fillna("black").values

    # --- Load training data (if requested) ---
    df_train = None
    if training_csv and (overlay or ad):
        df_train = _load_training(training_csv, filter_formula)

    # --- AD computation ---
    ad_distances = ad_threshold = None
    if ad and df_train is not None:
        ad_feat_list = None
        if ad_features:
            ad_feat_list = [f.strip() for f in ad_features.split(",")]
        ad_distances, ad_threshold = _compute_ad(
            df_train, df, k=ad_k, percentile=ad_percentile,
            ad_features=ad_feat_list)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    if ad and ad_distances is not None:
        in_mask  = ad_distances <= ad_threshold
        out_mask = ~in_mask
        if in_mask.any():
            ax.scatter(x[in_mask], y[in_mask], z[in_mask],
                       s=point_size, c=colors[in_mask],
                       marker="o", depthshade=True, alpha=0.9)
        if out_mask.any():
            ax.scatter(x[out_mask], y[out_mask], z[out_mask],
                       s=point_size * 2, c=colors[out_mask],
                       marker="^", depthshade=True, alpha=0.7)
    else:
        ax.scatter(x, y, z, s=point_size, c=colors,
                   depthshade=True, alpha=0.9)

    # --- Training-data overlay (Option A) ---
    if overlay and df_train is not None:
        df_train["size (nm)"] = pd.to_numeric(
            df_train["size (nm)"], errors="coerce")
        tx = df_train["Shape"].map(shape_to_pos)
        ty = df_train["Synthesis Method"].map(method_to_pos)
        tz = df_train["size (nm)"]
        valid = tx.notna() & ty.notna() & tz.notna()
        if valid.any():
            ax.scatter(
                tx[valid].astype(float).values,
                ty[valid].astype(float).values,
                tz[valid].astype(float).values,
                s=point_size * 5, facecolors="none", edgecolors="black",
                linewidths=1.5, depthshade=False)
            print(f"Overlaid {valid.sum()} training points "
                  f"({(~valid).sum()} skipped – unmapped categories)")

    # --- Axes ---
    ax.set_xlabel("Shape",            labelpad=12, fontproperties=tnr_font)
    ax.set_ylabel("Synthesis Method", labelpad=12, fontproperties=tnr_font)
    ax.set_zlabel("Size (nm)",        labelpad=12, fontproperties=tnr_font)
    ax.set_zlim(0, 120)

    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, rotation=15, ha="right", fontproperties=tnr_font)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontproperties=tnr_font)
    for tick in ax.get_zticklabels():
        tick.set_fontproperties(tnr_font)

    ax.grid(True, which="both", alpha=0.35)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    ax.view_init(elev=22, azim=35)

    # --- Legend ---
    legend_elems = []
    for cls in ["strong", "moderate", "weak"]:
        if (df["MIC_class"].astype(str).str.lower() == cls).any():
            legend_elems.append(
                Line2D([0], [0], marker="o", linestyle="", markersize=9,
                       markerfacecolor=color_map[cls],
                       markeredgecolor="none", label=cls.capitalize()))

    if ad and ad_distances is not None:
        legend_elems.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=7,
                   markerfacecolor="gray", markeredgecolor="none",
                   label="Inside AD"))
        legend_elems.append(
            Line2D([0], [0], marker="^", linestyle="", markersize=9,
                   markerfacecolor="gray", markeredgecolor="none",
                   label="Outside AD"))

    if overlay and df_train is not None:
        legend_elems.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor="none", markeredgecolor="black",
                   markeredgewidth=1.5, label="Training data"))

    leg = fig.legend(handles=legend_elems, title="MIC_class",
                     loc="upper right", bbox_to_anchor=(1.15, 0.85),
                     frameon=True)
    for text in leg.get_texts():
        text.set_fontproperties(tnr_font)
    if leg.get_title() is not None:
        leg.get_title().set_fontproperties(tnr_font)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="3D scatter of Shape × Synthesis Method × Size (nm) "
                    "colored by MIC_class. Supports training-data overlay "
                    "(Option A) and k-NN AD analysis (Option C).")

    p.add_argument("input_csv",
                   help="CSV with columns: size (nm), Shape, "
                        "Synthesis Method, MIC_class")
    p.add_argument("--out",
                   help="Output image path (omit to open window)")
    p.add_argument("--point-size", type=float, default=10)
    p.add_argument("--jitter",     type=float, default=0.08)

    g = p.add_argument_group("Option A – training overlay")
    g.add_argument("--training-csv",
                   help="Path to training CSV (latin-1 encoded)")
    g.add_argument("--overlay", action="store_true",
                   help="Overlay training samples as hollow markers")
    g.add_argument("--filter-formula",
                   help="Show only this Formula in training data (e.g. Ag)")

    g2 = p.add_argument_group("Option C – k-NN applicability domain")
    g2.add_argument("--ad", action="store_true",
                    help="Mark out-of-AD points with triangle markers")
    g2.add_argument("--ad-k", type=int, default=5,
                    help="Number of neighbours (default 5)")
    g2.add_argument("--ad-percentile", type=float, default=95,
                    help="Training distance percentile for AD threshold "
                         "(default 95)")
    g2.add_argument("--ad-features", type=str, default=None,
                    help="Comma-separated feature names for AD distance. "
                         "E.g. 'size (nm),Shape,Synthesis Method'")

    args = p.parse_args()

    main(args.input_csv,
         output_path=args.out,
         point_size=args.point_size,
         jitter=args.jitter,
         training_csv=args.training_csv,
         overlay=args.overlay,
         ad=args.ad,
         ad_k=args.ad_k,
         ad_percentile=args.ad_percentile,
         filter_formula=args.filter_formula,
         ad_features=args.ad_features)
