"""
3D scatter: Shape x Synthesis Method x Size (nm), coloured by prediction value
or k-NN applicability-domain distance.

Optional flags
--------------
--training-csv  Path to training CSV (needed for --overlay and --ad)
--overlay       Overlay training samples as hollow markers
--ad            Colour by k-NN distance; mark out-of-AD points
--filter-formula  Filter training data to a single Formula (e.g. "Ag")
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _resolve_column(df, name):
    if name in df.columns:
        return name
    prefix = name.split("(")[0].strip()
    matches = [c for c in df.columns if c.startswith(prefix)]
    if matches:
        print(f"Matched column: {repr(matches[0])}")
        return matches[0]
    raise ValueError(
        f"Column '{name}' not found and no prefix match. "
        f"Available: {list(df.columns)}")


def _load_training(path, filter_formula=None):
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
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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


def main(input_csv, output_path=None, point_size=10, jitter=0.08,
         value_col="MIC_pred (ug/mL)", cmap_name="plasma_r",
         vmin=0, vmax=200,
         training_csv=None, overlay=False, ad=False,
         ad_k=5, ad_percentile=95, filter_formula=None,
         ad_features=None):

    sns.set_theme(style="whitegrid", context="notebook")

    df = pd.read_csv(input_csv)
    display_label = value_col
    value_col = _resolve_column(df, value_col)

    req = ["size (nm)", "Shape", "Synthesis Method", value_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["size (nm)"] = pd.to_numeric(df["size (nm)"], errors="coerce")
    df[value_col]   = pd.to_numeric(df[value_col],   errors="coerce")
    df = df.dropna(subset=req).copy()
    if df.empty:
        raise ValueError("No rows left after dropping NaNs in required columns.")

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

    df_train = None
    if training_csv and (overlay or ad):
        df_train = _load_training(training_csv, filter_formula)

    ad_distances = ad_threshold = None
    if ad and df_train is not None:
        ad_feat_list = None
        if ad_features:
            ad_feat_list = [f.strip() for f in ad_features.split(",")]
        ad_distances, ad_threshold = _compute_ad(
            df_train, df, k=ad_k, percentile=ad_percentile,
            ad_features=ad_feat_list)

    vals = df[value_col].astype(float).values
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb_label = display_label

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    if ad and ad_distances is not None:
        in_mask  = ad_distances <= ad_threshold
        out_mask = ~in_mask
        if in_mask.any():
            ax.scatter(x[in_mask], y[in_mask], z[in_mask],
                       s=point_size, c=vals[in_mask], cmap=cmap, norm=norm,
                       marker="o", depthshade=True, alpha=0.9,
                       label="Inside AD")
        if out_mask.any():
            ax.scatter(x[out_mask], y[out_mask], z[out_mask],
                       s=point_size * 2, c=vals[out_mask], cmap=cmap, norm=norm,
                       marker="^", depthshade=True, alpha=0.7,
                       label="Outside AD")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.9, aspect=25)
    else:
        sc = ax.scatter(x, y, z, s=point_size, c=vals, cmap=cmap, norm=norm,
                        depthshade=True, alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.9, aspect=25)

    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}" for t in ticks])
    cbar.set_label(cb_label)

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
                s=point_size * 5, facecolors="none", edgecolors="red",
                linewidths=1.5, depthshade=False, label="Training data")
            print(f"Overlaid {valid.sum()} training points "
                  f"({(~valid).sum()} skipped)")

    ax.set_xlabel("Shape",            labelpad=12)
    ax.set_ylabel("Synthesis Method", labelpad=12)
    ax.set_zlabel("Size (nm)",        labelpad=12)
    ax.set_zlim(0, 120)

    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, rotation=15, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    ax.grid(True, which="both", alpha=0.35)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    ax.view_init(elev=22, azim=35)

    if overlay or ad:
        ax.legend(loc="upper left")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="3D scatter of Shape x Synthesis Method x Size (nm).")

    p.add_argument("input_csv",
                   help="Enumeration CSV with prediction columns")
    p.add_argument("--out", help="Output image path (omit to open window)")
    p.add_argument("--point-size", type=float, default=10)
    p.add_argument("--jitter",     type=float, default=0.08)
    p.add_argument("--value-col",  default="MIC_pred (ug/mL)",
                   help="Column to colour by")
    p.add_argument("--cmap",       default="plasma_r")
    p.add_argument("--vmin", type=float, default=0)
    p.add_argument("--vmax", type=float, default=200)

    g = p.add_argument_group("Training overlay")
    g.add_argument("--training-csv", help="Path to training CSV")
    g.add_argument("--overlay", action="store_true")
    g.add_argument("--filter-formula")

    g2 = p.add_argument_group("k-NN applicability domain")
    g2.add_argument("--ad", action="store_true")
    g2.add_argument("--ad-k", type=int, default=5)
    g2.add_argument("--ad-percentile", type=float, default=95)
    g2.add_argument("--ad-features", type=str, default=None)

    args = p.parse_args()
    main(args.input_csv,
         output_path=args.out, point_size=args.point_size,
         jitter=args.jitter, value_col=args.value_col,
         cmap_name=args.cmap, vmin=args.vmin, vmax=args.vmax,
         training_csv=args.training_csv, overlay=args.overlay,
         ad=args.ad, ad_k=args.ad_k, ad_percentile=args.ad_percentile,
         filter_formula=args.filter_formula, ad_features=args.ad_features)
