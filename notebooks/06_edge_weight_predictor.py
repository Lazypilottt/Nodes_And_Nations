"""
06_edge_weight_predictor.py
Full pipeline for predicting edge migrant stock (edge weight) using gradient-boosting.

Produces:
 - data/exports/edge_model_metrics.json
 - data/exports/edge_model_ablation.csv      (R² as each feature group is added)
 - data/exports/edge_feature_importance.csv / .png
 - models/edge_weight_predictor.joblib

Usage:
    source venv/bin/activate
    python notebooks/06_edge_weight_predictor.py

Notes:
 - Trains on every bilateral corridor in data/processed/migration_long.csv, not
   just the top-N export, so the model sees small corridors as well as large ones.
 - Train/test and CV splits are grouped by corridor (origin, dest). A random row
   split would put the same corridor in both sets for different years and
   overstate accuracy.
 - The served model uses measured data only (see data/features.py): the
   synthetic religion / integration indices are left out. The ablation table
   shows what each feature group adds on the same held-out corridors.
 - If xgboost is not installed, falls back to sklearn's HistGradientBoostingRegressor
"""

import os
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import sys
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.loader import (
    paths, load_factors, load_migration_long, load_country_metadata, load_geography_dyadic,
)
from data.features import (
    FEATURE_COLUMNS, ORIGINAL_FEATURES, GEOGRAPHY_FEATURES, DEVELOPMENT_FEATURES,
    NETWORK_FEATURES, build_dyad_features, network_lags,
)

EXPORTS_DIR = paths["EXPORTS_DIR"]
MODELS_DIR = paths["MODELS_DIR"]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Try to use XGBoost if available (often faster / better). Fallback to sklearn's HistGradientBoostingRegressor.
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def load_and_build_features():
    migration = load_migration_long()
    factors = load_factors()
    country_meta = load_country_metadata()

    edges = (
        migration[migration["origin_iso3"] != migration["dest_iso3"]]
        .groupby(["origin_iso3", "dest_iso3", "year"], as_index=False)["migrant_stock"].sum()
        .rename(columns={"migrant_stock": "weight"})
    )
    edges = edges[edges["weight"] > 0]

    df = build_dyad_features(
        edges, factors, country_meta,
        geography=load_geography_dyadic(), lags=network_lags(migration),
    )
    df["log_weight"] = np.log1p(df["weight"].astype(float))
    df["corridor"] = df["origin_iso3"] + "->" + df["dest_iso3"]

    # Require the core gravity terms; everything else may be missing
    df = df.dropna(subset=["log_weight", "origin_log_gdp", "dest_log_gdp", "origin_log_pop", "dest_log_pop"])

    print(f"Built feature frame: {df.shape[0]:,} rows, {df['corridor'].nunique():,} corridors, "
          f"{len(FEATURE_COLUMNS)} features")
    return df, list(FEATURE_COLUMNS)


GRAVITY_CORE = ["origin_log_pop", "dest_log_pop", "origin_log_gdp", "dest_log_gdp",
                "log_dist", "contig", "comlang_ethno", "colony"]

ABLATION = [
    ("Classic gravity (size, income, distance, ties)", GRAVITY_CORE),
    ("Original 31 features", ORIGINAL_FEATURES),
    ("+ CEPII geography", ORIGINAL_FEATURES + GEOGRAPHY_FEATURES),
    ("+ development, governance, displacement",
     ORIGINAL_FEATURES + GEOGRAPHY_FEATURES + DEVELOPMENT_FEATURES),
    ("+ diaspora network lags",
     ORIGINAL_FEATURES + GEOGRAPHY_FEATURES + DEVELOPMENT_FEATURES + NETWORK_FEATURES),
    ("Served model: measured data only", FEATURE_COLUMNS),
]
ABLATION_PARAMS = dict(learning_rate=0.1, max_iter=400, max_leaf_nodes=63,
                       min_samples_leaf=20, l2_regularization=0.1, random_state=42)


def grouped_split(df):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    return next(splitter.split(df, groups=df["corridor"]))


def run_ablation(df, train_idx, test_idx):
    """Fit one fixed-parameter model per feature set on the same corridor split."""
    rows = []
    y_train, y_test = df["log_weight"].iloc[train_idx], df["log_weight"].iloc[test_idx]
    for label, cols in ABLATION:
        model = HistGradientBoostingRegressor(**ABLATION_PARAMS)
        model.fit(df[cols].iloc[train_idx], y_train)
        pred = model.predict(df[cols].iloc[test_idx])
        rows.append({
            "feature_set": label,
            "n_features": len(cols),
            "r2": r2_score(y_test, pred),
            "rmse_log": mean_squared_error(y_test, pred) ** 0.5,
        })
        print(f"  {label:<52} {len(cols):>3} features  R² = {rows[-1]['r2']:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXPORTS_DIR, "edge_model_ablation.csv"), index=False)
    return out


def temporal_check(df, feature_cols):
    """
    Train on 1990–2015 and score the observed 2020 snapshot: how well the model
    forecasts a future period rather than unseen corridors.
    """
    train = df[df["year"] <= 2015]
    test = df[df["year"] == 2020]
    model = HistGradientBoostingRegressor(**ABLATION_PARAMS).fit(train[feature_cols], train["log_weight"])
    pred = model.predict(test[feature_cols])
    return {"r2": float(r2_score(test["log_weight"], pred)),
            "rmse_log": float(mean_squared_error(test["log_weight"], pred) ** 0.5),
            "train_years": "1990–2015", "test_year": 2020}


def build_and_train(df, feature_cols, train_idx, test_idx):
    X = df[feature_cols]
    y = df["log_weight"]
    groups = df["corridor"]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Tree ensembles are scale-invariant, so no scaler is needed.
    if XGB_AVAILABLE:
        estimator = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0, n_jobs=4, random_state=42)
        param_dist = {
            "estimator__n_estimators": [100, 200, 400],
            "estimator__max_depth": [3, 5, 7, 9],
            "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "estimator__subsample": [0.6, 0.8, 1.0],
            "estimator__colsample_bytree": [0.6, 0.8, 1.0],
        }
        estimator_name = "xgboost"
    else:
        estimator = HistGradientBoostingRegressor(random_state=42)
        param_dist = {
            "estimator__learning_rate": [0.03, 0.05, 0.1],
            "estimator__max_iter": [200, 400, 600],
            "estimator__max_leaf_nodes": [15, 31, 63],
            "estimator__min_samples_leaf": [20, 50, 100],
            "estimator__l2_regularization": [0.0, 0.1, 1.0],
        }
        estimator_name = "histgb"

    pipe = Pipeline([("estimator", estimator)])

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=8, cv=GroupKFold(n_splits=3),
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=4, verbose=1,
    )

    print("Starting randomized CV search (this may take a few minutes)...")
    search.fit(X_train, y_train, groups=groups.iloc[train_idx])

    print(f"Best params ({estimator_name}): {search.best_params_}")

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE (log-weight): {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    model_path = os.path.join(MODELS_DIR, "edge_weight_predictor.joblib")
    joblib.dump({"model": best, "features": feature_cols}, model_path)
    print(f"Saved model: {model_path}")

    metrics = {
        "estimator": estimator_name,
        "rmse_log": float(rmse),
        "r2": float(r2),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(feature_cols),
        "split": "grouped by corridor",
        "temporal": temporal_check(df, feature_cols),
    }
    metrics_path = os.path.join(EXPORTS_DIR, "edge_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # HistGradientBoosting has no built-in importances; permutation importance
    # on a held-out sample works for either estimator.
    sample = X_test.sample(min(5000, len(X_test)), random_state=42)
    pi = permutation_importance(best, sample, y_test.loc[sample.index], n_repeats=3, random_state=42, n_jobs=4)
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": pi.importances_mean})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(EXPORTS_DIR, "edge_feature_importance.csv"), index=False)
    top = fi_df.head(25)
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.28)))
    ax.barh(top["feature"], top["importance"], color="#2563eb")
    ax.invert_yaxis()
    ax.set_xlabel("permutation importance (Δ R²)")
    plt.tight_layout()
    out_fig = os.path.join(EXPORTS_DIR, "edge_feature_importance.png")
    fig.savefig(out_fig, dpi=150)
    print(f"Saved feature importance: {out_fig}")

    return metrics, best


def main():
    df, feature_cols = load_and_build_features()
    train_idx, test_idx = grouped_split(df)
    print("\nAblation (fixed parameters, same held-out corridors):")
    run_ablation(df, train_idx, test_idx)
    print("\nFinal model:")
    build_and_train(df, feature_cols, train_idx, test_idx)
    print("Done.")


if __name__ == "__main__":
    main()
