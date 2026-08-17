"""
06_edge_weight_predictor.py
Full pipeline for predicting edge migrant stock (edge weight) using gradient-boosting.

Produces:
 - data/exports/edge_model_metrics.csv
 - data/exports/edge_feature_importance.png
 - models/edge_weight_predictor.joblib

Usage:
    source venv/bin/activate
    python notebooks/06_edge_weight_predictor.py

Notes:
 - Requires outputs from earlier pipeline: data/exports/network_edges.csv and data/processed/factors_panel.csv
 - If xgboost is not installed, falls back to sklearn's HistGradientBoostingRegressor
"""

import os
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import sys
ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.loader import paths, load_factors, load_edges, load_country_metadata
ROOT = paths["ROOT"]
EXPORTS_DIR = paths["EXPORTS_DIR"]
PROCESSED = paths["PROCESSED_DIR"]
MODELS_DIR = paths["MODELS_DIR"]
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Paths for expected input files (used as a quick existence check)
EDGE_CSV = os.path.join(EXPORTS_DIR, "network_edges.csv")
FACTORS_CSV = os.path.join(PROCESSED, "factors_panel.csv")

if not os.path.exists(EDGE_CSV) or not os.path.exists(FACTORS_CSV):
    raise FileNotFoundError("Required input files missing. Run earlier pipeline steps to produce network_edges.csv and factors_panel.csv")

# Try to use XGBoost if available (often faster / better). Fallback to sklearn's HistGradientBoostingRegressor.
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def load_and_build_features():
    edges = load_edges()
    factors = load_factors()
    country_meta = load_country_metadata()

    # Basic renaming: origin_iso3, dest_iso3, year, weight
    if "origin_iso3" not in edges.columns or "dest_iso3" not in edges.columns:
        raise ValueError("network_edges.csv must contain origin_iso3 and dest_iso3 columns")

    # Merge origin factors
    origin = factors.rename(columns=lambda c: f"origin_{c}" if c != "iso3" else "origin_iso3")
    dest = factors.rename(columns=lambda c: f"dest_{c}" if c != "iso3" else "dest_iso3")

    df = edges.merge(origin, left_on=["origin_iso3", "year"], right_on=["origin_iso3", "origin_year"], how="left")
    # Note: origin_year column produced by renaming is 'origin_year'
    # But factors file has column name 'year' — the rename above produced 'origin_year' so right_on is correct
    df = df.merge(dest, left_on=["dest_iso3", "year"], right_on=["dest_iso3", "dest_year"], how="left")

    # If country metadata exists, use continent/income group to create relational features
    if country_meta is not None:
        df = df.merge(country_meta.rename(columns={"iso3": "origin_iso3", "continent": "origin_continent", "income_group": "origin_income_group"}), on="origin_iso3", how="left")
        df = df.merge(country_meta.rename(columns={"iso3": "dest_iso3", "continent": "dest_continent", "income_group": "dest_income_group"}), on="dest_iso3", how="left")

    # Target
    df["weight"] = df["weight"].astype(float)
    df["log_weight"] = np.log1p(df["weight"])  # target for regression

    # Feature engineering: logs of gdp / population
    for prefix in ["origin_", "dest_"]:
        if f"{prefix}gdp_per_capita" in df.columns:
            df[f"{prefix}log_gdp"] = np.where(df[f"{prefix}gdp_per_capita"] > 0, np.log(df[f"{prefix}gdp_per_capita"]), np.nan)
        else:
            df[f"{prefix}log_gdp"] = np.nan
        if f"{prefix}population" in df.columns:
            df[f"{prefix}log_pop"] = np.where(df[f"{prefix}population"] > 0, np.log(df[f"{prefix}population"]), np.nan)
        else:
            df[f"{prefix}log_pop"] = np.nan

    # Differences / ratios
    df["gdp_log_diff"] = df["origin_log_gdp"] - df["dest_log_gdp"]
    df["pop_log_diff"] = df["origin_log_pop"] - df["dest_log_pop"]

    # Conflict, unemployment etc.
    for col in ["conflict_intensity", "unemployment", "education_index", "visa_openness_index", "climate_vulnerability"]:
        df[f"origin_{col}"] = df.get(f"origin_{col}", np.nan)
        df[f"dest_{col}"] = df.get(f"dest_{col}", np.nan)

    # Categorical relational features
    df["same_continent"] = (df.get("origin_continent") == df.get("dest_continent")).astype(float)
    df["same_income_group"] = (df.get("origin_income_group") == df.get("dest_income_group")).astype(float)

    # Year as numeric
    df["year"] = df["year"].astype(int)

    # Select feature columns
    feature_cols = [
        "origin_log_gdp", "dest_log_gdp", "origin_log_pop", "dest_log_pop",
        "gdp_log_diff", "pop_log_diff",
        "origin_conflict_intensity", "dest_conflict_intensity",
        "origin_unemployment", "dest_unemployment",
        "origin_education_index", "dest_education_index",
        "origin_visa_openness_index", "dest_visa_openness_index",
        "origin_climate_vulnerability", "dest_climate_vulnerability",
        "same_continent", "same_income_group",
        "year",
    ]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_features = df[["origin_iso3", "dest_iso3", "year", "weight", "log_weight"] + feature_cols].copy()

    # Drop rows where target is missing or where both origin/dest logs are missing
    df_features = df_features.dropna(subset=["log_weight"])
    # Require at least some core features non-null
    df_features = df_features.dropna(subset=[c for c in ["origin_log_gdp", "dest_log_gdp", "origin_log_pop", "dest_log_pop"] if c in df_features.columns], how="any")

    print(f"Built feature frame: {df_features.shape[0]:,} rows, {len(feature_cols)} features")

    return df_features, feature_cols


def build_and_train(df, feature_cols):
    X = df[feature_cols]
    y = df["log_weight"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
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
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_iter": [100, 200, 400],
            "estimator__max_leaf_nodes": [15, 31, 63, None],
        }
        estimator_name = "histgb"

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("estimator", estimator)
    ])

    # Randomized search
    n_iter = 20
    search = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=3,
                                scoring="neg_root_mean_squared_error", random_state=42, n_jobs=4, verbose=1)

    print("Starting randomized CV search (this may take a few minutes)...")
    search.fit(X_train, y_train)

    print(f"Best params ({estimator_name}): {search.best_params_}")

    # Evaluate
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    # mean_squared_error in some sklearn versions does not accept `squared` kwarg — take sqrt of MSE for RMSE
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE (log-weight): {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # Feature importance if available
    fi = None
    if XGB_AVAILABLE:
        try:
            fi = best.named_steps["estimator"].feature_importances_
        except Exception:
            fi = None
    else:
        try:
            fi = best.named_steps["estimator"].feature_importances_
        except Exception:
            fi = None

    # Save model
    model_path = os.path.join(MODELS_DIR, "edge_weight_predictor.joblib")
    joblib.dump({"model": best, "features": feature_cols}, model_path)
    print(f"Saved model: {model_path}")

    # Save metrics & feature importance
    metrics = {"rmse_log": float(rmse), "r2": float(r2), "n_test": int(len(y_test))}
    metrics_path = os.path.join(EXPORTS_DIR, "edge_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if fi is not None:
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.25)))
        ax.barh(fi_df["feature"], fi_df["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("feature importance")
        plt.tight_layout()
        out_fig = os.path.join(EXPORTS_DIR, "edge_feature_importance.png")
        fig.savefig(out_fig, dpi=150)
        print(f"Saved feature importance: {out_fig}")

    return metrics, best


def main():
    df, feature_cols = load_and_build_features()
    metrics, model = build_and_train(df, feature_cols)
    print("Done.")


if __name__ == "__main__":
    main()
