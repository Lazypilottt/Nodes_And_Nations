"""data.features

Dyadic feature engineering shared by model training (notebooks/06) and the
prediction API (api/predict.py). Keeping a single implementation guarantees the
served model sees exactly the features it was trained on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Node-level columns pulled from the factors panel for both endpoints of a dyad.
_NODE_COLS = [
    "gdp_per_capita", "population", "conflict_intensity", "unemployment",
    "education_index", "visa_openness_index", "climate_vulnerability",
    "integration_policy_index", "cultural_openness_index", "labour_market_integration",
    "anti_discrimination_score", "naturalization_access_score",
    "religious_diversity_index", "religiosity_importance_score", "dominant_religion",
]

FEATURE_COLUMNS = [
    "origin_log_gdp", "dest_log_gdp", "origin_log_pop", "dest_log_pop",
    "gdp_log_diff", "pop_log_diff",
    "origin_conflict_intensity", "dest_conflict_intensity",
    "origin_unemployment", "dest_unemployment",
    "origin_education_index", "dest_education_index",
    "origin_visa_openness_index", "dest_visa_openness_index",
    "origin_climate_vulnerability", "dest_climate_vulnerability",
    "origin_cultural_openness_index", "dest_cultural_openness_index",
    "cultural_openness_diff", "dest_integration_policy_index",
    "dest_labour_market_integration", "dest_anti_discrimination_score",
    "dest_naturalization_access_score",
    "same_dominant_religion", "origin_religiosity_importance_score",
    "dest_religiosity_importance_score", "origin_religious_diversity_index",
    "dest_religious_diversity_index",
    "same_continent", "same_income_group",
    "year",
]


def _safe_log(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0))


def build_dyad_features(
    pairs: pd.DataFrame,
    factors: pd.DataFrame,
    country_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach model features to a frame of (origin_iso3, dest_iso3, year) rows.

    Any extra columns on `pairs` (e.g. the training target) are preserved.
    Missing node attributes stay NaN; the gradient-boosting model handles them.
    """
    node_cols = [c for c in _NODE_COLS if c in factors.columns]
    node = factors[["iso3", "year"] + node_cols].drop_duplicates(["iso3", "year"])

    df = pairs.copy()
    df["year"] = df["year"].astype(int)
    for side in ("origin", "dest"):
        renamed = node.rename(columns={c: f"{side}_{c}" for c in node_cols})
        renamed = renamed.rename(columns={"iso3": f"{side}_iso3"})
        df = df.merge(renamed, on=[f"{side}_iso3", "year"], how="left")

    for side in ("origin", "dest"):
        df[f"{side}_log_gdp"] = _safe_log(df.get(f"{side}_gdp_per_capita"))
        df[f"{side}_log_pop"] = _safe_log(df.get(f"{side}_population"))

    df["gdp_log_diff"] = df["origin_log_gdp"] - df["dest_log_gdp"]
    df["pop_log_diff"] = df["origin_log_pop"] - df["dest_log_pop"]
    df["cultural_openness_diff"] = (
        df["origin_cultural_openness_index"] - df["dest_cultural_openness_index"]
    ).abs()

    o_rel, d_rel = df["origin_dominant_religion"], df["dest_dominant_religion"]
    df["same_dominant_religion"] = (o_rel.notna() & (o_rel == d_rel)).astype(float)

    if country_meta is not None and not country_meta.empty:
        meta = country_meta.set_index("iso3")
        for attr, feat in (("continent", "same_continent"), ("income_group", "same_income_group")):
            o = df["origin_iso3"].map(meta[attr])
            d = df["dest_iso3"].map(meta[attr])
            df[feat] = (o.notna() & (o == d)).astype(float)
    else:
        df["same_continent"] = 0.0
        df["same_income_group"] = 0.0

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df
