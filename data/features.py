"""data.features

Dyadic feature engineering shared by model training (notebooks/06) and the
prediction API (api/predict.py). Keeping a single implementation guarantees the
served model sees exactly the features it was trained on.

Features are organised in groups so notebooks/06 can measure what each group
adds (the ablation table in data/exports/edge_model_ablation.csv):

    original      the 31 features of the first model version
    geography     CEPII distance, shared border, language and colonial ties
    development   World Bank, WGI and UNHCR indicators for both endpoints
    network       last snapshot's diaspora totals at each endpoint

`MODELLED_FEATURES` are derived from the synthetic religion / integration
indices in data/raw/generate_cultural_datasets.py. The served model excludes
them so its predictions rest on measured data only.
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
    # Development, demography, connectivity, safety and governance
    "gdp_pc_ppp", "gdp_growth", "inflation", "remittances_gdp", "trade_openness",
    "resource_rents", "agri_employment", "urban_pct", "pop_density", "youth_share",
    "elderly_share", "fertility_rate", "life_expectancy", "internet_users",
    "mobile_subscriptions", "homicide_rate", "political_stability",
    "government_effectiveness", "voice_accountability", "rule_of_law",
    "refugees_abroad_per_1k", "refugees_hosted_per_1k", "idps_per_1k",
    "landlocked", "area_km2",
]

ORIGINAL_FEATURES = [
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

MODELLED_FEATURES = [
    "origin_cultural_openness_index", "dest_cultural_openness_index",
    "cultural_openness_diff", "dest_integration_policy_index",
    "dest_labour_market_integration", "dest_anti_discrimination_score",
    "dest_naturalization_access_score",
    "same_dominant_religion", "origin_religiosity_importance_score",
    "dest_religiosity_importance_score", "origin_religious_diversity_index",
    "dest_religious_diversity_index",
]

GEOGRAPHY_FEATURES = [
    "log_dist", "contig", "comlang_off", "comlang_ethno", "colony", "comcol",
    "col45", "smctry", "origin_landlocked", "dest_landlocked",
    "origin_log_area", "dest_log_area",
]

_DEVELOPMENT_NODE = [
    "log_gdp_ppp", "gdp_growth", "inflation", "remittances_gdp", "trade_openness",
    "resource_rents", "agri_employment", "urban_pct", "log_pop_density", "youth_share",
    "elderly_share", "fertility_rate", "life_expectancy", "internet_users",
    "mobile_subscriptions", "homicide_rate", "political_stability",
    "government_effectiveness", "voice_accountability", "rule_of_law",
    "refugees_abroad_per_1k", "refugees_hosted_per_1k", "idps_per_1k",
]
DEVELOPMENT_FEATURES = (
    [f"{side}_{c}" for side in ("origin", "dest") for c in _DEVELOPMENT_NODE]
    + ["gdp_ppp_log_diff", "political_stability_diff", "life_expectancy_diff", "youth_share_diff"]
)

NETWORK_FEATURES = ["origin_emigrants_lag", "dest_immigrants_lag"]

FEATURE_GROUPS = {
    "original": ORIGINAL_FEATURES,
    "geography": GEOGRAPHY_FEATURES,
    "development": DEVELOPMENT_FEATURES,
    "network": NETWORK_FEATURES,
}

ALL_FEATURES = ORIGINAL_FEATURES + GEOGRAPHY_FEATURES + DEVELOPMENT_FEATURES + NETWORK_FEATURES
# Served model: every group, minus the synthetic indices
FEATURE_COLUMNS = [c for c in ALL_FEATURES if c not in MODELLED_FEATURES]

_GEO_PAIR_COLS = ["dist_km", "contig", "comlang_off", "comlang_ethno", "colony", "comcol", "col45", "smctry"]


def _safe_log(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0))


def network_lags(migration: pd.DataFrame) -> dict:
    """
    Lookup tables for the network features: total emigrants per origin, total
    immigrants per destination, and each corridor's own stock, by snapshot year.
    """
    m = migration[(migration["origin_iso3"] != migration["dest_iso3"]) & (migration["migrant_stock"] > 0)]
    years = sorted(m["year"].unique())
    return {
        "prev_year": dict(zip(years[1:], years[:-1])),
        "emigrants": m.groupby(["origin_iso3", "year"])["migrant_stock"].sum(),
        "immigrants": m.groupby(["dest_iso3", "year"])["migrant_stock"].sum(),
        "corridor": m.groupby(["origin_iso3", "dest_iso3", "year"])["migrant_stock"].sum(),
    }


def _attach_network(df: pd.DataFrame, lags: dict) -> pd.DataFrame:
    """
    Diaspora size at each endpoint in the previous snapshot, excluding the
    corridor itself, so the features carry no information about the target
    corridor and remain valid for country pairs the model has never seen.
    """
    lag_year = df["year"].map(lags["prev_year"])
    has_lag = lag_year.notna()
    ly = lag_year.fillna(-1).astype(int)

    def lookup(series, keys):
        idx = pd.MultiIndex.from_arrays(keys)
        return pd.Series(series.reindex(idx).to_numpy(), index=df.index)

    own = lookup(lags["corridor"], [df["origin_iso3"], df["dest_iso3"], ly]).fillna(0)
    emig = lookup(lags["emigrants"], [df["origin_iso3"], ly]).fillna(0) - own
    immig = lookup(lags["immigrants"], [df["dest_iso3"], ly]).fillna(0) - own
    df["origin_emigrants_lag"] = np.log1p(emig.clip(lower=0)).where(has_lag)
    df["dest_immigrants_lag"] = np.log1p(immig.clip(lower=0)).where(has_lag)
    return df


def build_dyad_features(
    pairs: pd.DataFrame,
    factors: pd.DataFrame,
    country_meta: pd.DataFrame | None = None,
    geography: pd.DataFrame | None = None,
    lags: dict | None = None,
) -> pd.DataFrame:
    """
    Attach model features to a frame of (origin_iso3, dest_iso3, year) rows.

    `geography` is the CEPII pair table (data.loader.load_geography_dyadic) and
    `lags` the output of network_lags(); when either is omitted its features
    stay NaN. Any extra columns on `pairs` (e.g. the training target) are
    preserved. Missing values stay NaN; the gradient-boosting model handles them.
    """
    node_cols = [c for c in _NODE_COLS if c in factors.columns]
    node = factors[["iso3", "year"] + node_cols].drop_duplicates(["iso3", "year"])

    df = pairs.copy()
    df["year"] = df["year"].astype(int)
    for side in ("origin", "dest"):
        renamed = node.rename(columns={c: f"{side}_{c}" for c in node_cols})
        renamed = renamed.rename(columns={"iso3": f"{side}_iso3"})
        df = df.merge(renamed, on=[f"{side}_iso3", "year"], how="left")
        # Columns absent from this factors panel become NaN rather than errors
        for c in _NODE_COLS:
            if f"{side}_{c}" not in df.columns:
                df[f"{side}_{c}"] = np.nan

    for side in ("origin", "dest"):
        df[f"{side}_log_gdp"] = _safe_log(df[f"{side}_gdp_per_capita"])
        df[f"{side}_log_pop"] = _safe_log(df[f"{side}_population"])
        df[f"{side}_log_gdp_ppp"] = _safe_log(df[f"{side}_gdp_pc_ppp"])
        df[f"{side}_log_pop_density"] = _safe_log(df[f"{side}_pop_density"])
        df[f"{side}_log_area"] = _safe_log(df[f"{side}_area_km2"])

    df["gdp_log_diff"] = df["origin_log_gdp"] - df["dest_log_gdp"]
    df["pop_log_diff"] = df["origin_log_pop"] - df["dest_log_pop"]
    df["gdp_ppp_log_diff"] = df["origin_log_gdp_ppp"] - df["dest_log_gdp_ppp"]
    for c in ("political_stability", "life_expectancy", "youth_share"):
        if f"origin_{c}" in df.columns:
            df[f"{c}_diff"] = df[f"origin_{c}"] - df[f"dest_{c}"]
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

    if geography is not None and not geography.empty:
        geo = geography[["origin_iso3", "dest_iso3"] + _GEO_PAIR_COLS]
        df = df.merge(geo, on=["origin_iso3", "dest_iso3"], how="left")
        df["log_dist"] = _safe_log(df["dist_km"])

    if lags is not None:
        df = _attach_network(df, lags)

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    return df
