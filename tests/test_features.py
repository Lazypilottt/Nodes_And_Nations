"""Dyad features: network lags must exclude the corridor's own stock (no target leakage)."""
import numpy as np
import pandas as pd

from data.features import FEATURE_COLUMNS, MODELLED_FEATURES, build_dyad_features, network_lags


def _migration():
    return pd.DataFrame([
        {"origin_iso3": "AAA", "dest_iso3": "BBB", "year": 1990, "migrant_stock": 100.0},
        {"origin_iso3": "AAA", "dest_iso3": "CCC", "year": 1990, "migrant_stock": 50.0},
        {"origin_iso3": "DDD", "dest_iso3": "BBB", "year": 1990, "migrant_stock": 30.0},
        {"origin_iso3": "AAA", "dest_iso3": "BBB", "year": 1995, "migrant_stock": 120.0},
    ])


def _factors():
    return pd.DataFrame([
        {"iso3": iso, "year": y, "gdp_per_capita": 1000.0, "population": 1e6}
        for iso in ("AAA", "BBB", "CCC", "DDD") for y in (1990, 1995)
    ])


def test_network_lags_exclude_own_corridor():
    pairs = pd.DataFrame([{"origin_iso3": "AAA", "dest_iso3": "BBB", "year": 1995}])
    out = build_dyad_features(pairs, _factors(), lags=network_lags(_migration()))
    # AAA had 150 emigrants in 1990, 100 of them in this corridor
    assert np.isclose(out["origin_emigrants_lag"].iloc[0], np.log1p(50))
    # BBB had 130 immigrants in 1990, 100 of them from AAA
    assert np.isclose(out["dest_immigrants_lag"].iloc[0], np.log1p(30))


def test_first_snapshot_has_no_lag():
    pairs = pd.DataFrame([{"origin_iso3": "AAA", "dest_iso3": "BBB", "year": 1990}])
    out = build_dyad_features(pairs, _factors(), lags=network_lags(_migration()))
    assert out["origin_emigrants_lag"].isna().all()


def test_served_features_exclude_modelled_indices():
    assert not set(FEATURE_COLUMNS) & set(MODELLED_FEATURES)


def test_all_feature_columns_present():
    pairs = pd.DataFrame([{"origin_iso3": "AAA", "dest_iso3": "BBB", "year": 1995}])
    out = build_dyad_features(pairs, _factors())
    assert set(FEATURE_COLUMNS) <= set(out.columns)
