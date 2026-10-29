"""PPML with fixed effects must recover known coefficients and match a dummy-variable GLM."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from data.gravity import ols_fe, ppml_fe


def _synthetic(n_countries=25, seed=0):
    rng = np.random.default_rng(seed)
    iso = [f"C{i:02d}" for i in range(n_countries)]
    alpha = rng.normal(0, 1, n_countries)
    gamma = rng.normal(0, 1, n_countries)
    rows = []
    for i in range(n_countries):
        for j in range(n_countries):
            if i == j:
                continue
            log_dist = rng.uniform(5, 9)
            contig = float(rng.random() < 0.1)
            mu = np.exp(4 + alpha[i] + gamma[j] - 1.2 * log_dist + 0.8 * contig + 6)
            rows.append({"origin_iso3": iso[i], "dest_iso3": iso[j], "log_dist": log_dist,
                         "contig": contig, "y": rng.poisson(mu)})
    return pd.DataFrame(rows)


def test_ppml_recovers_coefficients():
    df = _synthetic()
    res = ppml_fe(df, "y", ["log_dist", "contig"])
    assert abs(res.coef["log_dist"] + 1.2) < 0.05
    assert abs(res.coef["contig"] - 0.8) < 0.1


def test_ppml_matches_dummy_glm():
    df = _synthetic(n_countries=15, seed=1)
    res = ppml_fe(df, "y", ["log_dist", "contig"])
    dummies = pd.get_dummies(df[["origin_iso3", "dest_iso3"]], dtype=float)
    X = pd.concat([df[["log_dist", "contig"]], dummies.iloc[:, :-1]], axis=1)
    glm = sm.GLM(df["y"], X, family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(res.coef.values, glm.params[["log_dist", "contig"]].values, rtol=1e-6)


def test_ppml_keeps_zeros_and_ols_drops_them():
    df = _synthetic(n_countries=12, seed=2)
    df.loc[df.sample(frac=0.2, random_state=0).index, "y"] = 0
    assert ppml_fe(df, "y", ["log_dist"]).n_obs == len(df)
    assert ols_fe(df, "y", ["log_dist"]).n_obs == int((df["y"] > 0).sum())


def test_fitted_values_match_totals_with_fixed_effects():
    # With origin and destination fixed effects, PPML fitted values reproduce each
    # origin's and destination's total exactly (the FE first-order conditions).
    df = _synthetic(n_countries=12, seed=3)
    res = ppml_fe(df, "y", ["log_dist", "contig"])
    fitted = res.fitted.reindex(df.index)
    by_o = df.assign(f=fitted).groupby("origin_iso3")[["y", "f"]].sum()
    np.testing.assert_allclose(by_o["f"], by_o["y"], rtol=1e-6)


def test_single_fixed_effect_matches_dummy_glm():
    df = _synthetic(n_countries=10, seed=4)
    df["dest_size"] = df["dest_iso3"].str[1:].astype(float) / 10
    res = ppml_fe(df, "y", ["log_dist", "contig", "dest_size"], fe=("origin_iso3",))
    dummies = pd.get_dummies(df["origin_iso3"], dtype=float)
    X = pd.concat([df[["log_dist", "contig", "dest_size"]], dummies], axis=1)
    glm = sm.GLM(df["y"], X, family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(res.coef.values, glm.params[["log_dist", "contig", "dest_size"]].values, rtol=1e-6)
