"""
Phase 7: Migration Drivers
===========================
Nodes and Nations: A Complex Network Study of Global Migration

What do countries with a lot of immigration, or a lot of emigration, have in
common, and which pair attributes make a corridor large?

Steps:
  1. Country profiles: immigrant share of population and emigrant rate per
     country-year (countries with at least 1 million residents).
  2. Correlates: Spearman correlation of ~30 measured factors with both
     outcomes in 2020 (last observed snapshot), raw and controlling for income
     and population size, with bootstrap confidence intervals.
  3. Top vs bottom quintile profiles for each outcome.
  4. The migration hump: emigration against GDP per capita (PPP), pooled across
     countries and within countries over time.
  5. Structural gravity: PPML with origin and destination fixed effects for
     each snapshot, including zero corridors, next to OLS on log stock.

Inputs:  data/processed/migration_long.csv, data/processed/factors_panel.csv,
         data/raw/cepii_dyadic.csv
Outputs: data/exports/country_migration_profile.csv
         data/exports/drivers_correlations.csv
         data/exports/drivers_profile.csv
         data/exports/migration_hump.csv, migration_hump_curve.csv
         data/exports/gravity_ppml.csv, gravity_fit.csv
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.gravity import ols_fe, ppml_fe
from data.loader import (
    paths, load_country_metadata, load_factors, load_geography_dyadic, load_migration_long,
)

EXPORTS_DIR = paths["EXPORTS_DIR"]
REFERENCE_YEAR = 2020          # last snapshot that is observed, not extrapolated
OBSERVED_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
MIN_POPULATION = 1_000_000     # micro-states have extreme shares that swamp cross-country patterns
N_BOOT = 500
RNG = np.random.default_rng(42)

# factor -> (label, category, note). The note flags factors that are partly a
# mechanical consequence of migration rather than a candidate cause.
FACTORS = {
    "log_gdp_pc_ppp":              ("GDP per capita (PPP, log)", "Economy", ""),
    "gdp_growth":                  ("GDP growth", "Economy", ""),
    "inflation":                   ("Inflation", "Economy", ""),
    "unemployment":                ("Unemployment", "Economy", ""),
    "gini":                        ("Income inequality (Gini)", "Economy", ""),
    "trade_openness":              ("Trade openness", "Economy", ""),
    "resource_rents":              ("Natural-resource rents", "Economy", ""),
    "agri_employment":             ("Employment in agriculture", "Economy", ""),
    "female_labour_participation": ("Female labour participation", "Economy", ""),
    "remittances_gdp":             ("Remittances received", "Economy", "Largely a consequence of emigration"),
    "log_population":              ("Population (log)", "Demography", ""),
    "log_pop_density":             ("Population density (log)", "Demography", ""),
    "urban_pct":                   ("Urban population", "Demography", ""),
    "youth_share":                 ("Population under 15", "Demography", ""),
    "elderly_share":               ("Population 65+", "Demography", ""),
    "fertility_rate":              ("Fertility rate", "Demography", ""),
    "life_expectancy":             ("Life expectancy", "Demography", ""),
    "education_index":             ("Mean years of schooling", "Demography", ""),
    "internet_users":              ("Internet users", "Connectivity", ""),
    "mobile_subscriptions":        ("Mobile subscriptions", "Connectivity", ""),
    "tourism_per_capita":          ("Tourist arrivals per resident", "Connectivity", ""),
    "visa_openness_index":         ("Passport strength (visa-free access)", "Connectivity", ""),
    "political_stability":         ("Political stability", "Governance", ""),
    "government_effectiveness":    ("Government effectiveness", "Governance", ""),
    "voice_accountability":        ("Voice & accountability", "Governance", ""),
    "rule_of_law":                 ("Rule of law", "Governance", ""),
    "homicide_rate":               ("Homicide rate", "Safety", ""),
    "conflict_intensity":          ("Armed-conflict intensity", "Safety", ""),
    "idps_per_1k":                 ("Internally displaced per 1,000", "Safety", ""),
    "refugees_abroad_per_1k":      ("Refugees abroad per 1,000", "Safety", "Counted inside emigrant stock"),
    "refugees_hosted_per_1k":      ("Refugees hosted per 1,000", "Safety", "Counted inside immigrant stock"),
    "climate_vulnerability":       ("Climate vulnerability (ND-GAIN)", "Environment", ""),
    "landlocked":                  ("Landlocked", "Environment", ""),
    "log_area":                    ("Land area (log)", "Environment", ""),
}
CONTROLS = ["log_gdp_pc_ppp", "log_population"]
OUTCOMES = {
    "immigrant_share": "Immigrants as % of population",
    "emigrant_rate": "Emigrants as % of native-born population",
}
GRAVITY_TERMS = {
    "log_dist": "Distance (elasticity)",
    "contig": "Shared border",
    "comlang_ethno": "Common language",
    "colony": "Colonial link",
    "comcol": "Same former coloniser",
    "smctry": "Were one country",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. COUNTRY PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def country_profiles(migration: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    m = migration[(migration["origin_iso3"] != migration["dest_iso3"]) & (migration["migrant_stock"] > 0)]
    imm = m.groupby(["dest_iso3", "year"])["migrant_stock"].sum().rename("immigrants")
    emi = m.groupby(["origin_iso3", "year"])["migrant_stock"].sum().rename("emigrants")
    imm.index.names = emi.index.names = ["iso3", "year"]

    # The World Bank panel also holds regional aggregates (WLD, SSF, ...); keep
    # only the countries and territories that appear in the UN DESA matrix.
    countries = set(migration["origin_iso3"]) | set(migration["dest_iso3"])
    df = factors[factors["iso3"].isin(countries)]
    df = df.merge(imm.reset_index(), on=["iso3", "year"], how="left")
    df = df.merge(emi.reset_index(), on=["iso3", "year"], how="left")
    df[["immigrants", "emigrants"]] = df[["immigrants", "emigrants"]].fillna(0)
    df = df[df["population"] >= MIN_POPULATION].copy()

    df["immigrant_share"] = df["immigrants"] / df["population"] * 100
    # Emigrants live abroad, so they are not in the resident population
    native = df["population"] - df["immigrants"] + df["emigrants"]
    df["emigrant_rate"] = df["emigrants"] / native * 100

    df["log_gdp_pc_ppp"] = np.log(df["gdp_pc_ppp"].where(df["gdp_pc_ppp"] > 0))
    df["log_population"] = np.log(df["population"])
    df["log_pop_density"] = np.log(df["pop_density"].where(df["pop_density"] > 0))
    df["log_area"] = np.log(df["area_km2"].where(df["area_km2"] > 0))

    meta = load_country_metadata()[["iso3", "country_name", "continent"]]
    df = df.drop(columns=[c for c in ("country_name",) if c in df.columns]).merge(meta, on="iso3", how="left")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CORRELATES
# ══════════════════════════════════════════════════════════════════════════════

def _rank_residual(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Residual of ranked y after regressing on ranked controls."""
    ry = stats.rankdata(y)
    if Z.shape[1] == 0:
        return ry - ry.mean()
    RZ = np.column_stack([np.ones(len(y))] + [stats.rankdata(z) for z in Z.T])
    beta, *_ = np.linalg.lstsq(RZ, ry, rcond=None)
    return ry - RZ @ beta


def partial_spearman(x, y, Z) -> float:
    rx, ry = _rank_residual(x, Z), _rank_residual(y, Z)
    return float(np.corrcoef(rx, ry)[0, 1])


def correlates(profile: pd.DataFrame) -> pd.DataFrame:
    snap = profile[profile["year"] == REFERENCE_YEAR]
    rows = []
    for outcome in OUTCOMES:
        for factor, (label, category, note) in FACTORS.items():
            controls = [c for c in CONTROLS if c != factor]
            d = snap[[factor, outcome] + controls].dropna()
            if len(d) < 30 or d[factor].nunique() < 2:
                continue
            x, y, Z = d[factor].to_numpy(float), d[outcome].to_numpy(float), d[controls].to_numpy(float)
            rho, p = stats.spearmanr(x, y)
            rho_p = partial_spearman(x, y, Z)
            # t-test for a partial correlation with k controls
            k, n = len(controls), len(d)
            t = rho_p * np.sqrt((n - 2 - k) / max(1e-12, 1 - rho_p ** 2))
            p_partial = 2 * stats.t.sf(abs(t), n - 2 - k)
            boot = []
            for _ in range(N_BOOT):
                i = RNG.integers(0, n, n)
                if np.unique(x[i]).size < 2:
                    continue
                boot.append(partial_spearman(x[i], y[i], Z[i]))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "outcome": outcome, "factor": factor, "label": label, "category": category,
                "note": note, "n": n, "rho": rho, "p_value": p,
                "rho_partial": rho_p, "p_partial": p_partial, "ci_low": lo, "ci_high": hi,
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXPORTS_DIR, "drivers_correlations.csv"), index=False)
    return out


def quintile_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    """Median of each factor among the top and bottom 20% of countries per outcome."""
    snap = profile[profile["year"] == REFERENCE_YEAR]
    rows = []
    for outcome in OUTCOMES:
        q_lo, q_hi = snap[outcome].quantile([0.2, 0.8])
        groups = {"top": snap[snap[outcome] >= q_hi], "bottom": snap[snap[outcome] <= q_lo], "all": snap}
        for factor, (label, category, _note) in FACTORS.items():
            row = {"outcome": outcome, "factor": factor, "label": label, "category": category}
            for g, d in groups.items():
                row[f"median_{g}"] = d[factor].median()
                row[f"n_{g}"] = int(d[factor].notna().sum())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXPORTS_DIR, "drivers_profile.csv"), index=False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. MIGRATION HUMP
# ══════════════════════════════════════════════════════════════════════════════

def migration_hump(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Log emigrant rate on log GDP per capita (PPP) and its square (Clemens,
    2020). Emigrant rates are heavily right-skewed, so the log keeps a handful
    of diaspora-heavy countries from driving the fit. The pooled specification
    compares countries; the within specification adds country fixed effects and
    asks whether a country's emigration rises as it grows.
    """
    d = profile[profile["year"].isin(OBSERVED_YEARS) & (profile["emigrant_rate"] > 0)][
        ["iso3", "year", "emigrant_rate", "log_gdp_pc_ppp"]].dropna().copy()
    d["log_rate"] = np.log(d["emigrant_rate"])
    d["lg"], d["lg2"] = d["log_gdp_pc_ppp"], d["log_gdp_pc_ppp"] ** 2
    year_fe = pd.get_dummies(d["year"], prefix="y", drop_first=True, dtype=float)
    country_fe = pd.get_dummies(d["iso3"], prefix="c", drop_first=True, dtype=float)

    rows, fits = [], {}
    for spec, extra in (("pooled", year_fe), ("within_country", pd.concat([year_fe, country_fe], axis=1))):
        X = sm.add_constant(pd.concat([d[["lg", "lg2"]], extra], axis=1))
        res = sm.OLS(d["log_rate"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso3"])[0]})
        b1, b2 = res.params["lg"], res.params["lg2"]
        peak = float(np.exp(-b1 / (2 * b2))) if b2 < 0 else np.nan
        rows.append({
            "spec": spec, "n_obs": int(res.nobs), "n_countries": d["iso3"].nunique(),
            "b_log_gdp": b1, "b_log_gdp_sq": b2, "p_log_gdp_sq": res.pvalues["lg2"],
            "peak_gdp_pc_ppp": peak, "r_squared": res.rsquared,
        })
        fits[spec] = res
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXPORTS_DIR, "migration_hump.csv"), index=False)

    # Curve for plotting: the pooled fit at the reference year, and binned means
    res = fits["pooled"]
    grid = np.linspace(d["lg"].quantile(0.01), d["lg"].quantile(0.99), 60)
    y_ref = f"y_{REFERENCE_YEAR}"
    shift = res.params.get(y_ref, 0.0)
    fitted = np.exp(res.params["const"] + shift + res.params["lg"] * grid + res.params["lg2"] * grid ** 2)
    curve = [pd.DataFrame({"kind": "fit", "gdp_pc_ppp": np.exp(grid), "emigrant_rate": fitted})]
    d["bin"] = pd.qcut(d["lg"], 12, labels=False)
    bins = d.groupby("bin").agg(gdp_pc_ppp=("lg", lambda s: float(np.exp(s.median()))),
                                emigrant_rate=("emigrant_rate", "median")).reset_index(drop=True)
    curve.append(bins.assign(kind="bin_median"))
    pd.concat(curve, ignore_index=True).to_csv(os.path.join(EXPORTS_DIR, "migration_hump_curve.csv"), index=False)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4. STRUCTURAL GRAVITY (PPML)
# ══════════════════════════════════════════════════════════════════════════════

def gravity(migration: pd.DataFrame, geography: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    countries = set(migration["origin_iso3"]) | set(migration["dest_iso3"])
    pairs = geography[geography["origin_iso3"].isin(countries) & geography["dest_iso3"].isin(countries)].copy()
    pairs["log_dist"] = np.log(pairs["dist_km"])
    pairs["pair_id"] = [
        "-".join(sorted(p)) for p in zip(pairs["origin_iso3"], pairs["dest_iso3"])
    ]
    terms = list(GRAVITY_TERMS)

    coef_rows, fit_rows = [], []
    for year in sorted(migration["year"].unique()):
        stock = migration[migration["year"] == year][["origin_iso3", "dest_iso3", "migrant_stock"]]
        d = pairs.merge(stock, on=["origin_iso3", "dest_iso3"], how="left")
        d["migrant_stock"] = d["migrant_stock"].fillna(0.0)
        for name, fitter in (("ppml", ppml_fe), ("ols_log", ols_fe)):
            res = fitter(d, "migrant_stock", terms, cluster="pair_id")
            tab = res.table()
            tab["estimator"], tab["year"] = name, int(year)
            tab["label"] = tab["term"].map(GRAVITY_TERMS)
            # Dummies: percentage change in stock when the tie is present
            tab["pct_effect"] = np.where(tab["term"] == "log_dist", np.nan, np.expm1(tab["coefficient"]) * 100)
            coef_rows.append(tab)
            fit_rows.append({
                "estimator": name, "year": int(year), "n_obs": res.n_obs,
                "zero_share": float((d["migrant_stock"] <= 0).mean()), "fit": res.pseudo_r2,
                "fit_measure": "pseudo-R² (corr² of fitted and actual)" if name == "ppml" else "within R²",
                "extrapolated": bool(year == 2025),
            })
        ppml_row = coef_rows[-2].set_index("term")
        print(f"  {year}: distance elasticity {ppml_row.loc['log_dist', 'coefficient']:+.3f}, "
              f"colonial link {ppml_row.loc['colony', 'pct_effect']:+.0f}%, "
              f"common language {ppml_row.loc['comlang_ethno', 'pct_effect']:+.0f}%")

    coefs = pd.concat(coef_rows, ignore_index=True)
    cols = ["estimator", "year", "term", "label", "coefficient", "std_error", "p_value",
            "ci_lower_95", "ci_upper_95", "pct_effect"]
    coefs[cols].to_csv(os.path.join(EXPORTS_DIR, "gravity_ppml.csv"), index=False)
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(os.path.join(EXPORTS_DIR, "gravity_fit.csv"), index=False)
    return coefs, fit


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 7: MIGRATION DRIVERS")
    print("=" * 70)

    migration = load_migration_long()
    factors = load_factors()
    geography = load_geography_dyadic()

    print("\n[1/4] Country migration profiles...")
    profile = country_profiles(migration, factors)
    keep = ["iso3", "country_name", "continent", "year", "population", "immigrants", "emigrants",
            "immigrant_share", "emigrant_rate", "gdp_pc_ppp"]
    profile[keep].to_csv(os.path.join(EXPORTS_DIR, "country_migration_profile.csv"), index=False)
    print(f"  ✓ country_migration_profile.csv: {len(profile):,} country-years")

    print(f"\n[2/4] Correlates of immigration and emigration ({REFERENCE_YEAR})...")
    corr = correlates(profile)
    quintile_profiles(profile)
    for outcome in OUTCOMES:
        top = corr[(corr["outcome"] == outcome) & (corr["p_partial"] < 0.05)]
        top = top.reindex(top["rho_partial"].abs().sort_values(ascending=False).index).head(8)
        print(f"  {OUTCOMES[outcome]} — strongest partial correlates (income & size held fixed):")
        for _, r in top.iterrows():
            print(f"    {r['label']:<38} ρ = {r['rho']:+.2f}   partial ρ = {r['rho_partial']:+.2f}")

    if geography.empty:
        print("\n  WARNING: data/raw/cepii_dyadic.csv missing; skipping the gravity step")

    print("\n[3/4] Migration hump...")
    hump = migration_hump(profile)
    print(hump.to_string(index=False))

    if not geography.empty:
        print("\n[4/4] Structural gravity (PPML with origin & destination fixed effects)...")
        gravity(migration, geography)

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
