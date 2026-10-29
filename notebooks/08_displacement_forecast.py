"""
Phase 8: Displacement Forecast
===============================
Nodes and Nations: A Complex Network Study of Global Migration

When a crisis forces people out of a country, where will they go? Humanitarian
agencies, host governments and donors need that answer in the first weeks, to
pre-position aid and plan reception capacity.

This step fits a destination-choice model on 25 years of UNHCR data (refugees,
asylum seekers and other people in need of international protection, by origin
and host country) and back-tests it on real crises: trained only on years
before each crisis began, how well would it have predicted where people went?

Model: PPML with origin × year fixed effects, so it explains how each origin's
displaced population splits across destinations:

    E[displaced_odt] = exp(α_ot + β · x_odt)

x_odt: distance (and its square), shared border, common language, colonial
link, the existing diaspora (UN DESA stock at the last snapshot before t), and
the destination's income, population, political stability, armed conflict and
the refugees it already hosts from other origins (all measured the year before
t). A second model adds where the origin's own refugees already were the year
before. The forecast averages the two: in the back-tests the blend is more
robust than either (the second helps when a crisis escalates an existing exodus,
the first when displacement is new).

Inputs:  UNHCR population API (cached in data/raw/cache/unhcr_bilateral.csv),
         data/raw/cepii_dyadic.csv, data/processed/migration_long.csv,
         data/raw/cache/worldbank/*.csv, data/raw/cache/ucdp_conflict.csv
Outputs: data/exports/displacement_model.csv          coefficients
         data/exports/displacement_backtest.csv       accuracy per crisis, model vs baselines
         data/exports/displacement_backtest_shares.csv predicted vs actual shares per crisis
         data/exports/displacement_shares.csv         forecast shares for every origin today
         data/exports/displacement_meta.json
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.gravity import ppml_fe
from data.loader import paths, load_geography_dyadic, load_migration_long

CACHE_DIR = paths["CACHE_DIR"]
EXPORTS_DIR = paths["EXPORTS_DIR"]

YEARS = list(range(2000, 2025))
MIN_ORIGIN_TOTAL = 10_000      # origins with fewer displaced abroad add noise, not signal
TOP_DESTINATIONS = 60          # destinations kept per origin in the served forecast

FEATURES = {
    "log_dist": "Distance (log km)",
    "log_dist2": "Distance squared (log km²)",
    "contig": "Shared border",
    "comlang_ethno": "Common language",
    "colony": "Colonial link",
    "log_diaspora": "Existing diaspora (log)",
    "dest_log_gdp_ppp": "Destination GDP per capita, PPP (log)",
    "dest_log_pop": "Destination population (log)",
    "dest_stability": "Destination political stability",
    "dest_log_conflict": "Destination armed conflict (log)",
    "dest_log_hosted": "Refugees already hosted per 1,000 (log)",
}
FEATURES_PRIOR = {**FEATURES, "log_prior": "Own refugees there last year (log)"}
MODELS = {"network": FEATURES, "network_prior": FEATURES_PRIOR}

# origin, onset year, evaluation year, label. The model for each crisis is
# trained only on years before onset and fed only pre-onset information.
CRISES = [
    ("SYR", 2012, 2016, "Syria (2012–)"),
    ("SSD", 2014, 2017, "South Sudan (2014–)"),
    ("VEN", 2017, 2020, "Venezuela (2017–)"),
    ("MMR", 2017, 2018, "Myanmar, Rohingya (2017–)"),
    ("AFG", 2021, 2023, "Afghanistan (2021–)"),
    ("UKR", 2022, 2023, "Ukraine (2022–)"),
    ("SDN", 2023, 2024, "Sudan (2023–)"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_unhcr_bilateral() -> pd.DataFrame:
    """Refugees + asylum seekers + others in need of protection, origin × host × year."""
    path = os.path.join(CACHE_DIR, "unhcr_bilateral.csv")
    if os.path.exists(path):
        print("  [cache] Loading unhcr_bilateral.csv")
        return pd.read_csv(path)

    print("  Downloading UNHCR origin × host statistics...")
    frames = []
    for start in range(YEARS[0], YEARS[-1] + 1, 3):
        end = min(start + 2, YEARS[-1])
        url = (f"https://api.unhcr.org/population/v1/population/?yearFrom={start}&yearTo={end}"
               f"&coo_all=true&coa_all=true&limit=100000")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                frames.append(pd.DataFrame(r.json()["items"]))
                break
            except Exception as e:
                print(f"    UNHCR API note ({start}–{end}, attempt {attempt + 1}): {e}")
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"UNHCR download failed for {start}–{end}")
    d = pd.concat(frames, ignore_index=True)
    for c in ("refugees", "asylum_seekers", "oip"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["displaced"] = d["refugees"] + d["asylum_seekers"] + d["oip"]
    d = d.rename(columns={"coo_iso": "origin_iso3", "coa_iso": "dest_iso3"})
    d = d[(d["origin_iso3"].str.len() == 3) & (d["dest_iso3"].str.len() == 3)
          & (d["origin_iso3"] != d["dest_iso3"]) & (d["displaced"] > 0)]
    d = d.groupby(["year", "origin_iso3", "dest_iso3"], as_index=False)["displaced"].sum()
    d.to_csv(path, index=False)
    print(f"  ✓ unhcr_bilateral.csv: {len(d):,} rows")
    return d


def annual_node_table(countries: set) -> pd.DataFrame:
    """Destination attributes per country and year, carried forward over gaps."""
    wb = os.path.join(CACHE_DIR, "worldbank")
    node = None
    for name in ("gdp_pc_ppp", "population", "political_stability"):
        d = pd.read_csv(os.path.join(wb, f"{name}.csv"))
        node = d if node is None else node.merge(d, on=["iso3", "year"], how="outer")
    conflict = pd.read_csv(os.path.join(CACHE_DIR, "ucdp_conflict.csv"))
    node = node.merge(conflict, on=["iso3", "year"], how="left")
    grid = pd.MultiIndex.from_product([sorted(countries), range(1995, 2026)], names=["iso3", "year"])
    node = node.set_index(["iso3", "year"]).reindex(grid).reset_index().sort_values(["iso3", "year"])
    for c in ("gdp_pc_ppp", "population", "political_stability"):
        node[c] = node.groupby("iso3")[c].transform(lambda s: s.ffill().bfill())
    node["conflict_intensity"] = node["conflict_intensity"].fillna(0)
    node["dest_log_gdp_ppp"] = np.log(node["gdp_pc_ppp"])
    node["dest_log_pop"] = np.log(node["population"])
    node["dest_stability"] = node["political_stability"]
    node["dest_log_conflict"] = np.log1p(node["conflict_intensity"])
    return node[["iso3", "year", "population", "dest_log_gdp_ppp", "dest_log_pop",
                 "dest_stability", "dest_log_conflict"]]


class FeatureBuilder:
    """Builds x_odt for any origin and year using only information available before t."""

    def __init__(self, unhcr: pd.DataFrame, migration: pd.DataFrame, geography: pd.DataFrame):
        self.countries = sorted(set(migration["origin_iso3"]) | set(migration["dest_iso3"]))
        self.unhcr = unhcr
        m = migration[migration["origin_iso3"] != migration["dest_iso3"]]
        self.diaspora = m.set_index(["origin_iso3", "dest_iso3", "year"])["migrant_stock"]
        self.snapshots = sorted(m["year"].unique())
        geo = geography[geography["origin_iso3"].isin(self.countries) & geography["dest_iso3"].isin(self.countries)]
        self.geo = geo.set_index(["origin_iso3", "dest_iso3"])
        self.node = annual_node_table(set(self.countries)).set_index(["iso3", "year"])
        self.hosted = unhcr.groupby(["dest_iso3", "year"])["displaced"].sum()
        self.flows = unhcr.set_index(["origin_iso3", "dest_iso3", "year"])["displaced"]

    def snapshot_before(self, year: int, allow_extrapolated: bool = False) -> int:
        observed = [s for s in self.snapshots if s <= year - 1 and (allow_extrapolated or s <= 2020)]
        return max(observed)

    def build(self, origins: list[str], year: int, serving: bool = False) -> pd.DataFrame:
        snap = self.snapshot_before(year + (1 if serving else 0), allow_extrapolated=serving)
        prev = year - 1 if not serving else year
        rows = pd.DataFrame(
            [(o, d) for o in origins for d in self.countries if d != o],
            columns=["origin_iso3", "dest_iso3"],
        )
        rows["year"] = year
        g = self.geo.reindex(pd.MultiIndex.from_frame(rows[["origin_iso3", "dest_iso3"]]))
        rows["dist_km"] = g["dist_km"].to_numpy()
        rows["log_dist"] = np.log(rows["dist_km"])
        rows["log_dist2"] = rows["log_dist"] ** 2
        for c in ("contig", "comlang_ethno", "colony"):
            rows[c] = g[c].to_numpy()
        dia = self.diaspora.reindex(pd.MultiIndex.from_arrays(
            [rows["origin_iso3"], rows["dest_iso3"], np.full(len(rows), snap)])).to_numpy()
        rows["diaspora"] = np.nan_to_num(dia)
        rows["log_diaspora"] = np.log1p(rows["diaspora"])
        node = self.node.reindex(pd.MultiIndex.from_arrays([rows["dest_iso3"], np.full(len(rows), prev)]))
        for c in ("dest_log_gdp_ppp", "dest_log_pop", "dest_stability", "dest_log_conflict"):
            rows[c] = node[c].to_numpy()
        # Refugees the destination already hosts from other origins, per 1,000 residents
        hosted = self.hosted.reindex(pd.MultiIndex.from_arrays([rows["dest_iso3"], np.full(len(rows), prev)])).fillna(0).to_numpy()
        own = self.flows.reindex(pd.MultiIndex.from_arrays(
            [rows["origin_iso3"], rows["dest_iso3"], np.full(len(rows), prev)])).fillna(0).to_numpy()
        pop = node["population"].to_numpy()
        rows["dest_log_hosted"] = np.log1p(np.clip(hosted - own, 0, None) / pop * 1000)
        rows["log_prior"] = np.log1p(own)
        rows["prior_displaced"] = own
        rows["dest_population"] = pop
        return rows

    def training_frame(self, years: list[int]) -> pd.DataFrame:
        frames = []
        totals = self.unhcr.groupby(["origin_iso3", "year"])["displaced"].sum()
        for t in years:
            origins = [o for o in self.countries if totals.get((o, t), 0) >= MIN_ORIGIN_TOTAL]
            if not origins:
                continue
            f = self.build(origins, t)
            y = self.flows.reindex(pd.MultiIndex.from_arrays([f["origin_iso3"], f["dest_iso3"], f["year"]]))
            f["displaced"] = y.fillna(0).to_numpy()
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
        df["origin_year"] = df["origin_iso3"] + "_" + df["year"].astype(str)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL, BASELINES AND BACK-TESTS
# ══════════════════════════════════════════════════════════════════════════════

def fit(train: pd.DataFrame) -> dict:
    return {name: ppml_fe(train, "displaced", list(feats), fe=("origin_year",), cluster="origin_iso3")
            for name, feats in MODELS.items()}


def shares_from(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.clip(scores, 0, None))
    total = scores.sum()
    return scores / total if total > 0 else scores


def predict_shares(models: dict, frame: pd.DataFrame) -> np.ndarray:
    """Average of each model's destination shares."""
    out = []
    for res in models.values():
        X = frame[list(res.coef.index)].copy()
        # Impute the rare missing destination attribute with the column median
        X = X.fillna(X.median())
        eta = X.to_numpy(float) @ res.coef.to_numpy()
        out.append(shares_from(np.exp(eta - eta.max())))
    return np.mean(out, axis=0)


def evaluate(pred: np.ndarray, actual: np.ndarray, k: int = 5) -> dict:
    top_p = set(np.argsort(-pred)[:k])
    top_a = set(np.argsort(-actual)[:k])
    return {
        "overlap": float(np.minimum(pred, actual).sum()),
        "top5_hits": len(top_p & top_a),
    }


def backtest(builder: FeatureBuilder, train_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, share_rows = [], []
    for origin, onset, eval_year, label in CRISES:
        train = train_all[train_all["year"] < onset]
        res = fit(train)
        frame = builder.build([origin], onset)
        actual = builder.flows.reindex(pd.MultiIndex.from_arrays(
            [frame["origin_iso3"], frame["dest_iso3"], np.full(len(frame), eval_year)])).fillna(0).to_numpy()
        actual_total = actual.sum()
        actual = shares_from(actual)
        preds = {
            "model": predict_shares(res, frame),
            "diaspora_only": shares_from(frame["diaspora"].to_numpy()),
            "distance_only": shares_from(frame["dist_km"].to_numpy() ** -2.0),
        }
        for method, p in preds.items():
            rows.append({"origin_iso3": origin, "crisis": label, "onset": onset, "eval_year": eval_year,
                         "displaced_eval_year": actual_total, "method": method,
                         "train_years": f"{YEARS[0]}–{onset - 1}", **evaluate(p, actual)})
        keep = set(np.argsort(-actual)[:10]) | set(np.argsort(-preds["model"])[:10])
        for i in sorted(keep, key=lambda i: -actual[i]):
            share_rows.append({"origin_iso3": origin, "crisis": label, "dest_iso3": frame["dest_iso3"].iloc[i],
                               "actual_share": actual[i], "model_share": preds["model"][i],
                               "diaspora_share": preds["diaspora_only"][i]})
        m = rows[-3]
        print(f"  {label:<28} model places {m['overlap']:.0%} correctly, top-5 hits {m['top5_hits']}/5   "
              f"(diaspora-only {rows[-2]['overlap']:.0%}, distance-only {rows[-1]['overlap']:.0%})")
    return pd.DataFrame(rows), pd.DataFrame(share_rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 8: DISPLACEMENT FORECAST")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    unhcr = fetch_unhcr_bilateral()
    builder = FeatureBuilder(unhcr, load_migration_long(), load_geography_dyadic())
    train_all = builder.training_frame(YEARS)
    print(f"  Training frame: {len(train_all):,} origin–destination–year rows, "
          f"{train_all['origin_year'].nunique():,} origin-years")

    print("\n[2/4] Back-testing on past crises (trained only on earlier years)...")
    bt, bt_shares = backtest(builder, train_all)
    bt.to_csv(os.path.join(EXPORTS_DIR, "displacement_backtest.csv"), index=False)
    bt_shares.to_csv(os.path.join(EXPORTS_DIR, "displacement_backtest_shares.csv"), index=False)
    summary = bt.groupby("method")[["overlap", "top5_hits"]].mean()
    print(summary.round(3).to_string())

    print("\n[3/4] Fitting the full model (2000–2024)...")
    res = fit(train_all)
    tabs = []
    for name, r in res.items():
        t = r.table()
        t.insert(0, "model", name)
        t["label"] = t["term"].map(FEATURES_PRIOR)
        tabs.append(t)
        print(f"  {name}: pseudo-R² {r.pseudo_r2:.3f}, n = {r.n_obs:,}")
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(os.path.join(EXPORTS_DIR, "displacement_model.csv"), index=False)
    print(tab[["model", "label", "coefficient", "std_error", "p_value"]].round(3).to_string(index=False))

    print("\n[4/4] Forecasting destination shares for every origin...")
    latest = YEARS[-1]
    frame = builder.build(builder.countries, latest, serving=True)
    out = []
    for origin, f in frame.groupby("origin_iso3", sort=False):
        f = f.copy()
        f["share"] = predict_shares(res, f)
        out.append(f.nlargest(TOP_DESTINATIONS, "share"))
    shares = pd.concat(out, ignore_index=True)
    cols = ["origin_iso3", "dest_iso3", "share", "dist_km", "contig", "comlang_ethno", "colony",
            "diaspora", "prior_displaced", "dest_population"]
    shares = shares[cols]
    shares["share"] = shares["share"].round(6)
    shares["dist_km"] = shares["dist_km"].round(0)
    shares.to_csv(os.path.join(EXPORTS_DIR, "displacement_shares.csv"), index=False)
    print(f"  ✓ displacement_shares.csv: {len(shares):,} rows ({shares['origin_iso3'].nunique()} origins)")

    meta = {
        "train_years": f"{YEARS[0]}–{latest}",
        "n_obs": res["network"].n_obs,
        "pseudo_r2": {k: r.pseudo_r2 for k, r in res.items()},
        "features": FEATURES_PRIOR,
        "models": "average of 'network' and 'network_prior' PPML models",
        "diaspora_snapshot": int(builder.snapshot_before(latest + 1, allow_extrapolated=True)),
        "backtest_mean": summary.reset_index().to_dict(orient="records"),
        "displaced_definition": "UNHCR refugees + asylum seekers + other people in need of international protection",
    }
    with open(os.path.join(EXPORTS_DIR, "displacement_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
