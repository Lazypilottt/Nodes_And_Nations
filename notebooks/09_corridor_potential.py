"""
Phase 9: Corridor Potential
============================
Nodes and Nations: A Complex Network Study of Global Migration

Which corridors are far larger or far smaller than geography and each
country's overall migration would predict?

A PPML gravity model with origin and destination fixed effects is fitted to
every country pair, including the zeros. The fixed effects absorb how much
each country sends and receives overall; distance, borders, language and
colonial history then predict how that migration should be spread across
pairs. Comparing actual with expected stock separates:

  * under-performing corridors: much smaller than expected. These point to
    barriers (visas, recognition of qualifications, cost of moving) and are
    candidates for labour-mobility agreements or diaspora outreach;
  * over-performing corridors: much larger than expected. These are channels
    the model cannot see (refugee movements, recruitment schemes, historical
    ties, free movement) and are where consular and integration services
    matter most.

Inputs:  data/processed/migration_long.csv, data/raw/cepii_dyadic.csv
Outputs: data/exports/corridor_potential.csv
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.gravity import ppml_fe
from data.loader import paths, load_country_metadata, load_geography_dyadic, load_migration_long

EXPORTS_DIR = paths["EXPORTS_DIR"]
YEAR = 2020                 # last observed snapshot
MIN_SIZE = 2_000            # keep pairs where actual or expected stock reaches this
TERMS = ["log_dist", "contig", "comlang_ethno", "colony", "comcol", "smctry", "same_region"]


def main():
    print("=" * 70)
    print("PHASE 9: CORRIDOR POTENTIAL")
    print("=" * 70)

    migration = load_migration_long()
    geography = load_geography_dyadic()
    meta = load_country_metadata().set_index("iso3")

    countries = set(migration["origin_iso3"]) | set(migration["dest_iso3"])
    pairs = geography[geography["origin_iso3"].isin(countries) & geography["dest_iso3"].isin(countries)].copy()
    stock = migration[migration["year"] == YEAR][["origin_iso3", "dest_iso3", "migrant_stock"]]
    pairs = pairs.merge(stock, on=["origin_iso3", "dest_iso3"], how="left")
    pairs["migrant_stock"] = pairs["migrant_stock"].fillna(0.0)
    pairs["log_dist"] = np.log(pairs["dist_km"])
    region = meta["un_region"]
    pairs["same_region"] = (pairs["origin_iso3"].map(region) == pairs["dest_iso3"].map(region)).astype(float)
    pairs["pair_id"] = ["-".join(sorted(p)) for p in zip(pairs["origin_iso3"], pairs["dest_iso3"])]

    res = ppml_fe(pairs, "migrant_stock", TERMS, cluster="pair_id")
    print(f"\n  PPML on {res.n_obs:,} pairs ({YEAR}), pseudo-R² {res.pseudo_r2:.3f}")
    print(res.table()[["term", "coefficient", "std_error"]].round(3).to_string(index=False))

    pairs["expected"] = res.fitted.reindex(pairs.index)
    pairs = pairs.dropna(subset=["expected"])
    pairs = pairs[(pairs["migrant_stock"] >= MIN_SIZE) | (pairs["expected"] >= MIN_SIZE)].copy()
    pairs["ratio"] = (pairs["migrant_stock"] + 100) / (pairs["expected"] + 100)
    pairs["log_ratio"] = np.log(pairs["ratio"])
    pairs["gap"] = pairs["migrant_stock"] - pairs["expected"]
    pairs["status"] = np.select(
        [pairs["ratio"] <= 0.25, pairs["ratio"] >= 4],
        ["under", "over"],
        default="as_expected",
    )
    pairs["origin_name"] = pairs["origin_iso3"].map(meta["country_name"])
    pairs["dest_name"] = pairs["dest_iso3"].map(meta["country_name"])

    out = pairs[["origin_iso3", "dest_iso3", "origin_name", "dest_name", "migrant_stock", "expected",
                 "ratio", "log_ratio", "gap", "status", "dist_km", "contig", "comlang_ethno", "colony"]].copy()
    out[["expected", "gap", "dist_km"]] = out[["expected", "gap", "dist_km"]].round(0)
    out[["ratio", "log_ratio"]] = out[["ratio", "log_ratio"]].round(4)
    out = out.sort_values("log_ratio")
    # Country names are in country_metadata.csv; leaving them out keeps the export small
    out.drop(columns=["origin_name", "dest_name"]).to_csv(
        os.path.join(EXPORTS_DIR, "corridor_potential.csv"), index=False)

    print(f"\n  ✓ corridor_potential.csv: {len(out):,} corridors "
          f"({(out['status'] == 'under').sum()} under, {(out['status'] == 'over').sum()} over)")
    show = ["origin_name", "dest_name", "migrant_stock", "expected", "ratio"]
    big_under = out[out["status"] == "under"].nsmallest(10, "gap")
    big_over = out[out["status"] == "over"].nlargest(10, "gap")
    print("\n  Largest shortfalls (expected − actual):")
    print(big_under[show].round(2).to_string(index=False))
    print("\n  Largest excesses (actual − expected):")
    print(big_over[show].round(2).to_string(index=False))

    print("\n" + "=" * 70)
    print("PHASE 9 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
