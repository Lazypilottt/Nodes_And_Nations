"""
Phase 1: Data Collection & Cleaning
====================================
Nodes and Nations: A Complex Network Study of Global Migration

Steps:
  1. Parse UN DESA bilateral migration stock Excel → long format (covering all 235 countries/territories)
  2. Download/cache supplementary factor data (26 World Bank indicators incl. governance,
     UNHCR displacement, UNDP Education, UCDP Conflict, CEPII geography)
  3. Merge and harmonize Henley Passport Index and ND-GAIN Climate Vulnerability
  4. Extrapolate snapshots for complete 1990-2025 panel series
  5. Export migration_long.csv and factors_panel.csv

Outputs:  data/processed/migration_long.csv
          data/processed/factors_panel.csv
          data/processed/factors_panel_enriched.csv
"""

import os
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
import openpyxl
import sys
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Paths ──────────────────────────────────────────────────────────────────────
from data.loader import (
    paths, load_ndgain, load_henley, load_conflict_long, load_religions,
    load_cultural_integration, load_geography_country,
)

ROOT        = paths["ROOT"]
RAW_DIR     = paths["RAW_DIR"]
CACHE_DIR   = paths["CACHE_DIR"]
PROCESSED   = paths["PROCESSED_DIR"]
os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(PROCESSED,   exist_ok=True)

UNDESA_FILE    = os.path.join(RAW_DIR, "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
OFFICIAL_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
SNAPSHOT_YEARS = OFFICIAL_YEARS + [2025]
# Latest UN DESA estimate; parsed only to anchor the 2025 extrapolation
LATEST_OBSERVED_YEAR = 2024
PARSE_YEARS = OFFICIAL_YEARS + [LATEST_OBSERVED_YEAR]

API_TIMEOUT = 60   # seconds per request
WB_INDICATORS = {
    # Economy
    "gdp_per_capita":              "NY.GDP.PCAP.CD",
    "gdp_pc_ppp":                  "NY.GDP.PCAP.PP.KD",   # constant 2021 international $
    "gdp_growth":                  "NY.GDP.MKTP.KD.ZG",
    "inflation":                   "FP.CPI.TOTL.ZG",
    "unemployment":                "SL.UEM.TOTL.ZS",
    "remittances_gdp":             "BX.TRF.PWKR.DT.GD.ZS",
    "trade_openness":              "NE.TRD.GNFS.ZS",
    "resource_rents":              "NY.GDP.TOTL.RT.ZS",
    "agri_employment":             "SL.AGR.EMPL.ZS",
    "female_labour_participation": "SL.TLF.CACT.FE.ZS",
    "gini":                        "SI.POV.GINI",
    # Demography
    "population":                  "SP.POP.TOTL",
    "pop_density":                 "EN.POP.DNST",
    "urban_pct":                   "SP.URB.TOTL.IN.ZS",
    "youth_share":                 "SP.POP.0014.TO.ZS",
    "elderly_share":               "SP.POP.65UP.TO.ZS",
    "fertility_rate":              "SP.DYN.TFRT.IN",
    "life_expectancy":             "SP.DYN.LE00.IN",
    # Connectivity
    "internet_users":              "IT.NET.USER.ZS",
    "mobile_subscriptions":        "IT.CEL.SETS.P2",
    "tourism_arrivals":            "ST.INT.ARVL",
    # Safety & governance (Worldwide Governance Indicators, 1996+)
    "homicide_rate":               "VC.IHR.PSRC.P5",
    "political_stability":         "GOV_WGI_PV.EST",
    "government_effectiveness":    "GOV_WGI_GE.EST",
    "voice_accountability":        "GOV_WGI_VA.EST",
    "rule_of_law":                 "GOV_WGI_RL.EST",
}
# Stock-like levels: extrapolated to 2025 from recent growth when not yet published
WB_LEVELS = {"gdp_per_capita", "gdp_pc_ppp", "population"}
# Slow-moving institutional scores that start in 1996: back-filled to 1990/1995
WB_BACKFILL = {"political_stability", "government_effectiveness", "voice_accountability", "rule_of_law"}
# Annual values within ±SNAPSHOT_WINDOW years of a snapshot fill gaps in sparse series
SNAPSHOT_WINDOW = 2

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSE UN DESA (Full 235 Country Entities)
# ══════════════════════════════════════════════════════════════════════════════

M49_TO_ISO3 = {
    4: "AFG", 8: "ALB", 12: "DZA", 16: "ASM", 20: "AND", 24: "AGO", 28: "ATG", 31: "AZE",
    32: "ARG", 36: "AUS", 40: "AUT", 44: "BHS", 48: "BHR", 50: "BGD", 51: "ARM", 52: "BRB",
    56: "BEL", 60: "BMU", 64: "BTN", 68: "BOL", 70: "BIH", 72: "BWA", 76: "BRA", 84: "BLZ",
    90: "SLB", 92: "VGB", 96: "BRN", 100: "BGR", 104: "MMR", 108: "BDI", 112: "BLR", 116: "KHM",
    120: "CMR", 124: "CAN", 132: "CPV", 136: "CYM", 140: "CAF", 144: "LKA", 148: "TCD", 152: "CHL",
    156: "CHN", 158: "TWN", 170: "COL", 174: "COM", 175: "MYT", 178: "COG", 180: "COD", 184: "COK",
    188: "CRI", 191: "HRV", 192: "CUB", 196: "CYP", 203: "CZE", 204: "BEN", 208: "DNK", 212: "DMA",
    214: "DOM", 218: "ECU", 222: "SLV", 226: "GNQ", 231: "ETH", 232: "ERI", 233: "EST", 234: "FRO",
    238: "FLK", 242: "FJI", 246: "FIN", 250: "FRA", 254: "GUF", 258: "PYF", 262: "DJI", 266: "GAB",
    268: "GEO", 270: "GMB", 275: "PSE", 276: "DEU", 288: "GHA", 292: "GIB", 296: "KIR", 300: "GRC",
    304: "GRL", 308: "GRD", 312: "GLP", 316: "GUM", 320: "GTM", 324: "GIN", 328: "GUY", 332: "HTI",
    336: "VAT", 340: "HND", 344: "HKG", 348: "HUN", 352: "ISL", 356: "IND", 360: "IDN", 364: "IRN",
    368: "IRQ", 372: "IRL", 376: "ISR", 380: "ITA", 384: "CIV", 388: "JAM", 392: "JPN", 398: "KAZ",
    400: "JOR", 404: "KEN", 408: "PRK", 410: "KOR", 414: "KWT", 417: "KGZ", 418: "LAO", 422: "LBN",
    426: "LSO", 428: "LVA", 430: "LBR", 434: "LBY", 438: "LIE", 440: "LTU", 442: "LUX", 446: "MAC",
    450: "MDG", 454: "MWI", 458: "MYS", 462: "MDV", 466: "MLI", 470: "MLT", 474: "MTQ", 478: "MRT",
    480: "MUS", 484: "MEX", 492: "MCO", 496: "MNG", 498: "MDA", 499: "MNE", 500: "MSR", 504: "MAR",
    508: "MOZ", 512: "OMN", 516: "NAM", 520: "NRU", 524: "NPL", 528: "NLD", 531: "CUW", 533: "ABW",
    534: "SXM", 535: "BES", 540: "NCL", 548: "VUT", 554: "NZL", 558: "NIC", 562: "NER", 566: "NGA",
    570: "NIU", 578: "NOR", 580: "MNP", 583: "FSM", 584: "MHL", 585: "PLW", 586: "PAK", 591: "PAN",
    598: "PNG", 600: "PRY", 604: "PER", 608: "PHL", 616: "POL", 620: "PRT", 624: "GNB", 626: "TLS",
    630: "PRI", 634: "QAT", 638: "REU", 642: "ROU", 643: "RUS", 646: "RWA", 652: "BLM", 654: "SHN",
    659: "KNA", 660: "AIA", 662: "LCA", 663: "MAF", 666: "SPM", 670: "VCT", 674: "SMR", 678: "STP",
    682: "SAU", 686: "SEN", 688: "SRB", 690: "SYC", 694: "SLE", 702: "SGP", 703: "SVK", 704: "VNM",
    705: "SVN", 706: "SOM", 710: "ZAF", 716: "ZWE", 724: "ESP", 728: "SSD", 729: "SDN", 732: "ESH",
    740: "SUR", 748: "SWZ", 752: "SWE", 756: "CHE", 760: "SYR", 762: "TJK", 764: "THA", 768: "TGO",
    772: "TKL", 776: "TON", 780: "TTO", 784: "ARE", 788: "TUN", 792: "TUR", 795: "TKM", 796: "TCA",
    798: "TUV", 800: "UGA", 804: "UKR", 807: "MKD", 818: "EGY", 826: "GBR", 830: "CHI", 833: "IMN",
    834: "TZA", 840: "USA", 850: "VIR", 854: "BFA", 858: "URY", 860: "UZB", 862: "VEN", 876: "WLF",
    882: "WSM", 887: "YEM", 894: "ZMB"
}


def parse_undesa_fast(filepath: str) -> pd.DataFrame:
    """Fast parsing of UN DESA IMS Excel using openpyxl read_only mode."""
    cache_path = os.path.join(CACHE_DIR, "undesa_parsed.csv")
    if os.path.exists(cache_path):
        df_cached = pd.read_csv(cache_path)
        # Validate coverage, and that the cache was built with the both-sexes
        # columns including the latest observed year (older caches were not).
        if df_cached["dest_iso3"].nunique() >= 220 and str(LATEST_OBSERVED_YEAR) in df_cached.columns:
            print(f"  [cache] Loading undesa_parsed.csv ({len(df_cached):,} rows, {df_cached['dest_iso3'].nunique()} countries)")
            return df_cached

    print(f"  Parsing UN DESA Excel (openpyxl read_only)...")
    t0 = time.time()

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb["Table 1"]
    rows_iter = ws.iter_rows(values_only=True)

    for _ in range(10):
        next(rows_iter)
    header = next(rows_iter)

    # Table 1 repeats the year headers three times: both sexes, males, females.
    # Keep the first occurrence of each year so we read the both-sexes totals.
    year_col_indices = {}
    for col_idx, val in enumerate(header):
        if isinstance(val, int) and val in PARSE_YEARS and val not in year_col_indices:
            year_col_indices[val] = col_idx

    NAME_DEST = 1
    LOCD_DEST = 4
    NAME_ORIG = 5
    LOCD_ORIG = 6

    records = []
    for row in rows_iter:
        if len(row) <= LOCD_ORIG:
            continue
        dest_loc = row[LOCD_DEST]
        orig_loc = row[LOCD_ORIG]

        try:
            dest_loc = int(dest_loc)
            orig_loc = int(orig_loc)
        except (TypeError, ValueError):
            continue

        if dest_loc >= 900 or orig_loc >= 900 or dest_loc == orig_loc:
            continue

        dest_iso3 = M49_TO_ISO3.get(dest_loc)
        origin_iso3 = M49_TO_ISO3.get(orig_loc)
        if not dest_iso3 or not origin_iso3:
            continue

        rec = {
            "dest_name": row[NAME_DEST],
            "origin_name": row[NAME_ORIG],
            "dest_iso3": dest_iso3,
            "origin_iso3": origin_iso3,
        }
        for yr, ci in year_col_indices.items():
            val = row[ci] if ci < len(row) else None
            rec[str(yr)] = float(val) if isinstance(val, (int, float)) and val is not None else np.nan
        records.append(rec)

    wb.close()
    print(f"  Parsed {len(records):,} country-pair rows in {time.time()-t0:.1f}s")
    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"  [cache] Saved undesa_parsed.csv")
    return df


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot bilateral stocks from wide to long format."""
    id_cols = ["dest_iso3", "origin_iso3"]
    str_years = [str(y) for y in PARSE_YEARS]
    df_wide = df[id_cols + str_years].copy()

    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=str_years,
        var_name="year",
        value_name="migrant_stock",
    )
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long.dropna(subset=["migrant_stock"])
    df_long = df_long[df_long["migrant_stock"] > 0]
    df_long = df_long.groupby(["dest_iso3", "origin_iso3", "year"], as_index=False)["migrant_stock"].sum()
    print(f"  Long-format rows (non-zero flows): {len(df_long):,}")
    return df_long.reset_index(drop=True)


def extrapolate_2025(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Geometric extrapolation of 2025 migration stock.

    Uses the observed 2024 stock grown by one year at the corridor's annualised
    2020→2024 rate. Corridors first observed in 2024 carry their 2024 value.
    The 2024 rows are then dropped so the output keeps the 5-year snapshot grid.
    """
    print("  Extrapolating 2025 migration stocks from 2020→2024 trend...")
    base = LATEST_OBSERVED_YEAR
    s2020 = df_long[df_long["year"] == 2020].set_index(["dest_iso3", "origin_iso3"])["migrant_stock"]
    s_base = df_long[df_long["year"] == base].set_index(["dest_iso3", "origin_iso3"])["migrant_stock"]

    v_base = s_base.values
    v2020 = s2020.reindex(s_base.index).values
    with np.errstate(divide="ignore", invalid="ignore"):
        annual = np.where(v2020 > 0, (v_base / v2020) ** (1 / (base - 2020)), 1.0)
    annual = np.where(np.isfinite(annual) & (annual > 0), annual, 1.0)
    stock25 = np.maximum(v_base * annual ** (2025 - base), 1.0)

    df_2025 = pd.DataFrame({
        "dest_iso3": s_base.index.get_level_values(0),
        "origin_iso3": s_base.index.get_level_values(1),
        "year": 2025,
        "migrant_stock": stock25,
        "is_extrapolated": True,
    })
    df_long = df_long[df_long["year"] != base].copy()
    df_long["is_extrapolated"] = False
    result = pd.concat([df_long, df_2025], ignore_index=True)
    print(f"  Total rows after extrapolation: {len(result):,}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. WORLD BANK DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_wb_indicator(indicator_code: str, indicator_name: str) -> pd.DataFrame:
    """Fetch one World Bank indicator (all countries, 1985–2025)."""
    records = []
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
            f"?format=json&date=1985:2025&per_page=5000&page={page}"
        )
        data = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=API_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                print(f"    WB API note ({indicator_code} page {page}, attempt {attempt + 1}): {e}")
                time.sleep(2 * (attempt + 1))
        if data is None:
            # A partial download would silently bias the panel; fail loudly instead.
            raise RuntimeError(f"World Bank download failed for {indicator_code}")

        if len(data) < 2 or not data[1]:
            break
        for item in data[1]:
            if item.get("value") is not None and item.get("countryiso3code"):
                records.append({
                    "iso3": item["countryiso3code"],
                    "year": int(item["date"]),
                    indicator_name: float(item["value"]),
                })
        meta = data[0]
        if page >= meta.get("pages", 1):
            break
        page += 1

    df = pd.DataFrame(records)
    return df


def to_snapshots(annual: pd.DataFrame, col: str, level: bool = False) -> pd.DataFrame:
    """
    Collapse an annual (iso3, year, col) series onto the snapshot years.

    Each snapshot takes the exact-year value when published, otherwise the mean
    of values within ±SNAPSHOT_WINDOW years, which fills gaps in sparse series
    (household surveys, homicide statistics) without inventing trends. For 2025
    the most recent value since 2021 is used; stock-like levels (`level=True`)
    are grown from that value at the country's latest 3-year growth rate.
    """
    s = annual[["iso3", "year", col]].dropna()
    out = []
    for snap in SNAPSHOT_YEARS:
        if snap == SNAPSHOT_YEARS[-1]:
            recent = s[s["year"].between(snap - 4, snap)].sort_values("year")
            latest = recent.groupby("iso3").tail(1).set_index("iso3")
            vals = latest[col].copy()
            if level:
                base = s.set_index(["iso3", "year"])[col]
                prior = pd.Series(
                    [base.get((iso, yr - 3), np.nan) for iso, yr in zip(latest.index, latest["year"])],
                    index=latest.index,
                )
                growth = (latest[col] / prior) ** (1 / 3)
                growth = growth.where(np.isfinite(growth) & (growth > 0), 1.0)
                vals = latest[col] * growth ** (snap - latest["year"])
        else:
            exact = s[s["year"] == snap].set_index("iso3")[col]
            window = s[s["year"].between(snap - SNAPSHOT_WINDOW, snap + SNAPSHOT_WINDOW)]
            vals = window.groupby("iso3")[col].mean()
            vals.update(exact)
        out.append(pd.DataFrame({"iso3": vals.index, "year": snap, col: vals.values}))
    return pd.concat(out, ignore_index=True)


def fetch_all_worldbank() -> pd.DataFrame:
    """
    Download every indicator in WB_INDICATORS (cached one file per indicator in
    data/raw/cache/worldbank/) and return a snapshot-year panel.
    """
    wb_cache = os.path.join(CACHE_DIR, "worldbank")
    os.makedirs(wb_cache, exist_ok=True)

    panel = None
    for name, code in WB_INDICATORS.items():
        path = os.path.join(wb_cache, f"{name}.csv")
        if os.path.exists(path):
            annual = pd.read_csv(path)
        else:
            print(f"  Downloading {name} ({code})...")
            annual = fetch_wb_indicator(code, name)
            if annual.empty:
                print(f"    WARNING: no data for {code}; skipping")
                continue
            annual.to_csv(path, index=False)

        snap = to_snapshots(annual, name, level=name in WB_LEVELS)
        if name in WB_BACKFILL:
            grid = pd.MultiIndex.from_product([snap["iso3"].unique(), SNAPSHOT_YEARS], names=["iso3", "year"])
            snap = snap.set_index(["iso3", "year"]).reindex(grid).reset_index()
            snap[name] = snap.groupby("iso3")[name].bfill()
        panel = snap if panel is None else panel.merge(snap, on=["iso3", "year"], how="outer")

    print(f"  World Bank factors complete ({len(panel):,} rows, {panel['iso3'].nunique()} countries, "
          f"{len(WB_INDICATORS)} indicators)")
    return panel


def fetch_unhcr() -> pd.DataFrame:
    """
    UNHCR forced-displacement stocks per country and year (1990–latest):
    refugees and asylum seekers originating from a country, those hosted by it,
    and internally displaced people. Cached in data/raw/cache/unhcr_population.csv.
    """
    cache_path = os.path.join(CACHE_DIR, "unhcr_population.csv")
    if os.path.exists(cache_path):
        print("  [cache] Loading unhcr_population.csv")
        annual = pd.read_csv(cache_path)
    else:
        print("  Downloading UNHCR population statistics...")
        base = "https://api.unhcr.org/population/v1/population/?yearFrom=1990&yearTo=2025&limit=20000"
        frames = {}
        for side, flag, key in (("origin", "coo_all", "coo_iso"), ("asylum", "coa_all", "coa_iso")):
            try:
                r = requests.get(f"{base}&{flag}=true", timeout=API_TIMEOUT * 2)
                r.raise_for_status()
                items = pd.DataFrame(r.json().get("items", []))
            except Exception as e:
                print(f"    UNHCR API note ({side}): {e}")
                return pd.DataFrame(columns=["iso3", "year"])
            for c in ("refugees", "asylum_seekers", "idps"):
                items[c] = pd.to_numeric(items[c], errors="coerce").fillna(0)
            items = items.rename(columns={key: "iso3"})
            items = items[items["iso3"].str.len() == 3]
            frames[side] = items
        o, a = frames["origin"], frames["asylum"]
        annual = (
            o.assign(refugees_abroad=o["refugees"] + o["asylum_seekers"])[["iso3", "year", "refugees_abroad", "idps"]]
            .merge(a.assign(refugees_hosted=a["refugees"] + a["asylum_seekers"])[["iso3", "year", "refugees_hosted"]],
                   on=["iso3", "year"], how="outer")
        )
        annual.to_csv(cache_path, index=False)

    panel = None
    for col in ("refugees_abroad", "refugees_hosted", "idps"):
        snap = to_snapshots(annual, col)
        panel = snap if panel is None else panel.merge(snap, on=["iso3", "year"], how="outer")
    # UNHCR lists every country it reports on, so a missing value means none recorded
    panel[["refugees_abroad", "refugees_hosted", "idps"]] = panel[
        ["refugees_abroad", "refugees_hosted", "idps"]].fillna(0.0)
    print(f"  UNHCR displacement complete ({len(panel):,} rows)")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNDP EDUCATION INDEX
# ══════════════════════════════════════════════════════════════════════════════

def fetch_undp_education() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "undp_education.csv")
    if os.path.exists(cache_path):
        df_cached = pd.read_csv(cache_path)
        if 2025 in df_cached["year"].values and len(df_cached) > 6000:
            print(f"  [cache] Loading undp_education.csv")
            return df_cached

    print("  Downloading / harmonizing UNDP Education Index...")
    try:
        url = ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
               "HDR23-24_Composite_indices_complete_time_series.csv")
        df = pd.read_csv(url, encoding="latin-1", storage_options={"User-Agent": "Mozilla/5.0"})
        edu_cols = [c for c in df.columns if str(c).startswith("mys_") and len(str(c)) == 8]
        df_long = df[["iso3"] + edu_cols].melt(
            id_vars="iso3", value_vars=edu_cols,
            var_name="year_str", value_name="education_index"
        )
        df_long["year"] = df_long["year_str"].str.replace("mys_", "").astype(int)
        df_long = df_long[["iso3", "year", "education_index"]].dropna()

        # 2025 projection
        records = []
        for iso3, grp in df_long.groupby("iso3"):
            grp = grp.sort_values("year")
            val_18 = grp[grp["year"] == 2018]["education_index"].values
            val_22 = grp[grp["year"] == 2022]["education_index"].values
            if len(val_18) > 0 and len(val_22) > 0:
                slope = (val_22[0] - val_18[0]) / 4.0
                val_25 = np.clip(val_22[0] + slope * 3.0, 0.0, 1.0)
            elif len(val_22) > 0:
                val_25 = val_22[0]
            else:
                val_25 = grp["education_index"].iloc[-1]
            records.append({"iso3": iso3, "year": 2025, "education_index": round(float(val_25), 4)})

        df_long = pd.concat([df_long, pd.DataFrame(records)], ignore_index=True)
        df_long.to_csv(cache_path, index=False)
        print(f"  UNDP Education Index complete: {len(df_long):,} rows")
        return df_long
    except Exception as e:
        print(f"  UNDP fallback ({e}).")
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
        return pd.DataFrame(columns=["iso3", "year", "education_index"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. UCDP CONFLICT
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ucdp_conflict() -> pd.DataFrame:
    """Load unified UCDP conflict intensity dataset."""
    cache_path = os.path.join(CACHE_DIR, "ucdp_conflict.csv")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading ucdp_conflict.csv")
        df = pd.read_csv(cache_path)
        df["conflict_intensity"] = pd.to_numeric(df["conflict_intensity"], errors="coerce").fillna(0.0)
        return df

    return load_conflict_long()


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD FACTORS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def build_factors_panel(
    wb: pd.DataFrame,
    conflict: pd.DataFrame,
    edu: pd.DataFrame,
    henley: pd.DataFrame,
    ndgain: pd.DataFrame,
    religions: pd.DataFrame = None,
    cult_integration: pd.DataFrame = None,
    unhcr: pd.DataFrame = None,
    geography: pd.DataFrame = None,
) -> pd.DataFrame:
    """Merge all supplementary datasets including religion & cultural integration into one cohesive country-year panel for 1990-2025."""
    panel = wb[wb["year"].isin(SNAPSHOT_YEARS)].copy()

    for df, label in [
        (conflict, "conflict_intensity"),
        (edu, "education_index"),
        (henley, "visa_openness_index"),
        (ndgain, "climate_vulnerability"),
        (cult_integration, "cultural_integration"),
        (unhcr, "unhcr_displacement"),
    ]:
        if df is None or df.empty:
            continue
        df_snap = df[df["year"].isin(SNAPSHOT_YEARS)].copy()
        panel = panel.merge(df_snap, on=["iso3", "year"], how="left")

    # Merge static religion profile
    if religions is not None and not religions.empty:
        rel_cols = [
            "iso3", "dominant_religion", "christianity_pct", "islam_pct",
            "hinduism_pct", "buddhism_pct", "judaism_pct", "unaffiliated_pct",
            "religious_fractionalization", "religious_diversity_index", "religiosity_importance_score"
        ]
        present_rel_cols = [c for c in rel_cols if c in religions.columns]
        panel = panel.merge(religions[present_rel_cols], on="iso3", how="left")

    # Static geography (CEPII): landlocked flag and land area
    if geography is not None and not geography.empty:
        panel = panel.merge(geography[["iso3", "landlocked", "area_km2"]], on="iso3", how="left")

    # Displacement and tourism as rates, so they compare across country sizes
    pop = panel["population"].where(panel["population"] > 0)
    for col in ("refugees_abroad", "refugees_hosted", "idps"):
        if col in panel.columns:
            panel[f"{col}_per_1k"] = panel[col] / pop * 1000
    if "tourism_arrivals" in panel.columns:
        panel["tourism_per_capita"] = panel["tourism_arrivals"] / pop

    # Conflict intensity defaults to 0 for peaceful country-years
    if "conflict_intensity" in panel.columns:
        panel["conflict_intensity"] = panel["conflict_intensity"].fillna(0.0)
    else:
        panel["conflict_intensity"] = 0.0

    # Smooth forward/backward fills per country for bounded indices where country existed
    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    numeric_smooth_cols = [
        "education_index", "climate_vulnerability", "visa_openness_index",
        "unemployment", "gdp_per_capita", "integration_policy_index",
        "cultural_openness_index", "labour_market_integration",
        "anti_discrimination_score", "naturalization_access_score",
        "religious_fractionalization", "religious_diversity_index", "religiosity_importance_score"
    ]
    # World Bank series are already gap-filled within each snapshot window
    numeric_smooth_cols = [c for c in numeric_smooth_cols if c not in WB_INDICATORS]
    for col in numeric_smooth_cols:
        if col in panel.columns:
            panel[col] = panel.groupby("iso3")[col].transform(lambda s: s.ffill().bfill())

    panel = panel.drop_duplicates(subset=["iso3", "year"])
    panel = panel[panel["year"].isin(SNAPSHOT_YEARS)]
    
    # Standardize column order
    cols_order = [
        "iso3", "year", "gdp_per_capita", "population", "unemployment",
        "conflict_intensity", "education_index", "climate_vulnerability", "visa_openness_index",
        "dominant_religion", "religious_fractionalization", "religious_diversity_index", "religiosity_importance_score",
        "integration_policy_index", "cultural_openness_index", "labour_market_integration",
        "anti_discrimination_score", "naturalization_access_score",
    ] + [c for c in WB_INDICATORS if c not in ("gdp_per_capita", "population", "unemployment")] + [
        "tourism_per_capita", "refugees_abroad", "refugees_hosted", "idps",
        "refugees_abroad_per_1k", "refugees_hosted_per_1k", "idps_per_1k",
        "landlocked", "area_km2",
    ]
    cols_present = [c for c in cols_order if c in panel.columns] + [c for c in panel.columns if c not in cols_order]
    panel = panel[cols_present]

    print(f"  Factors panel: {panel.shape[0]:,} rows, {panel['iso3'].nunique()} countries, columns: {panel.columns.tolist()}")
    return panel.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 1: DATA COLLECTION & CLEANING")
    print("=" * 70)

    t_start = time.time()

    print("\n[1/8] Parsing UN DESA bilateral migration data...")
    df_wide = parse_undesa_fast(UNDESA_FILE)
    df_long = melt_to_long(df_wide)
    df_long = extrapolate_2025(df_long)

    out_mig = os.path.join(PROCESSED, "migration_long.csv")
    df_long.to_csv(out_mig, index=False)
    print(f"  ✓ migration_long.csv: {len(df_long):,} rows")

    print("\n[2/8] World Bank data...")
    wb = fetch_all_worldbank()

    print("\n[3/8] UCDP conflict data...")
    conflict = fetch_ucdp_conflict()

    print("\n[4/8] UNDP Education Index...")
    edu = fetch_undp_education()

    print("\n[5/8] Henley Passport + ND-GAIN climate...")
    henley = load_henley()
    ndgain = load_ndgain()

    print("\n[6/8] Loading World Religion Demographics & Cultural Integration Index...")
    religions = load_religions()
    cult_integration = load_cultural_integration()

    print("\n[7/8] UNHCR displacement + CEPII geography...")
    unhcr = fetch_unhcr()
    geography = load_geography_country()
    if geography.empty:
        print("  WARNING: data/raw/cepii_country.csv missing; run data/raw/build_geography.py")

    print("\n[8/8] Building unified multi-dimensional factors panel...")
    factors = build_factors_panel(wb, conflict, edu, henley, ndgain, religions, cult_integration,
                                  unhcr, geography)
    
    out_fac = os.path.join(PROCESSED, "factors_panel.csv")
    out_fac_enrich = os.path.join(PROCESSED, "factors_panel_enriched.csv")
    factors.to_csv(out_fac, index=False)
    factors.to_csv(out_fac_enrich, index=False)
    
    # Save enriched metadata
    meta = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows": len(factors),
        "countries": int(factors["iso3"].nunique()),
        "columns": factors.columns.tolist(),
        "years": sorted(factors["year"].unique().tolist()),
    }
    with open(os.path.join(PROCESSED, "factors_panel_enriched.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ factors_panel.csv: {len(factors):,} rows")
    print(f"  ✓ factors_panel_enriched.csv: {len(factors):,} rows")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  migration_long.csv : {len(df_long):,} rows, years={df_long['year'].unique().tolist()}")
    print(f"  factors_panel.csv  : {len(factors):,} rows, cols={factors.columns.tolist()}")


if __name__ == "__main__":
    main()
