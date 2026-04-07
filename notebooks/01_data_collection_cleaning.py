"""
Phase 1: Data Collection & Cleaning
====================================
Nodes and Nations: A Complex Network Study of Global Migration

Steps:
  1. Parse UN DESA bilateral migration stock Excel → long format (fast openpyxl path)
  2. Download/cache supplementary factor data (WB API, UNDP, UCDP)
  3. Merge all on (iso3, year)
  4. Log-linear extrapolation for 2025 snapshot
  5. Export migration_long.csv and factors_panel.csv

Performance notes:
  - Excel is parsed with openpyxl read_only mode (much faster than pd.read_excel default)
  - Downloaded API data is cached to data/raw/cache/*.csv; re-runs skip downloads
  - All API calls have a 20s timeout; failures produce NaN (analysis continues)

Outputs:  data/processed/migration_long.csv
          data/processed/factors_panel.csv
"""

import os
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
import openpyxl

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(ROOT, "data", "raw")
CACHE_DIR   = os.path.join(RAW_DIR, "cache")
PROCESSED   = os.path.join(ROOT, "data", "processed")
os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(PROCESSED,   exist_ok=True)

UNDESA_FILE    = os.path.join(RAW_DIR, "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx")
OFFICIAL_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
SNAPSHOT_YEARS = OFFICIAL_YEARS + [2025]

API_TIMEOUT = 20   # seconds per request
WB_INDICATORS = {
    "gdp_per_capita":  "NY.GDP.PCAP.CD",
    "population":      "SP.POP.TOTL",
    "unemployment":    "SL.UEM.TOTL.ZS",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def cached(cache_file: str):
    """Decorator: load cache CSV if it exists, else run func and save result."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            path = os.path.join(CACHE_DIR, cache_file)
            if os.path.exists(path):
                print(f"  [cache] Loading {cache_file}")
                return pd.read_csv(path)
            df = func(*args, **kwargs)
            if df is not None and not df.empty:
                df.to_csv(path, index=False)
                print(f"  [cache] Saved {cache_file} ({len(df):,} rows)")
            return df
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARSE UN DESA  (fast openpyxl read-only path)
# ══════════════════════════════════════════════════════════════════════════════

# Compact M49 → ISO3 mapping (UN numeric M49 codes)
M49_TO_ISO3 = {
    4:"AFG",8:"ALB",12:"DZA",16:"ASM",20:"AND",24:"AGO",28:"ATG",32:"ARG",
    36:"AUS",40:"AUT",31:"AZE",44:"BHS",48:"BHR",50:"BGD",52:"BRB",112:"BLR",
    56:"BEL",84:"BLZ",204:"BEN",64:"BTN",68:"BOL",70:"BIH",72:"BWA",76:"BRA",
    96:"BRN",100:"BGR",854:"BFA",108:"BDI",116:"KHM",120:"CMR",124:"CAN",132:"CPV",
    140:"CAF",148:"TCD",152:"CHL",156:"CHN",170:"COL",174:"COM",178:"COG",
    180:"COD",188:"CRI",191:"HRV",192:"CUB",196:"CYP",203:"CZE",208:"DNK",262:"DJI",
    212:"DMA",214:"DOM",218:"ECU",818:"EGY",222:"SLV",231:"ETH",242:"FJI",
    246:"FIN",250:"FRA",266:"GAB",270:"GMB",268:"GEO",276:"DEU",288:"GHA",300:"GRC",
    308:"GRD",320:"GTM",324:"GIN",328:"GUY",332:"HTI",340:"HND",348:"HUN",356:"IND",
    360:"IDN",364:"IRN",368:"IRQ",372:"IRL",376:"ISR",380:"ITA",388:"JAM",392:"JPN",
    400:"JOR",398:"KAZ",404:"KEN",296:"KIR",408:"PRK",410:"KOR",414:"KWT",417:"KGZ",
    418:"LAO",422:"LBN",426:"LSO",430:"LBR",434:"LBY",440:"LTU",442:"LUX",450:"MDG",
    454:"MWI",458:"MYS",462:"MDV",466:"MLI",470:"MLT",584:"MHL",478:"MRT",480:"MUS",
    484:"MEX",583:"FSM",496:"MNG",504:"MAR",508:"MOZ",516:"NAM",524:"NPL",528:"NLD",
    554:"NZL",558:"NIC",562:"NER",566:"NGA",578:"NOR",512:"OMN",586:"PAK",585:"PLW",
    591:"PAN",598:"PNG",600:"PRY",604:"PER",608:"PHL",616:"POL",620:"PRT",634:"QAT",
    642:"ROU",643:"RUS",646:"RWA",682:"SAU",686:"SEN",694:"SLE",706:"SOM",710:"ZAF",
    724:"ESP",736:"SDN",740:"SUR",748:"SWZ",752:"SWE",756:"CHE",760:"SYR",
    762:"TJK",764:"THA",626:"TLS",768:"TGO",776:"TON",780:"TTO",788:"TUN",792:"TUR",
    795:"TKM",800:"UGA",804:"UKR",784:"ARE",826:"GBR",840:"USA",858:"URY",860:"UZB",
    548:"VUT",704:"VNM",887:"YEM",894:"ZMB",716:"ZWE",659:"KNA",662:"LCA",
    670:"VCT",882:"WSM",678:"STP",688:"SRB",499:"MNE",807:"MKD",428:"LVA",233:"EST",
    498:"MDA",51:"ARM",275:"PSE",703:"SVK",705:"SVN",232:"ERI",834:"TZA",
    226:"GNQ",90:"SLB",144:"LKA",
}


def parse_undesa_fast(filepath: str) -> pd.DataFrame:
    """
    Fast parsing of UN DESA IMS Excel using openpyxl read_only mode.
    Table 1 contains total (both-sex) migrant stocks.
    Structure (after row 10 header):
      Col0=Index, Col1=Dest name, Col2=Coverage, Col3=Data type,
      Col4=Dest loccode, Col5=Origin name, Col6=Origin loccode,
      Col7..13 = 1990,1995,2000,2005,2010,2015,2020,  Col14=2024, ...
    """
    cache_path = os.path.join(CACHE_DIR, "undesa_parsed.csv")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading undesa_parsed.csv")
        return pd.read_csv(cache_path)

    print(f"  Parsing UN DESA Excel (openpyxl read_only)...")
    t0 = time.time()

    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb["Table 1"]

    rows_iter = ws.iter_rows(values_only=True)

    # Skip rows 0..9 (metadata), row 10 = header
    for _ in range(10):
        next(rows_iter)
    header = next(rows_iter)   # row index 10

    # Identify year column positions from header
    # Header looks like: Index, dest_name, Coverage, DataType, dest_loccode,
    #                    origin_name, origin_loccode, 1990, 1995, ..., 2020, 2024, ...
    year_col_indices = {}
    for col_idx, val in enumerate(header):
        if isinstance(val, int) and val in OFFICIAL_YEARS:
            year_col_indices[val] = col_idx

    print(f"  Year columns found: {year_col_indices}")

    NAME_DEST = 1   # column index of destination name
    LOCD_DEST = 4   # column index of destination location code
    NAME_ORIG = 5   # column index of origin name
    LOCD_ORIG = 6   # column index of origin location code

    records = []
    for row in rows_iter:
        if len(row) <= LOCD_ORIG:
            continue
        dest_loc  = row[LOCD_DEST]
        orig_loc  = row[LOCD_ORIG]

        # Keep country-level rows only (M49 code < 900)
        try:
            dest_loc = int(dest_loc)
            orig_loc = int(orig_loc)
        except (TypeError, ValueError):
            continue
        if dest_loc >= 900 or orig_loc >= 900:
            continue
        if dest_loc == orig_loc:
            continue

        dest_iso3  = M49_TO_ISO3.get(dest_loc)
        origin_iso3 = M49_TO_ISO3.get(orig_loc)
        if not dest_iso3 or not origin_iso3:
            continue

        rec = {
            "dest_name":    row[NAME_DEST],
            "origin_name":  row[NAME_ORIG],
            "dest_iso3":    dest_iso3,
            "origin_iso3":  origin_iso3,
        }
        for yr, ci in year_col_indices.items():
            val = row[ci] if ci < len(row) else None
            rec[yr] = float(val) if isinstance(val, (int, float)) and val is not None else np.nan
        records.append(rec)

    wb.close()
    print(f"  Parsed {len(records):,} country-pair rows in {time.time()-t0:.1f}s")

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    print(f"  [cache] Saved undesa_parsed.csv")
    return df


def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot bilateral stocks from wide (year columns) to long format."""
    year_cols = [c for c in df.columns if isinstance(c, int) and c in OFFICIAL_YEARS]
    # Convert str column names (from CSV reload) to int
    str_year_cols = [c for c in df.columns if str(c) in [str(y) for y in OFFICIAL_YEARS]]
    year_cols = year_cols or [int(c) for c in str_year_cols]

    id_cols = ["dest_iso3", "origin_iso3"]
    df_wide = df[id_cols + [str(y) if str(y) in df.columns else y for y in OFFICIAL_YEARS]].copy()
    # Ensure year columns are named as ints
    col_rename = {str(y): y for y in OFFICIAL_YEARS if str(y) in df_wide.columns}
    df_wide = df_wide.rename(columns=col_rename)
    year_cols_int = [y for y in OFFICIAL_YEARS if y in df_wide.columns]

    df_long = df_wide.melt(
        id_vars=id_cols,
        value_vars=year_cols_int,
        var_name="year",
        value_name="migrant_stock",
    )
    df_long["year"] = df_long["year"].astype(int)
    df_long = df_long.dropna(subset=["migrant_stock"])
    df_long = df_long[df_long["migrant_stock"] > 0]
    df_long = df_long.drop_duplicates(subset=["dest_iso3", "origin_iso3", "year"])
    print(f"  Long-format rows (non-zero flows): {len(df_long):,}")
    return df_long.reset_index(drop=True)


def extrapolate_2025(df_long: pd.DataFrame) -> pd.DataFrame:
    """Log-linear (geometric) extrapolation of 2025 from 2015→2020 trend."""
    print("  Extrapolating 2025 values...")
    s2015 = df_long[df_long["year"] == 2015].set_index(["dest_iso3", "origin_iso3"])["migrant_stock"]
    s2020 = df_long[df_long["year"] == 2020].set_index(["dest_iso3", "origin_iso3"])["migrant_stock"]

    common = s2015.index.intersection(s2020.index)
    v2015  = s2015.loc[common].values
    v2020  = s2020.loc[common].values

    # geometric step: s2025 = s2020 * (s2020 / s2015)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio  = np.where(v2015 > 0, v2020 / v2015, 1.0)
        stock25 = np.where(np.isfinite(ratio) & (ratio > 0), v2020 * ratio, v2020)
    stock25 = np.maximum(stock25, 1.0)

    df_2025 = pd.DataFrame({
        "dest_iso3":      [idx[0] for idx in common],
        "origin_iso3":    [idx[1] for idx in common],
        "year":           2025,
        "migrant_stock":  stock25,
        "is_extrapolated": True,
    })
    df_long["is_extrapolated"] = False
    result = pd.concat([df_long, df_2025], ignore_index=True)
    print(f"  Total rows after extrapolation: {len(result):,}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. WORLD BANK DATA  (cached)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_wb_indicator(indicator_code: str, indicator_name: str) -> pd.DataFrame:
    """Fetch one World Bank indicator (all countries, 1985–2025)."""
    # Page through API (up to 3 pages of 5000 each is enough for all countries × 40 years)
    records = []
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
            f"?format=json&date=1985:2025&per_page=5000&page={page}"
        )
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    WB API error (page {page}): {e}")
            break

        if len(data) < 2 or not data[1]:
            break
        for item in data[1]:
            if item.get("value") is not None and item.get("countryiso3code"):
                records.append({
                    "iso3":          item["countryiso3code"],
                    "year":          int(item["date"]),
                    indicator_name:  float(item["value"]),
                })
        # Check if more pages
        meta = data[0]
        if page >= meta.get("pages", 1):
            break
        page += 1

    df = pd.DataFrame(records)
    print(f"    {indicator_name}: {len(df):,} rows")
    return df


def fetch_all_worldbank() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "worldbank_factors.csv")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading worldbank_factors.csv")
        return pd.read_csv(cache_path)

    print("  Downloading World Bank indicators (gdp, population, unemployment)...")
    dfs = [fetch_wb_indicator(code, name) for name, code in WB_INDICATORS.items()]
    panel = dfs[0]
    for df in dfs[1:]:
        panel = panel.merge(df, on=["iso3", "year"], how="outer")
    panel.to_csv(cache_path, index=False)
    print(f"  [cache] Saved worldbank_factors.csv ({len(panel):,} rows)")
    return panel


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNDP EDUCATION INDEX  (cached)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_undp_education() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "undp_education.csv")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading undp_education.csv")
        return pd.read_csv(cache_path)

    print("  Downloading UNDP Education Index (bulk CSV)...")
    try:
        url = ("https://hdr.undp.org/sites/default/files/2023-24_HDR/"
               "HDR23-24_Composite_indices_complete_time_series.csv")
        df = pd.read_csv(url, encoding="latin-1")
        edu_cols = [c for c in df.columns if str(c).startswith("edi_")]
        if not edu_cols:
            raise ValueError("No edi_* columns")
        df_long = df[["iso3"] + edu_cols].melt(
            id_vars="iso3", value_vars=edu_cols,
            var_name="year_str", value_name="education_index"
        )
        df_long["year"] = df_long["year_str"].str.replace("edi_", "").astype(int)
        df_long = df_long[["iso3", "year", "education_index"]].dropna()
        df_long.to_csv(cache_path, index=False)
        print(f"  UNDP Education Index: {len(df_long):,} rows")
        return df_long
    except Exception as e:
        print(f"  UNDP download failed ({e}). education_index will be NaN.")
        return pd.DataFrame(columns=["iso3", "year", "education_index"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. UCDP CONFLICT  (cached)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ucdp_conflict() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "ucdp_conflict.csv")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading ucdp_conflict.csv")
        return pd.read_csv(cache_path)

    print("  Downloading UCDP PRIO Armed Conflict Dataset (public CSV)...")
    # GW-code → ISO3 mapping
    gw_to_iso3 = {
        2:"USA",20:"CAN",40:"CUB",41:"HTI",42:"DOM",51:"JAM",52:"TTO",
        70:"MEX",90:"GTM",92:"HND",93:"SLV",94:"NIC",95:"CRI",
        100:"COL",101:"VEN",110:"GUY",115:"SUR",130:"ECU",135:"PER",
        140:"BRA",145:"BOL",150:"PRY",155:"CHL",160:"ARG",165:"URY",
        200:"GBR",205:"IRL",210:"NLD",211:"BEL",212:"LUX",220:"FRA",
        225:"CHE",230:"ESP",235:"PRT",245:"DNK",255:"DEU",290:"POL",
        305:"AUT",310:"HUN",315:"CZE",317:"SVK",325:"ITA",339:"ALB",
        340:"SRB",341:"MNE",343:"MKD",344:"HRV",346:"BIH",350:"GRC",
        352:"CYP",355:"BGR",360:"ROU",365:"RUS",366:"EST",367:"LVA",
        368:"LTU",369:"UKR",370:"BLR",371:"MDA",372:"ARM",373:"AZE",
        374:"GEO",375:"FIN",380:"SWE",385:"NOR",432:"MLI",433:"SEN",
        434:"GMB",436:"GIN",437:"SLE",438:"LBR",439:"CIV",450:"GHA",
        451:"TGO",452:"BEN",461:"NGA",471:"CMR",481:"CAF",482:"COG",
        483:"COD",490:"GAB",500:"UGA",501:"KEN",510:"TZA",516:"AGO",
        517:"MOZ",520:"ZMB",522:"ZWE",540:"MDG",553:"MWI",560:"ZAF",
        565:"NAM",570:"LSO",571:"BWA",600:"MAR",616:"TUN",620:"DZA",
        625:"LBY",630:"IRN",640:"TUR",645:"IRQ",651:"EGY",652:"SYR",
        660:"LBN",663:"JOR",666:"ISR",667:"PSE",670:"SAU",678:"YEM",
        680:"KWT",690:"QAT",694:"ARE",696:"OMN",700:"AFG",701:"TJK",
        702:"UZB",703:"KGZ",704:"TKM",705:"KAZ",710:"CHN",713:"PRK",
        732:"KOR",740:"JPN",750:"IND",760:"PAK",770:"BGD",771:"LKA",
        790:"MMR",800:"THA",811:"KHM",812:"LAO",816:"VNM",840:"PHL",
        850:"IDN",900:"AUS",920:"PNG",940:"NZL",
    }
    try:
        # UCDP/PRIO Armed Conflict Dataset v24.1 — publicly available CSV
        url = "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-241.csv"
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        # Relevant columns: gwno_a (GW code of side A), year, intensity_level
        df = df[["gwno_a", "year", "intensity_level"]].copy()
        df = df.rename(columns={"gwno_a": "gwno", "intensity_level": "conflict_intensity"})
        df["gwno"] = pd.to_numeric(df["gwno"], errors="coerce")
        df["iso3"] = df["gwno"].map(gw_to_iso3)
        df = df.dropna(subset=["iso3"])
        df = df.groupby(["iso3", "year"])["conflict_intensity"].max().reset_index()
        df.to_csv(cache_path, index=False)
        print(f"  UCDP conflict: {len(df):,} country-year records")
        return df
    except Exception as e:
        print(f"  UCDP download failed ({e}). conflict_intensity will default to 0.")
        return pd.DataFrame(columns=["iso3", "year", "conflict_intensity"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. MANUAL STUBS  (Henley + ND-GAIN)
# ══════════════════════════════════════════════════════════════════════════════

def load_or_skip(filename: str, required_cols: list, label: str) -> pd.DataFrame:
    """Load a manually placed CSV from raw/; return empty DF if not found."""
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: {filename} missing columns {missing}. Skipping.")
            return pd.DataFrame(columns=required_cols)
        print(f"  Loaded {label}: {len(df):,} rows")
        return df[required_cols]
    else:
        print(f"  {label}: NOT FOUND — place '{filename}' in data/raw/ to include this variable")
        return pd.DataFrame(columns=required_cols)


def load_ndgain_climate() -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "ndgain_climate.csv")
    if os.path.exists(cache_path):
        print(f"  [cache] Loading ndgain_climate.csv")
        return pd.read_csv(cache_path)
    # Try manual file first
    df = load_or_skip("ndgain_country_index.csv", ["iso3", "year", "climate_vulnerability"],
                      "ND-GAIN Climate Index")
    if not df.empty:
        df.to_csv(cache_path, index=False)
        return df
    # Try public download
    try:
        print("  Trying ND-GAIN public download...")
        url = "https://gain.nd.edu/assets/521076/nd_gain_country_index_since_1995.csv"
        raw = pd.read_csv(url)
        yr_cols = [c for c in raw.columns if str(c).isdigit() and 1990 <= int(c) <= 2024]
        iso_col = "ISO3" if "ISO3" in raw.columns else (
                  "iso3" if "iso3" in raw.columns else raw.columns[1])
        raw = raw.rename(columns={iso_col: "iso3"})
        df_long = raw[["iso3"] + yr_cols].melt(
            id_vars="iso3", value_vars=yr_cols,
            var_name="year", value_name="climate_vulnerability"
        )
        df_long["year"] = df_long["year"].astype(int)
        df_long = df_long.dropna(subset=["climate_vulnerability"])
        df_long.to_csv(cache_path, index=False)
        print(f"  ND-GAIN downloaded: {len(df_long):,} rows")
        return df_long[["iso3", "year", "climate_vulnerability"]]
    except Exception as e:
        print(f"  ND-GAIN download failed ({e}). climate_vulnerability will be NaN.")
        return pd.DataFrame(columns=["iso3", "year", "climate_vulnerability"])


# ══════════════════════════════════════════════════════════════════════════════
# 6. BUILD FACTORS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def build_factors_panel(wb, conflict, edu, henley, ndgain) -> pd.DataFrame:
    """Merge all supplementary datasets into one country-year panel."""
    # Start with WB (broadest coverage)
    panel = wb[wb["year"].isin(SNAPSHOT_YEARS)].copy()

    for df, label in [(conflict, "conflict"), (edu, "education"),
                      (henley, "henley"), (ndgain, "ndgain")]:
        if df.empty:
            continue
        df_snap = df[df["year"].isin(SNAPSHOT_YEARS)].copy()
        panel = panel.merge(df_snap, on=["iso3", "year"], how="outer")

    if "conflict_intensity" in panel.columns:
        panel["conflict_intensity"] = panel["conflict_intensity"].fillna(0)
    panel = panel.drop_duplicates(subset=["iso3", "year"])
    panel = panel[panel["year"].isin(SNAPSHOT_YEARS)]
    print(f"  Factors panel: {panel.shape[0]:,} rows, columns: {panel.columns.tolist()}")
    return panel.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 1: DATA COLLECTION & CLEANING")
    print("=" * 70)

    t_start = time.time()

    print("\n[1/6] Parsing UN DESA bilateral migration data...")
    df_wide = parse_undesa_fast(UNDESA_FILE)
    df_long = melt_to_long(df_wide)
    df_long = extrapolate_2025(df_long)

    out_mig = os.path.join(PROCESSED, "migration_long.csv")
    df_long.to_csv(out_mig, index=False)
    print(f"  ✓ migration_long.csv: {len(df_long):,} rows")

    print("\n[2/6] World Bank data...")
    wb = fetch_all_worldbank()

    print("\n[3/6] UCDP conflict data...")
    conflict = fetch_ucdp_conflict()

    print("\n[4/6] UNDP Education Index...")
    edu = fetch_undp_education()

    print("\n[5/6] Henley Passport + ND-GAIN climate...")
    henley = load_or_skip("henley_passport_index.csv",
                          ["iso3", "year", "visa_openness_index"], "Henley Passport Index")
    ndgain = load_ndgain_climate()

    print("\n[6/6] Building factors panel...")
    factors = build_factors_panel(wb, conflict, edu, henley, ndgain)
    out_fac = os.path.join(PROCESSED, "factors_panel.csv")
    factors.to_csv(out_fac, index=False)
    print(f"  ✓ factors_panel.csv: {len(factors):,} rows")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE  ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  migration_long.csv : {len(df_long):,} rows, years={df_long['year'].unique().tolist()}")
    print(f"  factors_panel.csv  : {len(factors):,} rows, cols={factors.columns.tolist()}")

if __name__ == "__main__":
    main()
