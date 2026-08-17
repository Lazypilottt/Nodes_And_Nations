"""data.loader

Standardized data loading and enrichment utilities for Nodes and Nations.
Exposes paths and unified loaders for raw, processed, and export datasets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "cache"
PROCESSED_DIR = ROOT / "data" / "processed"
EXPORTS_DIR = ROOT / "data" / "exports"
MODELS_DIR = ROOT / "models"

paths: Dict[str, str] = {
    "ROOT": str(ROOT),
    "RAW_DIR": str(RAW_DIR),
    "CACHE_DIR": str(CACHE_DIR),
    "PROCESSED_DIR": str(PROCESSED_DIR),
    "EXPORTS_DIR": str(EXPORTS_DIR),
    "MODELS_DIR": str(MODELS_DIR),
}

# ── Static Reference Metadata for 235 Countries / Territories ────────────────
COUNTRY_METADATA_TABLE: Dict[str, tuple[str, str, str, str]] = {
    "ABW": ("Aruba", "Americas", "Caribbean", "High income"),
    "AFG": ("Afghanistan", "Asia", "South Asia", "Low income"),
    "AGO": ("Angola", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "AIA": ("Anguilla", "Americas", "Caribbean", "High income"),
    "ALB": ("Albania", "Europe", "Eastern Europe", "Upper middle income"),
    "AND": ("Andorra", "Europe", "Western Europe", "High income"),
    "ARE": ("United Arab Emirates", "Asia", "Western Asia", "High income"),
    "ARG": ("Argentina", "Americas", "Latin America", "Upper middle income"),
    "ARM": ("Armenia", "Asia", "Western Asia", "Upper middle income"),
    "ASM": ("American Samoa", "Oceania", "Polynesia", "Upper middle income"),
    "ATG": ("Antigua and Barbuda", "Americas", "Caribbean", "High income"),
    "AUS": ("Australia", "Oceania", "Australia/NZ", "High income"),
    "AUT": ("Austria", "Europe", "Western Europe", "High income"),
    "AZE": ("Azerbaijan", "Asia", "Western Asia", "Upper middle income"),
    "BDI": ("Burundi", "Africa", "Sub-Saharan Africa", "Low income"),
    "BEL": ("Belgium", "Europe", "Western Europe", "High income"),
    "BEN": ("Benin", "Africa", "Sub-Saharan Africa", "Low income"),
    "BES": ("Bonaire, Sint Eustatius and Saba", "Americas", "Caribbean", "High income"),
    "BFA": ("Burkina Faso", "Africa", "Sub-Saharan Africa", "Low income"),
    "BGD": ("Bangladesh", "Asia", "South Asia", "Lower middle income"),
    "BGR": ("Bulgaria", "Europe", "Eastern Europe", "Upper middle income"),
    "BHR": ("Bahrain", "Asia", "Western Asia", "High income"),
    "BHS": ("Bahamas", "Americas", "Caribbean", "High income"),
    "BIH": ("Bosnia and Herzegovina", "Europe", "Eastern Europe", "Upper middle income"),
    "BLM": ("Saint Barthélemy", "Americas", "Caribbean", "High income"),
    "BLR": ("Belarus", "Europe", "Eastern Europe", "Upper middle income"),
    "BLZ": ("Belize", "Americas", "Central America", "Lower middle income"),
    "BMU": ("Bermuda", "Americas", "Northern America", "High income"),
    "BOL": ("Bolivia", "Americas", "Latin America", "Lower middle income"),
    "BRA": ("Brazil", "Americas", "Latin America", "Upper middle income"),
    "BRB": ("Barbados", "Americas", "Caribbean", "High income"),
    "BRN": ("Brunei Darussalam", "Asia", "South-Eastern Asia", "High income"),
    "BTN": ("Bhutan", "Asia", "South Asia", "Lower middle income"),
    "BWA": ("Botswana", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "CAF": ("Central African Republic", "Africa", "Sub-Saharan Africa", "Low income"),
    "CAN": ("Canada", "Americas", "Northern America", "High income"),
    "CHE": ("Switzerland", "Europe", "Western Europe", "High income"),
    "CHI": ("Channel Islands", "Europe", "Northern Europe", "High income"),
    "CHL": ("Chile", "Americas", "Latin America", "High income"),
    "CHN": ("China", "Asia", "Eastern Asia", "Upper middle income"),
    "CIV": ("Côte d'Ivoire", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "CMR": ("Cameroon", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "COD": ("DR Congo", "Africa", "Sub-Saharan Africa", "Low income"),
    "COG": ("Congo", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "COK": ("Cook Islands", "Oceania", "Polynesia", "High income"),
    "COL": ("Colombia", "Americas", "Latin America", "Upper middle income"),
    "COM": ("Comoros", "Africa", "Sub-Saharan Africa", "Low income"),
    "CPV": ("Cabo Verde", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "CRI": ("Costa Rica", "Americas", "Latin America", "Upper middle income"),
    "CUB": ("Cuba", "Americas", "Latin America", "Upper middle income"),
    "CUW": ("Curaçao", "Americas", "Caribbean", "High income"),
    "CYM": ("Cayman Islands", "Americas", "Caribbean", "High income"),
    "CYP": ("Cyprus", "Europe", "Western Asia", "High income"),
    "CZE": ("Czechia", "Europe", "Eastern Europe", "High income"),
    "DEU": ("Germany", "Europe", "Western Europe", "High income"),
    "DJI": ("Djibouti", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "DMA": ("Dominica", "Americas", "Caribbean", "Upper middle income"),
    "DNK": ("Denmark", "Europe", "Northern Europe", "High income"),
    "DOM": ("Dominican Republic", "Americas", "Latin America", "Upper middle income"),
    "DZA": ("Algeria", "Africa", "Northern Africa", "Lower middle income"),
    "ECU": ("Ecuador", "Americas", "Latin America", "Upper middle income"),
    "EGY": ("Egypt", "Africa", "Northern Africa", "Lower middle income"),
    "ERI": ("Eritrea", "Africa", "Sub-Saharan Africa", "Low income"),
    "ESH": ("Western Sahara", "Africa", "Northern Africa", "Lower middle income"),
    "ESP": ("Spain", "Europe", "Southern Europe", "High income"),
    "EST": ("Estonia", "Europe", "Northern Europe", "High income"),
    "ETH": ("Ethiopia", "Africa", "Sub-Saharan Africa", "Low income"),
    "FIN": ("Finland", "Europe", "Northern Europe", "High income"),
    "FJI": ("Fiji", "Oceania", "Melanesia", "Upper middle income"),
    "FLK": ("Falkland Islands", "Americas", "South America", "High income"),
    "FRA": ("France", "Europe", "Western Europe", "High income"),
    "FRO": ("Faroe Islands", "Europe", "Northern Europe", "High income"),
    "FSM": ("Micronesia", "Oceania", "Micronesia", "Lower middle income"),
    "GAB": ("Gabon", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "GBR": ("United Kingdom", "Europe", "Northern Europe", "High income"),
    "GEO": ("Georgia", "Asia", "Western Asia", "Upper middle income"),
    "GHA": ("Ghana", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "GIB": ("Gibraltar", "Europe", "Southern Europe", "High income"),
    "GIN": ("Guinea", "Africa", "Sub-Saharan Africa", "Low income"),
    "GLP": ("Guadeloupe", "Americas", "Caribbean", "High income"),
    "GMB": ("Gambia", "Africa", "Sub-Saharan Africa", "Low income"),
    "GNB": ("Guinea-Bissau", "Africa", "Sub-Saharan Africa", "Low income"),
    "GNQ": ("Equatorial Guinea", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "GRC": ("Greece", "Europe", "Southern Europe", "High income"),
    "GRD": ("Grenada", "Americas", "Caribbean", "Upper middle income"),
    "GRL": ("Greenland", "Americas", "Northern America", "High income"),
    "GTM": ("Guatemala", "Americas", "Latin America", "Upper middle income"),
    "GUF": ("French Guiana", "Americas", "South America", "High income"),
    "GUM": ("Guam", "Oceania", "Micronesia", "High income"),
    "GUY": ("Guyana", "Americas", "Latin America", "Upper middle income"),
    "HKG": ("Hong Kong", "Asia", "Eastern Asia", "High income"),
    "HND": ("Honduras", "Americas", "Latin America", "Lower middle income"),
    "HRV": ("Croatia", "Europe", "Eastern Europe", "High income"),
    "HTI": ("Haiti", "Americas", "Latin America", "Low income"),
    "HUN": ("Hungary", "Europe", "Eastern Europe", "High income"),
    "IDN": ("Indonesia", "Asia", "South-Eastern Asia", "Upper middle income"),
    "IMN": ("Isle of Man", "Europe", "Northern Europe", "High income"),
    "IND": ("India", "Asia", "South Asia", "Lower middle income"),
    "IRL": ("Ireland", "Europe", "Northern Europe", "High income"),
    "IRN": ("Iran", "Asia", "Western Asia", "Lower middle income"),
    "IRQ": ("Iraq", "Asia", "Western Asia", "Upper middle income"),
    "ISL": ("Iceland", "Europe", "Northern Europe", "High income"),
    "ISR": ("Israel", "Asia", "Western Asia", "High income"),
    "ITA": ("Italy", "Europe", "Southern Europe", "High income"),
    "JAM": ("Jamaica", "Americas", "Latin America", "Upper middle income"),
    "JOR": ("Jordan", "Asia", "Western Asia", "Upper middle income"),
    "JPN": ("Japan", "Asia", "Eastern Asia", "High income"),
    "KAZ": ("Kazakhstan", "Asia", "Central Asia", "Upper middle income"),
    "KEN": ("Kenya", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "KGZ": ("Kyrgyzstan", "Asia", "Central Asia", "Lower middle income"),
    "KHM": ("Cambodia", "Asia", "South-Eastern Asia", "Lower middle income"),
    "KIR": ("Kiribati", "Oceania", "Micronesia", "Lower middle income"),
    "KNA": ("Saint Kitts and Nevis", "Americas", "Latin America", "High income"),
    "KOR": ("South Korea", "Asia", "Eastern Asia", "High income"),
    "KWT": ("Kuwait", "Asia", "Western Asia", "High income"),
    "LAO": ("Laos", "Asia", "South-Eastern Asia", "Lower middle income"),
    "LBN": ("Lebanon", "Asia", "Western Asia", "Lower middle income"),
    "LBR": ("Liberia", "Africa", "Sub-Saharan Africa", "Low income"),
    "LBY": ("Libya", "Africa", "Northern Africa", "Upper middle income"),
    "LCA": ("Saint Lucia", "Americas", "Latin America", "Upper middle income"),
    "LIE": ("Liechtenstein", "Europe", "Western Europe", "High income"),
    "LKA": ("Sri Lanka", "Asia", "South Asia", "Lower middle income"),
    "LSO": ("Lesotho", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "LTU": ("Lithuania", "Europe", "Northern Europe", "High income"),
    "LUX": ("Luxembourg", "Europe", "Western Europe", "High income"),
    "LVA": ("Latvia", "Europe", "Northern Europe", "High income"),
    "MAC": ("Macao", "Asia", "Eastern Asia", "High income"),
    "MAF": ("Saint Martin", "Americas", "Caribbean", "High income"),
    "MAR": ("Morocco", "Africa", "Northern Africa", "Lower middle income"),
    "MCO": ("Monaco", "Europe", "Western Europe", "High income"),
    "MDA": ("Moldova", "Europe", "Eastern Europe", "Lower middle income"),
    "MDG": ("Madagascar", "Africa", "Sub-Saharan Africa", "Low income"),
    "MDV": ("Maldives", "Asia", "South Asia", "Upper middle income"),
    "MEX": ("Mexico", "Americas", "Latin America", "Upper middle income"),
    "MHL": ("Marshall Islands", "Oceania", "Micronesia", "Upper middle income"),
    "MKD": ("North Macedonia", "Europe", "Eastern Europe", "Upper middle income"),
    "MLI": ("Mali", "Africa", "Sub-Saharan Africa", "Low income"),
    "MLT": ("Malta", "Europe", "Southern Europe", "High income"),
    "MMR": ("Myanmar", "Asia", "South-Eastern Asia", "Lower middle income"),
    "MNE": ("Montenegro", "Europe", "Eastern Europe", "Upper middle income"),
    "MNG": ("Mongolia", "Asia", "Eastern Asia", "Lower middle income"),
    "MNP": ("Northern Mariana Islands", "Oceania", "Micronesia", "High income"),
    "MOZ": ("Mozambique", "Africa", "Sub-Saharan Africa", "Low income"),
    "MRT": ("Mauritania", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "MSR": ("Montserrat", "Americas", "Caribbean", "Upper middle income"),
    "MTQ": ("Martinique", "Americas", "Caribbean", "High income"),
    "MUS": ("Mauritius", "Africa", "Sub-Saharan Africa", "High income"),
    "MWI": ("Malawi", "Africa", "Sub-Saharan Africa", "Low income"),
    "MYS": ("Malaysia", "Asia", "South-Eastern Asia", "Upper middle income"),
    "MYT": ("Mayotte", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "NAM": ("Namibia", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "NCL": ("New Caledonia", "Oceania", "Melanesia", "High income"),
    "NER": ("Niger", "Africa", "Sub-Saharan Africa", "Low income"),
    "NGA": ("Nigeria", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "NIC": ("Nicaragua", "Americas", "Latin America", "Lower middle income"),
    "NIU": ("Niue", "Oceania", "Polynesia", "Upper middle income"),
    "NLD": ("Netherlands", "Europe", "Western Europe", "High income"),
    "NOR": ("Norway", "Europe", "Northern Europe", "High income"),
    "NPL": ("Nepal", "Asia", "South Asia", "Lower middle income"),
    "NRU": ("Nauru", "Oceania", "Micronesia", "High income"),
    "NZL": ("New Zealand", "Oceania", "Australia/NZ", "High income"),
    "OMN": ("Oman", "Asia", "Western Asia", "High income"),
    "PAK": ("Pakistan", "Asia", "South Asia", "Lower middle income"),
    "PAN": ("Panama", "Americas", "Latin America", "High income"),
    "PER": ("Peru", "Americas", "Latin America", "Upper middle income"),
    "PHL": ("Philippines", "Asia", "South-Eastern Asia", "Lower middle income"),
    "PLW": ("Palau", "Oceania", "Micronesia", "High income"),
    "PNG": ("Papua New Guinea", "Oceania", "Melanesia", "Lower middle income"),
    "POL": ("Poland", "Europe", "Eastern Europe", "High income"),
    "PRI": ("Puerto Rico", "Americas", "Latin America", "High income"),
    "PRK": ("North Korea", "Asia", "Eastern Asia", "Low income"),
    "PRT": ("Portugal", "Europe", "Southern Europe", "High income"),
    "PRY": ("Paraguay", "Americas", "Latin America", "Upper middle income"),
    "PSE": ("Palestine", "Asia", "Western Asia", "Lower middle income"),
    "PYF": ("French Polynesia", "Oceania", "Polynesia", "High income"),
    "QAT": ("Qatar", "Asia", "Western Asia", "High income"),
    "REU": ("Réunion", "Africa", "Sub-Saharan Africa", "High income"),
    "ROU": ("Romania", "Europe", "Eastern Europe", "High income"),
    "RUS": ("Russia", "Europe", "Eastern Europe", "Upper middle income"),
    "RWA": ("Rwanda", "Africa", "Sub-Saharan Africa", "Low income"),
    "SAU": ("Saudi Arabia", "Asia", "Western Asia", "High income"),
    "SDN": ("Sudan", "Africa", "Sub-Saharan Africa", "Low income"),
    "SEN": ("Senegal", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "SGP": ("Singapore", "Asia", "South-Eastern Asia", "High income"),
    "SHN": ("Saint Helena", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "SLB": ("Solomon Islands", "Oceania", "Melanesia", "Lower middle income"),
    "SLE": ("Sierra Leone", "Africa", "Sub-Saharan Africa", "Low income"),
    "SLV": ("El Salvador", "Americas", "Latin America", "Lower middle income"),
    "SMR": ("San Marino", "Europe", "Southern Europe", "High income"),
    "SOM": ("Somalia", "Africa", "Sub-Saharan Africa", "Low income"),
    "SPM": ("Saint Pierre and Miquelon", "Americas", "Northern America", "High income"),
    "SRB": ("Serbia", "Europe", "Eastern Europe", "Upper middle income"),
    "SSD": ("South Sudan", "Africa", "Sub-Saharan Africa", "Low income"),
    "STP": ("Sao Tome and Principe", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "SUR": ("Suriname", "Americas", "Latin America", "Upper middle income"),
    "SVK": ("Slovakia", "Europe", "Eastern Europe", "High income"),
    "SVN": ("Slovenia", "Europe", "Eastern Europe", "High income"),
    "SWE": ("Sweden", "Europe", "Northern Europe", "High income"),
    "SWZ": ("Eswatini", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "SXM": ("Sint Maarten", "Americas", "Caribbean", "High income"),
    "SYC": ("Seychelles", "Africa", "Sub-Saharan Africa", "High income"),
    "SYR": ("Syria", "Asia", "Western Asia", "Low income"),
    "TCA": ("Turks and Caicos Islands", "Americas", "Caribbean", "High income"),
    "TCD": ("Chad", "Africa", "Sub-Saharan Africa", "Low income"),
    "TGO": ("Togo", "Africa", "Sub-Saharan Africa", "Low income"),
    "THA": ("Thailand", "Asia", "South-Eastern Asia", "Upper middle income"),
    "TJK": ("Tajikistan", "Asia", "Central Asia", "Low income"),
    "TKL": ("Tokelau", "Oceania", "Polynesia", "Lower middle income"),
    "TKM": ("Turkmenistan", "Asia", "Central Asia", "Upper middle income"),
    "TLS": ("Timor-Leste", "Asia", "South-Eastern Asia", "Lower middle income"),
    "TON": ("Tonga", "Oceania", "Polynesia", "Upper middle income"),
    "TTO": ("Trinidad and Tobago", "Americas", "Latin America", "High income"),
    "TUN": ("Tunisia", "Africa", "Northern Africa", "Lower middle income"),
    "TUR": ("Turkey", "Asia", "Western Asia", "Upper middle income"),
    "TUV": ("Tuvalu", "Oceania", "Polynesia", "Upper middle income"),
    "TWN": ("Taiwan", "Asia", "Eastern Asia", "High income"),
    "TZA": ("Tanzania", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "UGA": ("Uganda", "Africa", "Sub-Saharan Africa", "Low income"),
    "UKR": ("Ukraine", "Europe", "Eastern Europe", "Lower middle income"),
    "URY": ("Uruguay", "Americas", "Latin America", "High income"),
    "USA": ("United States", "Americas", "Northern America", "High income"),
    "UZB": ("Uzbekistan", "Asia", "Central Asia", "Lower middle income"),
    "VAT": ("Holy See", "Europe", "Southern Europe", "High income"),
    "VCT": ("Saint Vincent and the Grenadines", "Americas", "Caribbean", "Upper middle income"),
    "VEN": ("Venezuela", "Americas", "Latin America", "Lower middle income"),
    "VGB": ("British Virgin Islands", "Americas", "Caribbean", "High income"),
    "VIR": ("United States Virgin Islands", "Americas", "Caribbean", "High income"),
    "VNM": ("Vietnam", "Asia", "South-Eastern Asia", "Lower middle income"),
    "VUT": ("Vanuatu", "Oceania", "Melanesia", "Lower middle income"),
    "WLF": ("Wallis and Futuna Islands", "Oceania", "Polynesia", "Upper middle income"),
    "WSM": ("Samoa", "Oceania", "Polynesia", "Lower middle income"),
    "YEM": ("Yemen", "Asia", "Western Asia", "Low income"),
    "ZAF": ("South Africa", "Africa", "Sub-Saharan Africa", "Upper middle income"),
    "ZMB": ("Zambia", "Africa", "Sub-Saharan Africa", "Lower middle income"),
    "ZWE": ("Zimbabwe", "Africa", "Sub-Saharan Africa", "Low income"),
}


def _read_safe_csv(p: Path, **kwargs) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, **kwargs)


def load_ndgain() -> pd.DataFrame:
    """Load ND-GAIN climate vulnerability long CSV (iso3, year, climate_vulnerability).
    Falls back to vulnerability.csv if ndgain_country_index.csv is missing.
    """
    candidates = [
        RAW_DIR / "ndgain_country_index.csv",
        CACHE_DIR / "ndgain_climate.csv",
        RAW_DIR / "vulnerability.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                df = pd.read_csv(p, engine="python")
            cols = [c.lower() for c in df.columns]
            if set(["iso3", "year", "climate_vulnerability"]).issubset(set(cols)):
                df.columns = [c.lower() for c in df.columns]
                df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
                df["climate_vulnerability"] = pd.to_numeric(df["climate_vulnerability"], errors="coerce")
                return df[["iso3", "year", "climate_vulnerability"]].dropna(subset=["iso3", "year"])
            if df.shape[1] > 3 and ("iso3" in (c.lower() for c in df.columns) or "iso3" in (c.strip('"').lower() for c in df.columns)):
                iso_col = next((c for c in df.columns if c.lower().strip('"') == "iso3"), None)
                if iso_col is None:
                    continue
                id_vars = [iso_col]
                value_vars = [c for c in df.columns if c not in id_vars]
                melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="year", value_name="climate_vulnerability")
                melted = melted.rename(columns={iso_col: "iso3"})
                melted["iso3"] = melted["iso3"].astype(str).str.strip('"')
                melted["year"] = pd.to_numeric(melted["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce").astype(int)
                melted["climate_vulnerability"] = pd.to_numeric(melted["climate_vulnerability"], errors="coerce")
                return melted[["iso3", "year", "climate_vulnerability"]].dropna(subset=["iso3", "year"])
    return pd.DataFrame(columns=["iso3", "year", "climate_vulnerability"])


def load_conflict_long() -> pd.DataFrame:
    """Read conflict data and return long frame with columns: iso3, year, conflict_intensity."""
    candidates = [
        CACHE_DIR / "ucdp_conflict.csv",
        RAW_DIR / "Conflict_Intensity_Pivot_Updated.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "iso3" in [c.lower() for c in df.columns] and "conflict_intensity" in [c.lower() for c in df.columns]:
            df.columns = [c.lower() for c in df.columns]
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
            df["conflict_intensity"] = pd.to_numeric(df["conflict_intensity"], errors="coerce").fillna(0.0)
            return df[["iso3", "year", "conflict_intensity"]].dropna(subset=["iso3", "year"])
        iso_col = next((c for c in df.columns if c.strip().lower() in ("iso3", "iso")), df.columns[0])
        year_cols = [c for c in df.columns if str(c).strip().isdigit()]
        if not year_cols:
            year_cols = [c for c in df.columns if c not in (iso_col, "Country_Name", "Country", "country")]
        long = df.melt(id_vars=[iso_col], value_vars=year_cols, var_name="year", value_name="conflict_intensity")
        long = long.rename(columns={iso_col: "iso3"})
        long["year"] = pd.to_numeric(long["year"], errors="coerce").astype(int)
        long["conflict_intensity"] = pd.to_numeric(long["conflict_intensity"], errors="coerce").fillna(0.0)
        long = long.dropna(subset=["year", "iso3"])
        return long[["iso3", "year", "conflict_intensity"]]
    return pd.DataFrame(columns=["iso3", "year", "conflict_intensity"])


def load_henley() -> pd.DataFrame:
    """Load Henley Passport Index dataset (iso3, year, visa_openness_index)."""
    p = RAW_DIR / "henley_passport_index.csv"
    if not p.exists():
        return pd.DataFrame(columns=["iso3", "year", "visa_openness_index"])
    df = _read_safe_csv(p)
    cols = [c.lower() for c in df.columns]
    if set(["iso3", "year", "visa_openness_index"]).issubset(set(cols)):
        df.columns = [c.lower() for c in df.columns]
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        df["visa_openness_index"] = pd.to_numeric(df["visa_openness_index"], errors="coerce")
        return df[["iso3", "year", "visa_openness_index"]].dropna(subset=["iso3", "year"])
    return pd.DataFrame(columns=["iso3", "year", "visa_openness_index"])


def load_factors() -> pd.DataFrame:
    """Load processed factors panel with full schema and automatic in-memory enrichment."""
    p = PROCESSED_DIR / "factors_panel.csv"
    if not p.exists():
        p_enrich = PROCESSED_DIR / "factors_panel_enriched.csv"
        if p_enrich.exists():
            p = p_enrich
        else:
            raise FileNotFoundError(f"Processed factors file not found: {p}")
    factors = pd.read_csv(p)
    if "iso3" not in factors.columns or "year" not in factors.columns:
        raise ValueError("factors_panel.csv must contain iso3 and year columns")

    # In-memory fill if optional features are missing
    if "climate_vulnerability" not in factors.columns or factors["climate_vulnerability"].isna().all():
        nd = load_ndgain()
        if not nd.empty:
            factors = factors.merge(nd, on=["iso3", "year"], how="left")

    if "conflict_intensity" not in factors.columns or factors["conflict_intensity"].isna().all():
        conf = load_conflict_long()
        if not conf.empty:
            factors = factors.merge(conf, on=["iso3", "year"], how="left")
        factors["conflict_intensity"] = factors["conflict_intensity"].fillna(0.0)

    if "visa_openness_index" not in factors.columns or factors["visa_openness_index"].isna().all():
        hen = load_henley()
        if not hen.empty:
            factors = factors.merge(hen, on=["iso3", "year"], how="left")

    factors.columns = [c.strip() for c in factors.columns]
    return factors


def load_migration_long() -> pd.DataFrame:
    """Load processed migration long dataset."""
    p = PROCESSED_DIR / "migration_long.csv"
    if not p.exists():
        raise FileNotFoundError(f"Migration long file not found: {p}")
    return pd.read_csv(p)


def load_edges() -> pd.DataFrame:
    """Load network edges export."""
    p = EXPORTS_DIR / "network_edges.csv"
    if not p.exists():
        raise FileNotFoundError(f"Edges file not found: {p}")
    return pd.read_csv(p)


def load_country_metadata() -> pd.DataFrame:
    """Load complete country metadata with full 235 country entity reference."""
    p = EXPORTS_DIR / "country_metadata.csv"
    if p.exists():
        df = pd.read_csv(p)
        if len(df) >= 200:
            return df

    records = [
        {"iso3": iso3, "country_name": name, "continent": cont, "un_region": reg, "income_group": inc}
        for iso3, (name, cont, reg, inc) in COUNTRY_METADATA_TABLE.items()
    ]
    return pd.DataFrame(records)


def load_migration_full_flat() -> pd.DataFrame:
    """Load unified all-in-one bilateral migration flat dataset."""
    p = EXPORTS_DIR / "migration_full_flat.csv"
    if not p.exists():
        raise FileNotFoundError(f"migration_full_flat.csv not found at {p}. Run pipeline first.")
    return pd.read_csv(p)


def load_nodes_master() -> pd.DataFrame:
    """Load unified all-in-one country-year node master dataset."""
    p = EXPORTS_DIR / "nodes_master.csv"
    if not p.exists():
        raise FileNotFoundError(f"nodes_master.csv not found at {p}. Run pipeline first.")
    return pd.read_csv(p)