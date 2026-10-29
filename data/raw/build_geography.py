"""
Build the dyadic geography table from CEPII GeoDist.

Downloads CEPII's GeoDist files (Mayer & Zignago, 2011), maps their 2004-era
country codes onto the UN DESA ISO3 codes used in this project, and writes:

    data/raw/cepii_dyadic.csv    one row per ordered country pair
    data/raw/cepii_country.csv   one row per country (landlocked, area, coordinates)

Run once (requires `xlrd` to read the legacy .xls files):

    python data/raw/build_geography.py

CEPII codes that changed since 2004 are renamed, and successor states are
copied from their predecessor (Serbia and Montenegro from Yugoslavia, South
Sudan from Sudan, Curaçao / Sint Maarten / Caribbean Netherlands from the
Netherlands Antilles). Nine small territories that GeoDist omits are added from
the attributes in MISSING_TERRITORIES. For every pair that involves a copied or
added country, distance is the great-circle distance between main cities.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent
CACHE_DIR = RAW_DIR / "cache" / "cepii"
DIST_URL = "https://www.cepii.fr/distance/dist_cepii.zip"
GEO_URL = "https://www.cepii.fr/distance/geo_cepii.xls"

RENAME = {"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "PAL": "PSE"}

# successor -> (predecessor, main-city lat, lon, landlocked, area km²)
SUCCESSORS = {
    "SRB": ("YUG", 44.82, 20.46, 1, 77_474),
    "MNE": ("YUG", 42.44, 19.26, 0, 13_812),
    "SSD": ("SDN", 4.85, 31.60, 1, 619_745),
    "CUW": ("ANT", 12.12, -68.93, 0, 444),
    "SXM": ("ANT", 18.03, -63.05, 0, 34),
    "BES": ("ANT", 12.15, -68.27, 0, 322),
}
# Predecessors that disappear entirely once their successors exist
RETIRED = {"YUG", "ANT"}
# Sudan keeps its code but lost the south in 2011
SDN_AREA = 1_861_484

# iso3 -> lat, lon, official languages, sovereign / former colonizer, land neighbours, landlocked, area
MISSING_TERRITORIES = {
    "ASM": (-14.28, -170.70, {"English", "Samoan"}, "USA", set(), 0, 199),
    "CHI": (49.45, -2.58, {"English", "French"}, "GBR", set(), 0, 198),
    "GUM": (13.44, 144.79, {"English"}, "USA", set(), 0, 549),
    "IMN": (54.24, -4.55, {"English"}, "GBR", set(), 0, 572),
    "LIE": (47.14, 9.52, {"German"}, None, {"CHE", "AUT"}, 1, 160),
    "MCO": (43.74, 7.42, {"French"}, None, {"FRA"}, 0, 2),
    "MYT": (-12.83, 45.17, {"French"}, "FRA", set(), 0, 374),
    "VAT": (41.90, 12.45, {"Italian"}, None, {"ITA"}, 1, 0.44),
    "VIR": (18.34, -64.93, {"English"}, "USA", set(), 0, 347),
}

DYAD_COLS = ["contig", "comlang_off", "comlang_ethno", "colony", "comcol", "col45", "smctry"]


def _download(url: str, name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if not path.exists():
        print(f"  downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _langs(row, prefixes) -> set:
    out = set()
    for p in prefixes:
        for k in range(1, 5):
            v = row.get(f"{p}_{k}")
            if isinstance(v, str) and v.strip() not in ("", "."):
                out.add(v.strip())
    return out


def load_country_table() -> pd.DataFrame:
    geo = pd.read_excel(_download(GEO_URL, "geo_cepii.xls"))
    geo = geo[geo["maincity"] == 1].drop_duplicates("iso3").copy()
    geo["iso3"] = geo["iso3"].replace(RENAME)
    geo["off_langs"] = geo.apply(lambda r: _langs(r, ["langoff"]), axis=1)
    geo["all_langs"] = geo.apply(lambda r: _langs(r, ["langoff", "lang20", "lang9"]), axis=1)
    geo["colonizers"] = geo.apply(
        lambda r: {r[c] for c in ["colonizer1", "colonizer2", "colonizer3", "colonizer4"]
                   if isinstance(r[c], str) and r[c] != "."},
        axis=1,
    )
    geo = geo.set_index("iso3")

    rows = []
    for succ, (pred, lat, lon, landlocked, area) in SUCCESSORS.items():
        r = geo.loc[pred].copy()
        r.name = succ
        r["lat"], r["lon"], r["landlocked"], r["area"] = lat, lon, landlocked, area
        rows.append(r)
    for iso, (lat, lon, langs, sov, _, landlocked, area) in MISSING_TERRITORIES.items():
        rows.append(pd.Series({
            "lat": lat, "lon": lon, "landlocked": landlocked, "area": area,
            "off_langs": langs, "all_langs": langs, "colonizers": {sov} if sov else set(),
        }, name=iso))
    geo = pd.concat([geo, pd.DataFrame(rows)])
    geo.loc["SDN", "area"] = SDN_AREA
    return geo.drop(index=list(RETIRED))


def build_dyads(geo: pd.DataFrame) -> pd.DataFrame:
    zpath = _download(DIST_URL, "dist_cepii.zip")
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(".xls"))
        dist = pd.read_excel(io.BytesIO(z.read(name)))
    dist["iso_o"] = dist["iso_o"].replace(RENAME)
    dist["iso_d"] = dist["iso_d"].replace(RENAME)
    # Population-weighted distance where CEPII has it ("." marks missing), else capital-to-capital
    dist["dist_km"] = pd.to_numeric(dist["distw"], errors="coerce").fillna(
        pd.to_numeric(dist["dist"], errors="coerce"))
    dist = dist[["iso_o", "iso_d", "dist_km"] + DYAD_COLS]

    # Copy predecessor rows onto successors, on both sides of the pair.
    frames = [dist]
    for succ, (pred, *_rest) in SUCCESSORS.items():
        for side in ("iso_o", "iso_d"):
            c = dist[dist[side] == pred].copy()
            c[side] = succ
            frames.append(c)
    # Pairs of two successors (e.g. SRB-MNE) come from the predecessor's self-pair.
    for a, (pa, *_r1) in SUCCESSORS.items():
        for b, (pb, *_r2) in SUCCESSORS.items():
            if a == b:
                continue
            c = dist[(dist["iso_o"] == pa) & (dist["iso_d"] == pb)].copy()
            c["iso_o"], c["iso_d"] = a, b
            frames.append(c)
    dist = pd.concat(frames, ignore_index=True)
    dist = dist.drop_duplicates(["iso_o", "iso_d"], keep="last")
    dist = dist[~dist["iso_o"].isin(RETIRED) & ~dist["iso_d"].isin(RETIRED)]

    # States that were one country until recently share a border and history.
    for a, b in [("SRB", "MNE"), ("SDN", "SSD")]:
        for o, d in [(a, b), (b, a)]:
            m = (dist["iso_o"] == o) & (dist["iso_d"] == d)
            dist.loc[m, ["contig", "smctry", "comlang_off", "comlang_ethno"]] = 1
    for a, b in [("CUW", "SXM"), ("CUW", "BES"), ("SXM", "BES")]:
        for o, d in [(a, b), (b, a)]:
            m = (dist["iso_o"] == o) & (dist["iso_d"] == d)
            dist.loc[m, ["smctry", "comlang_off", "comlang_ethno"]] = 1
            dist.loc[m, "contig"] = 0

    # Add the territories GeoDist omits.
    new = []
    for iso, (lat, lon, langs, sov, neighbours, *_r) in MISSING_TERRITORIES.items():
        for other, o in geo.iterrows():
            if other == iso:
                continue
            ties = o["colonizers"] if isinstance(o["colonizers"], set) else set()
            rec = {
                "contig": int(other in neighbours),
                "comlang_off": int(bool(langs & o["off_langs"])),
                "comlang_ethno": int(bool(langs & o["all_langs"])),
                "colony": int(sov is not None and other == sov),
                "comcol": int(sov is not None and sov in ties),
                "col45": int(sov is not None and other == sov),
                "smctry": int(sov is not None and other == sov),
            }
            new.append({"iso_o": iso, "iso_d": other, **rec})
            new.append({"iso_o": other, "iso_d": iso, **rec})
    dist = pd.concat([dist, pd.DataFrame(new)], ignore_index=True)
    dist = dist.drop_duplicates(["iso_o", "iso_d"], keep="last")

    # Great-circle distance for every pair touching a copied or added country.
    touched = set(SUCCESSORS) | set(MISSING_TERRITORIES)
    m = dist["iso_o"].isin(touched) | dist["iso_d"].isin(touched) | dist["dist_km"].isna()
    lat, lon = geo["lat"].astype(float), geo["lon"].astype(float)
    dist.loc[m, "dist_km"] = haversine_km(
        dist.loc[m, "iso_o"].map(lat), dist.loc[m, "iso_o"].map(lon),
        dist.loc[m, "iso_d"].map(lat), dist.loc[m, "iso_d"].map(lon),
    )
    dist = dist[dist["iso_o"] != dist["iso_d"]]
    dist = dist.rename(columns={"iso_o": "origin_iso3", "iso_d": "dest_iso3"})
    dist["dist_km"] = pd.to_numeric(dist["dist_km"]).round(1)
    dist[DYAD_COLS] = dist[DYAD_COLS].fillna(0).astype(int)
    return dist.sort_values(["origin_iso3", "dest_iso3"]).reset_index(drop=True)


def main():
    geo = load_country_table()
    dyads = build_dyads(geo)
    dyads.to_csv(RAW_DIR / "cepii_dyadic.csv", index=False)

    country = geo[["lat", "lon", "landlocked", "area"]].rename(columns={"area": "area_km2"})
    country.index.name = "iso3"
    country.reset_index().sort_values("iso3").to_csv(RAW_DIR / "cepii_country.csv", index=False)
    print(f"  cepii_dyadic.csv: {len(dyads):,} pairs, {dyads['origin_iso3'].nunique()} countries")
    print(f"  cepii_country.csv: {len(country)} countries")


if __name__ == "__main__":
    main()
