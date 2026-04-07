"""
Phase 5: Power BI Export Finalization & Validation
====================================================
Nodes and Nations: A Complex Network Study of Global Migration

Steps:
  1. Validate all export CSVs for completeness and schema
  2. Generate country_metadata.csv (ISO3, name, region, income group)
  3. Generate summary_stats.csv for dashboard KPI cards
  4. Generate community_labels.csv with human-readable community names
  5. Print a final manifest of all exported files

Inputs:  All files in data/exports/ and data/processed/
Outputs: data/exports/country_metadata.csv
         data/exports/summary_stats.csv
         data/exports/community_labels.csv
         data/exports/powerbi_manifest.txt
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
EXPORTS_DIR = os.path.join(ROOT, "data", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

SNAPSHOT_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

# ── Expected export files (for validation) ────────────────────────────────────
EXPECTED_EXPORTS = {
    "centrality_metrics.csv":         ["iso3", "year", "in_degree_centrality", "out_degree_centrality",
                                        "betweenness_centrality", "pagerank"],
    "network_edges.csv":              ["origin_iso3", "dest_iso3", "year", "weight"],
    "network_summary.csv":            ["year", "n_nodes", "n_edges", "density"],
    "community_memberships.csv":      ["iso3", "year", "louvain_community"],
    "modularity_scores.csv":          ["year", "algorithm", "modularity_q"],
    "boundary_nodes.csv":             ["iso3", "n_changes", "boundary_score"],
    "temporal_drift.csv":             ["year_start", "year_end", "jaccard_similarity"],
    "regression_coefficients.csv":    ["model", "dependent", "predictor", "coefficient", "p_value"],
    "regression_model_comparison.csv":["model", "r_squared", "adj_r2", "aic"],
    "vif_analysis.csv":               ["predictor", "vif"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. VALIDATE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

def validate_exports() -> dict:
    """Check all expected export files exist and have required columns."""
    print("\n[1/5] Validating export files...")
    status = {}
    for filename, required_cols in EXPECTED_EXPORTS.items():
        path = os.path.join(EXPORTS_DIR, filename)
        if not os.path.exists(path):
            status[filename] = {"exists": False, "rows": 0, "missing_cols": required_cols}
            print(f"  ✗ MISSING: {filename}")
            continue
        try:
            df = pd.read_csv(path, nrows=5)
            missing = [c for c in required_cols if c not in df.columns]
            df_full = pd.read_csv(path)
            status[filename] = {
                "exists":       True,
                "rows":         len(df_full),
                "missing_cols": missing,
                "columns":      df_full.columns.tolist(),
            }
            if missing:
                print(f"  ⚠ {filename}: {len(df_full):,} rows, MISSING COLS: {missing}")
            else:
                print(f"  ✓ {filename}: {len(df_full):,} rows, {len(df_full.columns)} cols")
        except Exception as e:
            status[filename] = {"exists": True, "rows": 0, "error": str(e)}
            print(f"  ✗ ERROR reading {filename}: {e}")
    return status


# ══════════════════════════════════════════════════════════════════════════════
# 2. COUNTRY METADATA
# ══════════════════════════════════════════════════════════════════════════════

# World Bank income groups and UN regions — compact reference table
COUNTRY_META_STATIC = [
    ("AFG","Afghanistan","Asia","South Asia","Low income"),
    ("ALB","Albania","Europe","Eastern Europe","Upper middle income"),
    ("DZA","Algeria","Africa","Northern Africa","Lower middle income"),
    ("AGO","Angola","Africa","Sub-Saharan Africa","Lower middle income"),
    ("ARG","Argentina","Americas","Latin America","Upper middle income"),
    ("ARM","Armenia","Asia","Western Asia","Upper middle income"),
    ("AUS","Australia","Oceania","Australia/NZ","High income"),
    ("AUT","Austria","Europe","Western Europe","High income"),
    ("AZE","Azerbaijan","Asia","Western Asia","Upper middle income"),
    ("BGD","Bangladesh","Asia","South Asia","Lower middle income"),
    ("BLR","Belarus","Europe","Eastern Europe","Upper middle income"),
    ("BEL","Belgium","Europe","Western Europe","High income"),
    ("BEN","Benin","Africa","Sub-Saharan Africa","Low income"),
    ("BOL","Bolivia","Americas","Latin America","Lower middle income"),
    ("BIH","Bosnia and Herzegovina","Europe","Eastern Europe","Upper middle income"),
    ("BWA","Botswana","Africa","Sub-Saharan Africa","Upper middle income"),
    ("BRA","Brazil","Americas","Latin America","Upper middle income"),
    ("BGR","Bulgaria","Europe","Eastern Europe","Upper middle income"),
    ("BFA","Burkina Faso","Africa","Sub-Saharan Africa","Low income"),
    ("BDI","Burundi","Africa","Sub-Saharan Africa","Low income"),
    ("KHM","Cambodia","Asia","South-Eastern Asia","Lower middle income"),
    ("CMR","Cameroon","Africa","Sub-Saharan Africa","Lower middle income"),
    ("CAN","Canada","Americas","Northern America","High income"),
    ("CAF","Central African Republic","Africa","Sub-Saharan Africa","Low income"),
    ("TCD","Chad","Africa","Sub-Saharan Africa","Low income"),
    ("CHL","Chile","Americas","Latin America","High income"),
    ("CHN","China","Asia","Eastern Asia","Upper middle income"),
    ("COL","Colombia","Americas","Latin America","Upper middle income"),
    ("COG","Congo","Africa","Sub-Saharan Africa","Lower middle income"),
    ("COD","DR Congo","Africa","Sub-Saharan Africa","Low income"),
    ("CRI","Costa Rica","Americas","Latin America","Upper middle income"),
    ("HRV","Croatia","Europe","Eastern Europe","High income"),
    ("CUB","Cuba","Americas","Latin America","Upper middle income"),
    ("CYP","Cyprus","Europe","Western Asia","High income"),
    ("CZE","Czechia","Europe","Eastern Europe","High income"),
    ("DNK","Denmark","Europe","Northern Europe","High income"),
    ("DOM","Dominican Republic","Americas","Latin America","Upper middle income"),
    ("ECU","Ecuador","Americas","Latin America","Upper middle income"),
    ("EGY","Egypt","Africa","Northern Africa","Lower middle income"),
    ("SLV","El Salvador","Americas","Latin America","Lower middle income"),
    ("ETH","Ethiopia","Africa","Sub-Saharan Africa","Low income"),
    ("FIN","Finland","Europe","Northern Europe","High income"),
    ("FRA","France","Europe","Western Europe","High income"),
    ("GAB","Gabon","Africa","Sub-Saharan Africa","Upper middle income"),
    ("GMB","Gambia","Africa","Sub-Saharan Africa","Low income"),
    ("GEO","Georgia","Asia","Western Asia","Upper middle income"),
    ("DEU","Germany","Europe","Western Europe","High income"),
    ("GHA","Ghana","Africa","Sub-Saharan Africa","Lower middle income"),
    ("GRC","Greece","Europe","Southern Europe","High income"),
    ("GTM","Guatemala","Americas","Latin America","Upper middle income"),
    ("GIN","Guinea","Africa","Sub-Saharan Africa","Low income"),
    ("HND","Honduras","Americas","Latin America","Lower middle income"),
    ("HUN","Hungary","Europe","Eastern Europe","High income"),
    ("IND","India","Asia","South Asia","Lower middle income"),
    ("IDN","Indonesia","Asia","South-Eastern Asia","Upper middle income"),
    ("IRN","Iran","Asia","Western Asia","Lower middle income"),
    ("IRQ","Iraq","Asia","Western Asia","Upper middle income"),
    ("IRL","Ireland","Europe","Northern Europe","High income"),
    ("ISR","Israel","Asia","Western Asia","High income"),
    ("ITA","Italy","Europe","Southern Europe","High income"),
    ("JAM","Jamaica","Americas","Latin America","Upper middle income"),
    ("JPN","Japan","Asia","Eastern Asia","High income"),
    ("JOR","Jordan","Asia","Western Asia","Upper middle income"),
    ("KAZ","Kazakhstan","Asia","Central Asia","Upper middle income"),
    ("KEN","Kenya","Africa","Sub-Saharan Africa","Lower middle income"),
    ("PRK","North Korea","Asia","Eastern Asia","Low income"),
    ("KOR","South Korea","Asia","Eastern Asia","High income"),
    ("KWT","Kuwait","Asia","Western Asia","High income"),
    ("KGZ","Kyrgyzstan","Asia","Central Asia","Lower middle income"),
    ("LAO","Laos","Asia","South-Eastern Asia","Lower middle income"),
    ("LBN","Lebanon","Asia","Western Asia","Lower middle income"),
    ("LSO","Lesotho","Africa","Sub-Saharan Africa","Lower middle income"),
    ("LBR","Liberia","Africa","Sub-Saharan Africa","Low income"),
    ("LBY","Libya","Africa","Northern Africa","Upper middle income"),
    ("LTU","Lithuania","Europe","Northern Europe","High income"),
    ("LUX","Luxembourg","Europe","Western Europe","High income"),
    ("MDG","Madagascar","Africa","Sub-Saharan Africa","Low income"),
    ("MWI","Malawi","Africa","Sub-Saharan Africa","Low income"),
    ("MYS","Malaysia","Asia","South-Eastern Asia","Upper middle income"),
    ("MLI","Mali","Africa","Sub-Saharan Africa","Low income"),
    ("MLT","Malta","Europe","Southern Europe","High income"),
    ("MRT","Mauritania","Africa","Sub-Saharan Africa","Lower middle income"),
    ("MUS","Mauritius","Africa","Sub-Saharan Africa","High income"),
    ("MEX","Mexico","Americas","Latin America","Upper middle income"),
    ("MDA","Moldova","Europe","Eastern Europe","Lower middle income"),
    ("MNG","Mongolia","Asia","Eastern Asia","Lower middle income"),
    ("MAR","Morocco","Africa","Northern Africa","Lower middle income"),
    ("MOZ","Mozambique","Africa","Sub-Saharan Africa","Low income"),
    ("NAM","Namibia","Africa","Sub-Saharan Africa","Upper middle income"),
    ("NPL","Nepal","Asia","South Asia","Lower middle income"),
    ("NLD","Netherlands","Europe","Western Europe","High income"),
    ("NZL","New Zealand","Oceania","Australia/NZ","High income"),
    ("NIC","Nicaragua","Americas","Latin America","Lower middle income"),
    ("NER","Niger","Africa","Sub-Saharan Africa","Low income"),
    ("NGA","Nigeria","Africa","Sub-Saharan Africa","Lower middle income"),
    ("NOR","Norway","Europe","Northern Europe","High income"),
    ("OMN","Oman","Asia","Western Asia","High income"),
    ("PAK","Pakistan","Asia","South Asia","Lower middle income"),
    ("PAN","Panama","Americas","Latin America","High income"),
    ("PNG","Papua New Guinea","Oceania","Melanesia","Lower middle income"),
    ("PRY","Paraguay","Americas","Latin America","Upper middle income"),
    ("PER","Peru","Americas","Latin America","Upper middle income"),
    ("PHL","Philippines","Asia","South-Eastern Asia","Lower middle income"),
    ("POL","Poland","Europe","Eastern Europe","High income"),
    ("PRT","Portugal","Europe","Southern Europe","High income"),
    ("QAT","Qatar","Asia","Western Asia","High income"),
    ("ROU","Romania","Europe","Eastern Europe","High income"),
    ("RUS","Russia","Europe","Eastern Europe","Upper middle income"),
    ("RWA","Rwanda","Africa","Sub-Saharan Africa","Low income"),
    ("SAU","Saudi Arabia","Asia","Western Asia","High income"),
    ("SEN","Senegal","Africa","Sub-Saharan Africa","Lower middle income"),
    ("SLE","Sierra Leone","Africa","Sub-Saharan Africa","Low income"),
    ("SOM","Somalia","Africa","Sub-Saharan Africa","Low income"),
    ("ZAF","South Africa","Africa","Sub-Saharan Africa","Upper middle income"),
    ("SSD","South Sudan","Africa","Sub-Saharan Africa","Low income"),
    ("ESP","Spain","Europe","Southern Europe","High income"),
    ("LKA","Sri Lanka","Asia","South Asia","Lower middle income"),
    ("SDN","Sudan","Africa","Sub-Saharan Africa","Low income"),
    ("SWE","Sweden","Europe","Northern Europe","High income"),
    ("CHE","Switzerland","Europe","Western Europe","High income"),
    ("SYR","Syria","Asia","Western Asia","Low income"),
    ("TJK","Tajikistan","Asia","Central Asia","Low income"),
    ("THA","Thailand","Asia","South-Eastern Asia","Upper middle income"),
    ("TLS","Timor-Leste","Asia","South-Eastern Asia","Lower middle income"),
    ("TGO","Togo","Africa","Sub-Saharan Africa","Low income"),
    ("TTO","Trinidad and Tobago","Americas","Latin America","High income"),
    ("TUN","Tunisia","Africa","Northern Africa","Lower middle income"),
    ("TUR","Turkey","Asia","Western Asia","Upper middle income"),
    ("TKM","Turkmenistan","Asia","Central Asia","Upper middle income"),
    ("UGA","Uganda","Africa","Sub-Saharan Africa","Low income"),
    ("UKR","Ukraine","Europe","Eastern Europe","Lower middle income"),
    ("ARE","United Arab Emirates","Asia","Western Asia","High income"),
    ("GBR","United Kingdom","Europe","Northern Europe","High income"),
    ("USA","United States","Americas","Northern America","High income"),
    ("URY","Uruguay","Americas","Latin America","High income"),
    ("UZB","Uzbekistan","Asia","Central Asia","Lower middle income"),
    ("VNM","Vietnam","Asia","South-Eastern Asia","Lower middle income"),
    ("YEM","Yemen","Asia","Western Asia","Low income"),
    ("ZMB","Zambia","Africa","Sub-Saharan Africa","Lower middle income"),
    ("ZWE","Zimbabwe","Africa","Sub-Saharan Africa","Low income"),
    ("SRB","Serbia","Europe","Eastern Europe","Upper middle income"),
    ("MNE","Montenegro","Europe","Eastern Europe","Upper middle income"),
    ("MKD","North Macedonia","Europe","Eastern Europe","Upper middle income"),
    ("LVA","Latvia","Europe","Northern Europe","High income"),
    ("EST","Estonia","Europe","Northern Europe","High income"),
    ("SVK","Slovakia","Europe","Eastern Europe","High income"),
    ("SVN","Slovenia","Europe","Eastern Europe","High income"),
    ("ERI","Eritrea","Africa","Sub-Saharan Africa","Low income"),
    ("TZA","Tanzania","Africa","Sub-Saharan Africa","Lower middle income"),
    ("GNQ","Equatorial Guinea","Africa","Sub-Saharan Africa","Upper middle income"),
    ("PSE","Palestine","Asia","Western Asia","Lower middle income"),
    ("KNA","Saint Kitts and Nevis","Americas","Latin America","High income"),
    ("LCA","Saint Lucia","Americas","Latin America","Upper middle income"),
    ("WSM","Samoa","Oceania","Polynesia","Lower middle income"),
    ("STP","Sao Tome and Principe","Africa","Sub-Saharan Africa","Lower middle income"),
    ("MDV","Maldives","Asia","South Asia","Upper middle income"),
]


def build_country_metadata() -> pd.DataFrame:
    """Build country metadata DataFrame from static table + any ISO3 seen in exports."""
    df_meta = pd.DataFrame(COUNTRY_META_STATIC,
                           columns=["iso3", "country_name", "continent",
                                    "un_region", "income_group"])

    # Supplement with any ISO3 codes found in migration data but not in static table
    migration_path = os.path.join(PROCESSED, "migration_long.csv")
    if os.path.exists(migration_path):
        df_mig = pd.read_csv(migration_path, usecols=["dest_iso3", "origin_iso3"])
        all_iso3 = set(df_mig["dest_iso3"].tolist() + df_mig["origin_iso3"].tolist())
        known    = set(df_meta["iso3"].tolist())
        missing  = all_iso3 - known
        if missing:
            extra = pd.DataFrame({
                "iso3":         list(missing),
                "country_name": list(missing),   # fallback: use ISO3 as name
                "continent":    "Unknown",
                "un_region":    "Unknown",
                "income_group": "Unknown",
            })
            df_meta = pd.concat([df_meta, extra], ignore_index=True)
            print(f"  Added {len(missing)} unknown ISO3 codes with placeholder metadata: {sorted(missing)[:10]}")

    df_meta = df_meta.drop_duplicates(subset=["iso3"])
    return df_meta


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMMUNITY LABELS
# ══════════════════════════════════════════════════════════════════════════════

def label_communities(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach community memberships to country metadata for the latest year.
    Derives human-readable community names based on the dominant region per cluster.
    """
    mem_path = os.path.join(EXPORTS_DIR, "community_memberships.csv")
    if not os.path.exists(mem_path):
        print("  community_memberships.csv not found, skipping community labeling")
        return pd.DataFrame()

    df_mem = pd.read_csv(mem_path)
    latest = df_mem["year"].max()
    df_latest = df_mem[df_mem["year"] == latest].copy()

    # Merge with metadata to get regions
    df_merged = df_latest.merge(meta_df[["iso3", "un_region"]], on="iso3", how="left")

    # For each Louvain community, find the dominant region
    def dominant_region(group):
        return group["un_region"].value_counts().idxmax() if not group.empty else "Unknown"

    community_labels = (
        df_merged.groupby("louvain_community")
        .apply(dominant_region)
        .reset_index()
    )
    community_labels.columns = ["louvain_community", "dominant_region"]
    community_labels["community_label"] = (
        "Cluster " + community_labels["louvain_community"].astype(str)
        + ": " + community_labels["dominant_region"]
    )

    df_labeled = df_mem.merge(community_labels, on="louvain_community", how="left")
    return df_labeled


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY STATISTICS (Power BI KPI cards)
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_stats() -> pd.DataFrame:
    """Compute top-level dashboard KPI metrics."""
    rows = []

    # From network summary
    ns_path = os.path.join(EXPORTS_DIR, "network_summary.csv")
    if os.path.exists(ns_path):
        ns = pd.read_csv(ns_path)
        latest = ns.loc[ns["year"].idxmax()]
        earliest = ns.loc[ns["year"].idxmin()]
        rows.extend([
            {"metric": "total_countries_latest",    "value": int(latest["n_nodes"]),
             "year": int(latest["year"]), "description": "Countries in migration network (latest)"},
            {"metric": "total_corridors_latest",    "value": int(latest["n_edges"]),
             "year": int(latest["year"]), "description": "Active migration corridors (latest)"},
            {"metric": "total_migrant_stock_latest","value": float(latest["total_migrant_stock"]),
             "year": int(latest["year"]), "description": "Total migrant stock (latest)"},
            {"metric": "stock_growth_pct",          "value": round(
                 100 * (latest["total_migrant_stock"] - earliest["total_migrant_stock"])
                       / max(earliest["total_migrant_stock"], 1), 1),
             "year": None, "description": "% growth in global migrant stock 1990-latest"},
        ])

    # From modulariy scores
    mod_path = os.path.join(EXPORTS_DIR, "modularity_scores.csv")
    if os.path.exists(mod_path):
        mod = pd.read_csv(mod_path)
        best_louvain = mod[mod["algorithm"] == "louvain"].nlargest(1, "modularity_q")
        if not best_louvain.empty:
            rows.append({
                "metric": "best_louvain_modularity",
                "value":  round(float(best_louvain["modularity_q"].values[0]), 4),
                "year":   int(best_louvain["year"].values[0]),
                "description": "Best Louvain modularity Q (peak year)",
            })
        n_comms_latest = mod[(mod["algorithm"] == "louvain") &
                             (mod["year"] == mod["year"].max())]["n_communities"]
        if not n_comms_latest.empty:
            rows.append({
                "metric": "n_communities_latest",
                "value":  int(n_comms_latest.values[0]),
                "year":   int(mod["year"].max()),
                "description": "Louvain communities detected (latest year)",
            })

    # From regression
    reg_path = os.path.join(EXPORTS_DIR, "regression_model_comparison.csv")
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path)
        full_model = reg[reg["model"].str.startswith("full_inflows")]
        if not full_model.empty:
            rows.append({
                "metric": "full_model_r2_inflows",
                "value":  round(float(full_model["r_squared"].values[0]), 4),
                "year":   None,
                "description": "R² of full 7-factor model (inflows)",
            })

    # From boundary nodes
    bnd_path = os.path.join(EXPORTS_DIR, "boundary_nodes.csv")
    if os.path.exists(bnd_path):
        bnd = pd.read_csv(bnd_path)
        n_boundary = bnd[bnd["boundary_score"] >= 0.5].shape[0] if "boundary_score" in bnd.columns else 0
        rows.append({
            "metric": "n_boundary_nodes",
            "value":  n_boundary,
            "year":   None,
            "description": "Countries that changed community in 50%+ of periods",
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5. POWER BI MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def write_manifest(validation_status: dict):
    """Write a plain-text manifest describing all export files for Power BI setup."""
    manifest_lines = [
        "=" * 70,
        "NODES AND NATIONS — POWER BI DATA MANIFEST",
        "=" * 70,
        "",
        "DATA MODEL RELATIONSHIPS",
        "  All tables join on [iso3] and/or [year] fields.",
        "  Primary key: [iso3] in country_metadata.csv",
        "  Foreign keys: [iso3] in all other tables",
        "",
        "DASHBOARD ↔ TABLE MAPPING",
        "  Dashboard 1 (Global Overview):  network_summary.csv, network_edges.csv, country_metadata.csv",
        "  Dashboard 2 (Centrality):       centrality_metrics.csv, country_metadata.csv",
        "  Dashboard 3 (Communities):      community_memberships.csv, community_labels.csv, modularity_scores.csv",
        "  Dashboard 4 (Factors):          regression_coefficients.csv, factors (from factors_panel)",
        "  Dashboard 5 (Regression):       regression_model_comparison.csv, regression_coefficients.csv, vif_analysis.csv",
        "",
        "FILE LIST",
    ]

    for filename, status in validation_status.items():
        exists = status.get("exists", False)
        rows   = status.get("rows", 0)
        missing_cols = status.get("missing_cols", [])
        flag = "✓" if exists and not missing_cols else ("⚠" if exists else "✗")
        manifest_lines.append(f"  [{flag}] {filename:<45} {rows:>8,} rows")
        if missing_cols:
            manifest_lines.append(f"       MISSING COLUMNS: {missing_cols}")

    manifest_lines += [
        "",
        "NOTES FOR POWER BI",
        "  • Load files via: Home > Get Data > Text/CSV",
        "  • Set year fields as Whole Number type",
        "  • Set iso3/origin_iso3/dest_iso3 as Text (not auto-detected as number)",
        "  • For choropleth maps: use [iso3] as Location field (ISO 3166-1 alpha-3)",
        "  • Network graph: use Force-Directed Graph custom visual from AppSource",
        "  • Filter network_edges.csv to one year at a time for performance",
        "  • is_extrapolated=True rows in migration_long.csv = 2025 projections",
        "",
        "RECOMMENDED DAX MEASURES",
        "  Migration Growth Rate = DIVIDE([Stock Latest] - [Stock Previous], [Stock Previous])",
        "  Modularity Trend = CALCULATE(AVERAGE(modularity_scores[modularity_q]), ...)",
        "",
    ]

    manifest_path = os.path.join(EXPORTS_DIR, "powerbi_manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines))
    print(f"  ✓ Saved: {manifest_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 5: POWER BI EXPORT FINALIZATION & VALIDATION")
    print("=" * 70)

    # ── 1. Validate ───────────────────────────────────────────────────────────
    validation = validate_exports()

    # ── 2. Country metadata ───────────────────────────────────────────────────
    print("\n[2/5] Building country metadata table...")
    meta_df = build_country_metadata()
    out_meta = os.path.join(EXPORTS_DIR, "country_metadata.csv")
    meta_df.to_csv(out_meta, index=False)
    print(f"  ✓ Saved: {out_meta}  ({len(meta_df)} countries)")

    # ── 3. Community labels ───────────────────────────────────────────────────
    print("\n[3/5] Building community labels for latest year...")
    comm_labeled = label_communities(meta_df)
    if not comm_labeled.empty:
        out_cl = os.path.join(EXPORTS_DIR, "community_labels.csv")
        comm_labeled.to_csv(out_cl, index=False)
        print(f"  ✓ Saved: {out_cl}  ({len(comm_labeled):,} rows)")

    # ── 4. Summary statistics ──────────────────────────────────────────────────
    print("\n[4/5] Building Power BI KPI summary statistics...")
    summary_df = build_summary_stats()
    out_summ   = os.path.join(EXPORTS_DIR, "summary_stats.csv")
    summary_df.to_csv(out_summ, index=False)
    print(f"  ✓ Saved: {out_summ}")
    print(summary_df.to_string(index=False))

    # ── 5. Manifest ───────────────────────────────────────────────────────────
    print("\n[5/5] Writing Power BI manifest...")
    write_manifest(validation)

    # ── Final file listing ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL EXPORT FILES:")
    print("=" * 70)
    for f in sorted(os.listdir(EXPORTS_DIR)):
        path = os.path.join(EXPORTS_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:<50} {size:>10,} bytes")

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE — PROJECT PIPELINE DONE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Open Power BI Desktop")
    print("  2. Load all CSVs from data/exports/")
    print("  3. Follow powerbi_manifest.txt for table relationships and dashboard setup")


if __name__ == "__main__":
    main()
