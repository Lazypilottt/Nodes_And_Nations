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
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.loader import COUNTRY_METADATA_TABLE


def build_country_metadata() -> pd.DataFrame:
    """Build country metadata DataFrame with complete coverage of all 235 country entities."""
    records = [
        {"iso3": iso3, "country_name": name, "continent": cont, "un_region": reg, "income_group": inc}
        for iso3, (name, cont, reg, inc) in COUNTRY_METADATA_TABLE.items()
    ]
    df_meta = pd.DataFrame(records)

    # Supplement with any ISO3 codes found in migration data if any extra exist
    migration_path = os.path.join(PROCESSED, "migration_long.csv")
    if os.path.exists(migration_path):
        df_mig = pd.read_csv(migration_path, usecols=["dest_iso3", "origin_iso3"])
        all_iso3 = set(df_mig["dest_iso3"].tolist() + df_mig["origin_iso3"].tolist())
        known = set(df_meta["iso3"].tolist())
        missing = all_iso3 - known
        if missing:
            extra = pd.DataFrame({
                "iso3": list(missing),
                "country_name": list(missing),
                "continent": "Unknown",
                "un_region": "Unknown",
                "income_group": "Unknown",
            })
            df_meta = pd.concat([df_meta, extra], ignore_index=True)

    df_meta = df_meta.drop_duplicates(subset=["iso3"]).sort_values("iso3").reset_index(drop=True)
    return df_meta


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMMUNITY LABELS
# ══════════════════════════════════════════════════════════════════════════════

def label_communities(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach community memberships to country metadata for each snapshot year.
    Derives human-readable community names based on the dominant geographic region per cluster per year.
    """
    mem_path = os.path.join(EXPORTS_DIR, "community_memberships.csv")
    if not os.path.exists(mem_path):
        print("  community_memberships.csv not found, skipping community labeling")
        return pd.DataFrame()

    df_mem = pd.read_csv(mem_path)
    # Merge with metadata to get un_region and continent
    df_merged = df_mem.merge(meta_df[["iso3", "un_region", "continent"]], on="iso3", how="left")

    def get_cluster_label(grp: pd.DataFrame) -> str:
        reg_counts = grp["un_region"].dropna().value_counts()
        cont_counts = grp["continent"].dropna().value_counts()
        if reg_counts.empty:
            return "Global"

        top_reg = reg_counts.index[0]
        top_pct = reg_counts.iloc[0] / max(len(grp), 1)

        if top_pct >= 0.55:
            return top_reg
        elif len(reg_counts) > 1:
            top1 = reg_counts.index[0]
            top2 = reg_counts.index[1]
            if "Latin America" in [top1, top2] and "Caribbean" in [top1, top2]:
                return "Latin America & Caribbean"
            elif "South Asia" in [top1, top2] and "Western Asia" in [top1, top2]:
                return "South & Western Asia"
            elif "South Asia" in [top1, top2] and "South-Eastern Asia" in [top1, top2]:
                return "South & South-East Asia"
            elif ("Eastern Europe" in [top1, top2] or "Northern Europe" in [top1, top2] or "Western Europe" in [top1, top2]) and ("Europe" in cont_counts.head(2).index.tolist()):
                return "Pan-Europe"
            else:
                return f"{top1} & {top2}"
        else:
            return top_reg

    labels = []
    for (year, comm_id), grp in df_merged.groupby(["year", "louvain_community"]):
        dom = get_cluster_label(grp)
        labels.append({
            "year": year,
            "louvain_community": comm_id,
            "dominant_region": dom,
            "community_label": f"Cluster {comm_id}: {dom}",
        })

    comm_df = pd.DataFrame(labels)
    df_labeled = df_mem.merge(comm_df, on=["year", "louvain_community"], how="left")
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
# 5. UNIFIED ALL-IN-ONE EXPORT TABLES
# ══════════════════════════════════════════════════════════════════════════════

def build_unified_tables(meta_df: pd.DataFrame, comm_labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build single all-in-one unified export tables:
    1. nodes_master.csv: all node-level metadata, centrality metrics, communities, boundary scores, and socioeconomic factors
    2. migration_full_flat.csv: all-in-one bilateral corridor table with origin and destination node metrics pre-joined
    """
    print("\n[5/6] Building unified master datasets (nodes_master.csv & migration_full_flat.csv)...")

    cent_path = os.path.join(EXPORTS_DIR, "centrality_metrics.csv")
    bnd_path = os.path.join(EXPORTS_DIR, "boundary_nodes.csv")
    fac_path = os.path.join(PROCESSED, "factors_panel.csv")
    edges_path = os.path.join(EXPORTS_DIR, "network_edges.csv")

    cent_df = pd.read_csv(cent_path) if os.path.exists(cent_path) else pd.DataFrame()
    bnd_df = pd.read_csv(bnd_path) if os.path.exists(bnd_path) else pd.DataFrame()
    fac_df = pd.read_csv(fac_path) if os.path.exists(fac_path) else pd.DataFrame()
    edges_df = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()

    # 1. Node Master Table
    node_df = cent_df.copy()
    if not node_df.empty:
        node_df = node_df.merge(meta_df, on="iso3", how="left")
        if not comm_labeled.empty and "community_label" in comm_labeled.columns:
            comm_sub = comm_labeled[["iso3", "year", "louvain_community", "leiden_community", "gn_community", "dominant_region", "community_label"]]
            node_df = node_df.merge(comm_sub, on=["iso3", "year"], how="left")
        if not bnd_df.empty and "boundary_score" in bnd_df.columns:
            node_df = node_df.merge(bnd_df[["iso3", "boundary_score", "is_boundary_node"]], on="iso3", how="left")
        if not fac_df.empty:
            node_df = node_df.merge(fac_df, on=["iso3", "year"], how="left")

        out_node = os.path.join(EXPORTS_DIR, "nodes_master.csv")
        node_df.to_csv(out_node, index=False)
        print(f"  ✓ Saved: {out_node}  ({len(node_df):,} rows, {len(node_df.columns)} columns)")

    # 2. Migration Full Flat Table (Bilateral Corridors with Origin & Dest Attributes)
    if not edges_df.empty and not node_df.empty:
        origin_nodes = node_df.rename(columns=lambda c: f"origin_{c}" if c not in ["year"] else c)
        dest_nodes = node_df.rename(columns=lambda c: f"dest_{c}" if c not in ["year"] else c)

        flat_df = edges_df.merge(origin_nodes, on=["origin_iso3", "year"], how="left")
        flat_df = flat_df.merge(dest_nodes, on=["dest_iso3", "year"], how="left")

        flat_df["log_weight"] = np.log1p(flat_df["weight"])
        if "origin_continent" in flat_df.columns and "dest_continent" in flat_df.columns:
            flat_df["same_continent"] = (flat_df["origin_continent"] == flat_df["dest_continent"]).astype(int)
        if "origin_income_group" in flat_df.columns and "dest_income_group" in flat_df.columns:
            flat_df["same_income_group"] = (flat_df["origin_income_group"] == flat_df["dest_income_group"]).astype(int)
        if "origin_louvain_community" in flat_df.columns and "dest_louvain_community" in flat_df.columns:
            flat_df["same_community"] = (flat_df["origin_louvain_community"] == flat_df["dest_louvain_community"]).astype(int)

        out_flat = os.path.join(EXPORTS_DIR, "migration_full_flat.csv")
        flat_df.to_csv(out_flat, index=False)
        print(f"  ✓ Saved: {out_flat}  ({len(flat_df):,} rows, {len(flat_df.columns)} columns)")
        return node_df, flat_df

    return node_df, pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 6. POWER BI MANIFEST
# ══════════════════════════════════════════════════════════════════════════════

def write_manifest(validation_status: dict):
    """Write a plain-text manifest describing all export files for Power BI setup."""
    manifest_lines = [
        "=" * 70,
        "NODES AND NATIONS — POWER BI DATA MANIFEST",
        "=" * 70,
        "",
        "UNIFIED ALL-IN-ONE TABLES (EASIEST / DRAG-AND-DROP)",
        "  1. migration_full_flat.csv: Comprehensive bilateral edge table with origin and destination metrics",
        "  2. nodes_master.csv: Complete country-year panel with metadata, centrality, communities, & factors",
        "",
        "DATA MODEL RELATIONSHIPS (FOR RELATIONAL STAR SCHEMA)",
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
    print("\n[2/6] Building country metadata table...")
    meta_df = build_country_metadata()
    out_meta = os.path.join(EXPORTS_DIR, "country_metadata.csv")
    meta_df.to_csv(out_meta, index=False)
    print(f"  ✓ Saved: {out_meta}  ({len(meta_df)} countries)")

    # ── 3. Community labels ───────────────────────────────────────────────────
    print("\n[3/6] Building community labels for latest year...")
    comm_labeled = label_communities(meta_df)
    if not comm_labeled.empty:
        out_cl = os.path.join(EXPORTS_DIR, "community_labels.csv")
        comm_labeled.to_csv(out_cl, index=False)
        print(f"  ✓ Saved: {out_cl}  ({len(comm_labeled):,} rows)")

    # ── 4. Summary statistics ──────────────────────────────────────────────────
    print("\n[4/6] Building Power BI KPI summary statistics...")
    summary_df = build_summary_stats()
    out_summ   = os.path.join(EXPORTS_DIR, "summary_stats.csv")
    summary_df.to_csv(out_summ, index=False)
    print(f"  ✓ Saved: {out_summ}")
    print(summary_df.to_string(index=False))

    # ── 5. Unified Master Tables ───────────────────────────────────────────────
    build_unified_tables(meta_df, comm_labeled)

    # ── 6. Manifest ───────────────────────────────────────────────────────────
    print("\n[6/6] Writing Power BI manifest...")
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
    print("  2. Load all CSVs from data/exports/ (or migration_full_flat.csv for single-table setup)")
    print("  3. Follow powerbi_manifest.txt for table relationships and dashboard setup")


if __name__ == "__main__":
    main()

