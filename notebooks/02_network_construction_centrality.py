"""
Phase 2: Network Construction & Centrality Analysis
=====================================================
Nodes and Nations: A Complex Network Study of Global Migration

Steps:
  1. Load migration_long.csv
  2. Build 8 directed weighted graph snapshots in NetworkX
  3. Compute in-degree, out-degree, betweenness, PageRank per snapshot
  4. Export centrality_metrics.csv and network_edges.csv

Inputs:  data/processed/migration_long.csv
Outputs: data/exports/centrality_metrics.csv
         data/exports/network_edges.csv
         data/exports/network_summary.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
EXPORTS_DIR = os.path.join(ROOT, "data", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

SNAPSHOT_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

# Number of top edges to keep per year for Power BI performance
TOP_EDGES_PER_YEAR = 500


# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD GRAPH SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(df_year: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted graph for one year's migration data.
    Edges: origin_iso3 -> dest_iso3, weight = migrant_stock (total)
    Excludes self-loops and zero-weight edges.
    """
    G = nx.DiGraph()
    for _, row in df_year.iterrows():
        src = row["origin_iso3"]
        dst = row["dest_iso3"]
        w   = row["migrant_stock"]
        if src == dst or w <= 0 or pd.isna(w):
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += w
        else:
            G.add_edge(src, dst, weight=float(w))
    return G


def load_graphs(migration_path: str) -> dict:
    """Load migration_long.csv and build one DiGraph per snapshot year."""
    print("Loading migration_long.csv...")
    df = pd.read_csv(migration_path)

    # Aggregate: sum over any duplicate (origin, dest, year) combos (safety)
    df = df.groupby(["dest_iso3", "origin_iso3", "year"], as_index=False)["migrant_stock"].sum()

    graphs = {}
    for year in SNAPSHOT_YEARS:
        df_y = df[df["year"] == year]
        if df_y.empty:
            print(f"  WARNING: No data for year {year}, skipping.")
            continue
        G = build_graph(df_y)
        graphs[year] = G
        print(f"  {year}: {G.number_of_nodes():>4} nodes, {G.number_of_edges():>6} edges, "
              f"total stock = {sum(d['weight'] for _,_,d in G.edges(data=True)):,.0f}")
    return graphs


# ══════════════════════════════════════════════════════════════════════════════
# 2. CENTRALITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_centralities(G: nx.DiGraph, year: int) -> pd.DataFrame:
    """
    Compute four centrality metrics for all nodes in graph G.
    Returns DataFrame with columns:
        iso3, year, in_degree, out_degree, betweenness, pagerank
    """
    nodes = list(G.nodes())
    n     = len(nodes)

    # ── Degree centralities (normalized by n-1) ──────────────────────────
    in_deg  = nx.in_degree_centrality(G)      # fraction of nodes pointing in
    out_deg = nx.out_degree_centrality(G)     # fraction of nodes pointing out

    # Raw weighted in/out degree (for export separately)
    in_strength  = {node: sum(d["weight"] for _, _, d in G.in_edges(node,  data=True)) for node in nodes}
    out_strength = {node: sum(d["weight"] for _, _, d in G.out_edges(node, data=True)) for node in nodes}

    # ── Betweenness centrality (normalized, unweighted for speed; use weight as distance) ──
    print(f"    Computing betweenness centrality for {year}... (may take ~30s for large graphs)")
    # Use weight as distance (lower weight = harder to traverse)
    # Invert weights for shortest-path-based betweenness
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        max_w = max(dd["weight"] for _, _, dd in G.edges(data=True))
        G_inv[u][v]["inv_weight"] = max_w / max(d["weight"], 1)

    betweenness = nx.betweenness_centrality(
        G_inv,
        normalized=True,
        weight="inv_weight",
        k=min(n, 200),  # Sample k nodes for speed if large graph
    )

    # ── PageRank (weight-aware, alpha=0.85) ───────────────────────────────
    print(f"    Computing PageRank for {year}...")
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    rows = []
    for node in nodes:
        rows.append({
            "iso3":          node,
            "year":          year,
            "in_degree_centrality":  in_deg.get(node, 0.0),
            "out_degree_centrality": out_deg.get(node, 0.0),
            "in_strength":           in_strength.get(node, 0.0),
            "out_strength":          out_strength.get(node, 0.0),
            "betweenness_centrality": betweenness.get(node, 0.0),
            "pagerank":              pagerank.get(node, 0.0),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. NETWORK SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_network_summary(graphs: dict) -> pd.DataFrame:
    """
    Compute global network-level statistics for each snapshot year.
    """
    rows = []
    for year, G in sorted(graphs.items()):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        G_und = G.to_undirected()

        # Weak connected components
        wcc = list(nx.weakly_connected_components(G))

        # Average clustering (undirected approximation)
        try:
            avg_clust = nx.average_clustering(G_und)
        except Exception:
            avg_clust = np.nan

        # Average shortest path length (largest WCC only)
        largest_wcc = max(wcc, key=len) if wcc else set()
        if len(largest_wcc) > 1:
            G_sub = G.subgraph(largest_wcc).to_undirected()
            try:
                avg_path = nx.average_shortest_path_length(G_sub)
            except Exception:
                avg_path = np.nan
        else:
            avg_path = np.nan

        rows.append({
            "year":                   year,
            "n_nodes":                n,
            "n_edges":                m,
            "density":                nx.density(G),
            "total_migrant_stock":    sum(weights),
            "avg_weight":             np.mean(weights) if weights else 0,
            "max_weight":             max(weights) if weights else 0,
            "n_weakly_connected_components": len(wcc),
            "largest_wcc_size":       len(largest_wcc),
            "avg_clustering":         avg_clust,
            "avg_shortest_path_length": avg_path,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. EDGE EXPORT (TOP-N per year for Power BI)
# ══════════════════════════════════════════════════════════════════════════════

def export_top_edges(graphs: dict, top_n: int = TOP_EDGES_PER_YEAR) -> pd.DataFrame:
    """
    Export top N edges by weight per year for Power BI network visualization.
    """
    all_edges = []
    for year, G in sorted(graphs.items()):
        edges = [
            {"origin_iso3": u, "dest_iso3": v, "year": year, "weight": d["weight"]}
            for u, v, d in G.edges(data=True)
        ]
        edges_df = pd.DataFrame(edges)
        if not edges_df.empty:
            top = edges_df.nlargest(top_n, "weight")
            all_edges.append(top)

    return pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 2: NETWORK CONSTRUCTION & CENTRALITY ANALYSIS")
    print("=" * 70)

    migration_path = os.path.join(PROCESSED, "migration_long.csv")
    if not os.path.exists(migration_path):
        raise FileNotFoundError(
            f"migration_long.csv not found at {migration_path}\n"
            "Please run 01_data_collection_cleaning.py first."
        )

    # ── 1. Build graphs ──────────────────────────────────────────────────────
    print("\n[1/4] Building directed weighted graph snapshots...")
    graphs = load_graphs(migration_path)
    print(f"\n  Built {len(graphs)} graph snapshots for years: {sorted(graphs.keys())}")

    # ── 2. Compute centralities ──────────────────────────────────────────────
    print("\n[2/4] Computing centrality metrics for all snapshots...")
    centrality_frames = []
    for year, G in sorted(graphs.items()):
        print(f"\n  >>> Year {year} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        df_c = compute_centralities(G, year)
        centrality_frames.append(df_c)

    centrality_df = pd.concat(centrality_frames, ignore_index=True)
    out_c = os.path.join(EXPORTS_DIR, "centrality_metrics.csv")
    centrality_df.to_csv(out_c, index=False)
    print(f"\n  ✓ Saved: {out_c}  ({len(centrality_df):,} rows)")

    # Print top-5 PageRank for the latest year
    latest_year = max(graphs.keys())
    top5 = centrality_df[centrality_df["year"] == latest_year].nlargest(5, "pagerank")
    print(f"\n  Top-5 PageRank destinations in {latest_year}:")
    print(top5[["iso3", "pagerank", "in_strength", "in_degree_centrality"]].to_string(index=False))

    # ── 3. Network summary statistics ────────────────────────────────────────
    print("\n[3/4] Computing network-level summary statistics...")
    summary_df = compute_network_summary(graphs)
    out_s = os.path.join(EXPORTS_DIR, "network_summary.csv")
    summary_df.to_csv(out_s, index=False)
    print(f"  ✓ Saved: {out_s}")
    print(summary_df[["year", "n_nodes", "n_edges", "density", "total_migrant_stock"]].to_string(index=False))

    # ── 4. Export top edges for Power BI ─────────────────────────────────────
    print(f"\n[4/4] Exporting top {TOP_EDGES_PER_YEAR} edges per year for Power BI...")
    edges_df = export_top_edges(graphs, TOP_EDGES_PER_YEAR)
    out_e = os.path.join(EXPORTS_DIR, "network_edges.csv")
    edges_df.to_csv(out_e, index=False)
    print(f"  ✓ Saved: {out_e}  ({len(edges_df):,} rows)")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
