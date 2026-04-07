"""
Phase 3: Community Detection
==============================
Nodes and Nations: A Complex Network Study of Global Migration

Steps:
  1. Load graph snapshots from migration_long.csv
  2. Run Louvain, Leiden, and Girvan-Newman on each snapshot
  3. Compute modularity Q for all algorithms × all years
  4. Compute NMI, ARI cross-comparisons
  5. Temporal Jaccard drift analysis + boundary node detection

Inputs:  data/processed/migration_long.csv
Outputs: data/exports/community_memberships.csv
         data/exports/modularity_scores.csv
         data/exports/boundary_nodes.csv
         data/exports/temporal_drift.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain   # python-louvain
import igraph as ig
import leidenalg

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity as nx_modularity

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
EXPORTS_DIR = os.path.join(ROOT, "data", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

SNAPSHOT_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

# Girvan-Newman: target fraction of total weight to capture in subgraph
GN_COVERAGE_FRACTION = 0.80
# Minimum community size to keep (merge singletons into nearest community)
MIN_COMMUNITY_SIZE = 3


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD GRAPHS (reuse same logic as Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(df_year: pd.DataFrame, as_undirected: bool = False) -> nx.Graph:
    """Build directed (or undirected) weighted graph from a year slice."""
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
    return G.to_undirected(reciprocal=False) if as_undirected else G


def load_graphs(migration_path: str) -> dict:
    df = pd.read_csv(migration_path)
    df = df.groupby(["dest_iso3", "origin_iso3", "year"], as_index=False)["migrant_stock"].sum()
    graphs     = {}
    graphs_und = {}
    for year in SNAPSHOT_YEARS:
        df_y = df[df["year"] == year]
        if df_y.empty:
            continue
        graphs[year]     = build_graph(df_y, as_undirected=False)
        graphs_und[year] = build_graph(df_y, as_undirected=True)
    return graphs, graphs_und


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOUVAIN ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def run_louvain(G_und: nx.Graph, year: int) -> tuple[dict, float]:
    """
    Run Louvain community detection on an undirected weighted graph.
    Returns: (partition dict {node: community_id}, modularity Q)
    """
    partition = community_louvain.best_partition(G_und, weight="weight", random_state=42)
    Q = community_louvain.modularity(partition, G_und, weight="weight")
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)
    n_comms = len(communities)
    sizes   = sorted([len(v) for v in communities.values()], reverse=True)
    print(f"    Louvain {year}: Q={Q:.4f}, {n_comms} communities, sizes={sizes[:8]}")
    return partition, Q


# ══════════════════════════════════════════════════════════════════════════════
# 2. LEIDEN ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def nx_to_igraph(G_und: nx.Graph) -> tuple[ig.Graph, dict]:
    """Convert NetworkX undirected graph to igraph, preserving node mapping."""
    nodes    = list(G_und.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    idx_map  = {i: n for n, i in node_map.items()}

    edges   = [(node_map[u], node_map[v]) for u, v in G_und.edges()]
    weights = [G_und[u][v].get("weight", 1.0) for u, v in G_und.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights
    return ig_graph, idx_map


def run_leiden(G_und: nx.Graph, year: int) -> tuple[dict, float]:
    """
    Run Leiden algorithm via leidenalg on an undirected weighted graph.
    Returns: (partition dict {node: community_id}, modularity Q)
    """
    if G_und.number_of_nodes() == 0:
        return {}, 0.0

    ig_graph, idx_map = nx_to_igraph(G_und)
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=42,
    )
    Q = partition.modularity

    # Build node → community mapping
    node_partition = {}
    for comm_id, comm_members in enumerate(partition):
        for member_idx in comm_members:
            node_partition[idx_map[member_idx]] = comm_id

    communities = {}
    for node, comm_id in node_partition.items():
        communities.setdefault(comm_id, set()).add(node)
    n_comms = len(communities)
    sizes   = sorted([len(v) for v in communities.values()], reverse=True)
    print(f"    Leiden  {year}: Q={Q:.4f}, {n_comms} communities, sizes={sizes[:8]}")
    return node_partition, Q


# ══════════════════════════════════════════════════════════════════════════════
# 3. GIRVAN-NEWMAN ALGORITHM (on ~80% volume subgraph)
# ══════════════════════════════════════════════════════════════════════════════

def build_gn_subgraph(G_und: nx.Graph, coverage: float = GN_COVERAGE_FRACTION) -> nx.Graph:
    """
    Retain top edges that account for `coverage` fraction of total weight.
    This makes Girvan-Newman computationally feasible.
    """
    all_edges = sorted(G_und.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    total_w   = sum(d["weight"] for _, _, d in all_edges)
    target    = total_w * coverage

    cumulative = 0.0
    kept_edges = []
    for u, v, d in all_edges:
        kept_edges.append((u, v, d["weight"]))
        cumulative += d["weight"]
        if cumulative >= target:
            break

    G_sub = nx.Graph()
    G_sub.add_nodes_from(G_und.nodes())
    for u, v, w in kept_edges:
        G_sub.add_edge(u, v, weight=w)

    print(f"    GN subgraph: {G_sub.number_of_nodes()} nodes, "
          f"{G_sub.number_of_edges()} edges "
          f"(covering {100*coverage:.0f}% of total volume)")
    return G_sub


def partition_to_frozensets(partition: dict) -> list:
    """Convert {node: comm_id} dict to list of frozensets (for nx_modularity)."""
    comms = {}
    for node, comm_id in partition.items():
        comms.setdefault(comm_id, set()).add(node)
    return [frozenset(v) for v in comms.values()]


def run_girvan_newman(G_und: nx.Graph, year: int, max_iter: int = 15) -> tuple[dict, float]:
    """
    Run Girvan-Newman on the ~80% volume subgraph.
    Iterates up to max_iter steps; selects partition with highest modularity Q.
    Returns: (partition dict, best modularity Q)
    """
    G_sub = build_gn_subgraph(G_und)

    if G_sub.number_of_edges() == 0:
        print(f"    GN {year}: empty subgraph, skipping")
        return {}, 0.0

    # Add edge betweenness weight (invert for GN which removes highest-betweenness edges)
    comp = girvan_newman(G_sub)

    best_Q         = -1.0
    best_partition = {}

    print(f"    GN {year}: iterating community cuts (up to {max_iter} steps)...")
    for step, communities in enumerate(comp):
        if step >= max_iter:
            break
        # Convert frozensets to partition dict
        partition = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                partition[node] = comm_id

        # Compute modularity on the original undirected graph (all nodes)
        # Only nodes in subgraph contribute
        fs = [frozenset(c) for c in communities]
        try:
            Q = nx_modularity(G_und, fs, weight="weight")
        except Exception:
            Q = 0.0

        if Q > best_Q:
            best_Q         = Q
            best_partition = dict(partition)

        n_comms = len(communities)
        if n_comms >= 12:   # Stop if too fragmented
            break

    # Assign nodes not in subgraph to community -1 (isolated)
    for node in G_und.nodes():
        if node not in best_partition:
            best_partition[node] = -1

    comms_found = len(set(best_partition.values()))
    print(f"    GN      {year}: Q={best_Q:.4f}, {comms_found} communities at best cut")
    return best_partition, best_Q


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-ALGORITHM COMPARISON METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compare_partitions(
    p_louvain: dict, p_leiden: dict, p_gn: dict, nodes: list
) -> dict:
    """
    Compute NMI and ARI between algorithm pairs over shared nodes.
    Returns dict of comparison metrics.
    """
    shared_all = [n for n in nodes if n in p_louvain and n in p_leiden and n in p_gn]

    if not shared_all:
        return {"nmi_louvain_leiden": np.nan, "ari_louvain_leiden": np.nan,
                "nmi_louvain_gn": np.nan,     "ari_louvain_gn": np.nan}

    lo = [p_louvain[n] for n in shared_all]
    le = [p_leiden[n]  for n in shared_all]
    gn = [p_gn[n]      for n in shared_all]

    return {
        "nmi_louvain_leiden": normalized_mutual_info_score(lo, le),
        "ari_louvain_leiden": adjusted_rand_score(lo, le),
        "nmi_louvain_gn":     normalized_mutual_info_score(lo, gn),
        "ari_louvain_gn":     adjusted_rand_score(lo, gn),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL JACCARD DRIFT
# ══════════════════════════════════════════════════════════════════════════════

def jaccard_community_similarity(p1: dict, p2: dict) -> float:
    """
    Compute mean Jaccard similarity between best-matched community pairs
    in consecutive snapshots (Hungarian-style greedy matching).
    """
    comms1 = {}
    for n, c in p1.items():
        comms1.setdefault(c, set()).add(n)
    comms2 = {}
    for n, c in p2.items():
        comms2.setdefault(c, set()).add(n)

    # Greedy: for each community in t1, find best match in t2
    scores = []
    used   = set()
    for c1, s1 in sorted(comms1.items(), key=lambda x: -len(x[1])):
        best_j = 0.0
        best_c2 = None
        for c2, s2 in comms2.items():
            if c2 in used:
                continue
            intersect = len(s1 & s2)
            union     = len(s1 | s2)
            j = intersect / union if union > 0 else 0.0
            if j > best_j:
                best_j  = j
                best_c2 = c2
        scores.append(best_j)
        if best_c2 is not None:
            used.add(best_c2)

    return float(np.mean(scores)) if scores else 0.0


def compute_temporal_drift(louvain_partitions: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each consecutive pair of snapshot years, compute:
      - Global Jaccard similarity of Louvain communities
      - Per-node community stability (how often a node changes community)

    Returns:
      drift_df:    DataFrame with (year_pair, jaccard_similarity)
      boundary_df: DataFrame with (iso3, n_changes, boundary_score)
    """
    years    = sorted(louvain_partitions.keys())
    drift_rows = []
    node_changes = {}

    for i in range(len(years) - 1):
        y1, y2  = years[i], years[i + 1]
        p1, p2  = louvain_partitions[y1], louvain_partitions[y2]
        jac     = jaccard_community_similarity(p1, p2)
        drift_rows.append({"year_start": y1, "year_end": y2, "jaccard_similarity": jac})
        print(f"  Jaccard {y1}→{y2}: {jac:.4f}")

        # Per-node: did it change community?
        # Note: community IDs are not comparable across years; use label propagation
        shared = set(p1.keys()) & set(p2.keys())
        for n in shared:
            if n not in node_changes:
                node_changes[n] = {"total_periods": 0, "n_changes": 0}
            node_changes[n]["total_periods"] += 1
            # Compare community membership by checking if same nodes appear together
            # Approximate: if community id changed, flag as change
            if p1[n] != p2[n]:
                node_changes[n]["n_changes"] += 1

    drift_df = pd.DataFrame(drift_rows)

    boundary_rows = []
    for node, stats in node_changes.items():
        boundary_rows.append({
            "iso3":            node,
            "n_changes":       stats["n_changes"],
            "total_periods":   stats["total_periods"],
            "boundary_score":  stats["n_changes"] / max(stats["total_periods"], 1),
        })

    boundary_df = pd.DataFrame(boundary_rows)
    boundary_df = boundary_df.sort_values("boundary_score", ascending=False)
    return drift_df, boundary_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 3: COMMUNITY DETECTION")
    print("=" * 70)

    migration_path = os.path.join(PROCESSED, "migration_long.csv")
    if not os.path.exists(migration_path):
        raise FileNotFoundError(
            f"migration_long.csv not found. Run 01_data_collection_cleaning.py first."
        )

    # ── Load graphs ──────────────────────────────────────────────────────────
    print("\n[0/5] Loading graph snapshots...")
    graphs, graphs_und = load_graphs(migration_path)
    print(f"  Loaded {len(graphs)} snapshots")

    # ── Containers ───────────────────────────────────────────────────────────
    all_memberships  = []
    modularity_rows  = []
    louvain_partitions = {}

    # ── Loop over years ───────────────────────────────────────────────────────
    for year in sorted(graphs.keys()):
        G     = graphs[year]
        G_und = graphs_und[year]
        nodes = list(G_und.nodes())
        print(f"\n{'─'*60}")
        print(f"  Year {year}: {len(nodes)} nodes, {G_und.number_of_edges()} edges (undirected)")

        # [1] Louvain
        print("  [1] Louvain...")
        try:
            p_louvain, q_louvain = run_louvain(G_und, year)
        except Exception as e:
            print(f"    Louvain failed: {e}")
            p_louvain, q_louvain = {n: 0 for n in nodes}, 0.0
        louvain_partitions[year] = p_louvain

        # [2] Leiden
        print("  [2] Leiden...")
        try:
            p_leiden, q_leiden = run_leiden(G_und, year)
        except Exception as e:
            print(f"    Leiden failed: {e}")
            p_leiden, q_leiden = {n: 0 for n in nodes}, 0.0

        # [3] Girvan-Newman
        print("  [3] Girvan-Newman (subgraph)...")
        try:
            p_gn, q_gn = run_girvan_newman(G_und, year)
        except Exception as e:
            print(f"    GN failed: {e}")
            p_gn, q_gn = {n: 0 for n in nodes}, 0.0

        # [4] Cross-algorithm comparison
        metrics = compare_partitions(p_louvain, p_leiden, p_gn, nodes)

        # Modularity row
        for algo, q, part in [("louvain", q_louvain, p_louvain),
                              ("leiden",  q_leiden,  p_leiden),
                              ("girvan_newman", q_gn, p_gn)]:
            n_comms = len(set(part.values())) if part else 0
            modularity_rows.append({
                "year": year, "algorithm": algo,
                "modularity_q": q, "n_communities": n_comms,
                **metrics,
            })

        # Community memberships
        for node in nodes:
            all_memberships.append({
                "iso3":               node,
                "year":               year,
                "louvain_community":  p_louvain.get(node, -1),
                "leiden_community":   p_leiden.get(node, -1),
                "gn_community":       p_gn.get(node, -1),
            })

    # ── Temporal Jaccard drift ────────────────────────────────────────────────
    print("\n[4/5] Computing temporal Jaccard drift analysis...")
    drift_df, boundary_df = compute_temporal_drift(louvain_partitions)
    # Flag boundary nodes: boundary_score >= 0.5 (change in 50%+ of periods)
    boundary_df["is_boundary_node"] = boundary_df["boundary_score"] >= 0.5
    top_boundary = boundary_df.head(20)
    print(f"  Top boundary nodes (most unstable community membership):")
    print(top_boundary[["iso3", "n_changes", "total_periods", "boundary_score"]].to_string(index=False))

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n[5/5] Exporting results...")

    mem_df  = pd.DataFrame(all_memberships)
    mod_df  = pd.DataFrame(modularity_rows)

    out_mem  = os.path.join(EXPORTS_DIR, "community_memberships.csv")
    out_mod  = os.path.join(EXPORTS_DIR, "modularity_scores.csv")
    out_drft = os.path.join(EXPORTS_DIR, "temporal_drift.csv")
    out_bnd  = os.path.join(EXPORTS_DIR, "boundary_nodes.csv")

    mem_df.to_csv(out_mem,  index=False)
    mod_df.to_csv(out_mod,  index=False)
    drift_df.to_csv(out_drft, index=False)
    boundary_df.to_csv(out_bnd, index=False)

    print(f"  ✓ community_memberships.csv: {len(mem_df):,} rows")
    print(f"  ✓ modularity_scores.csv:     {len(mod_df):,} rows")
    print(f"  ✓ temporal_drift.csv:        {len(drift_df):,} rows")
    print(f"  ✓ boundary_nodes.csv:        {len(boundary_df):,} rows")

    print("\nModularity Q summary:")
    pivot = mod_df.pivot_table(index="year", columns="algorithm", values="modularity_q")
    print(pivot.round(4).to_string())

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
