"""
make_figures.py
Render the static figures in Graphs/ from the pipeline outputs.

Usage:
    python scripts/make_figures.py

Reads data/exports/, data/processed/ and the dashboard's world map geometry
(dashboard/lib/worldGeoSvg.ts, dashboard/lib/countryGeo.ts). Run it after
scripts/build_pipeline.py so the figures match the data.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "exports"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "Graphs"
OUT.mkdir(exist_ok=True)

# ── Palette (validated categorical order; single-hue sequential ramp) ────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e7e6e2"
LAND = "#e3e2de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 26,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
YEAR_LABELS = [str(y) for y in YEARS[:-1]] + ["2025*"]
FOOT_EXTRAP = "* 2025 extrapolated from the 2020→2024 UN DESA trend."


def millions(x, _=None):
    return f"{x / 1e6:.0f}M"


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def footnote(fig, text):
    fig.text(0.01, 0.01, text, color=INK_2, fontsize=8, ha="left", va="bottom")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  ✓ {path.relative_to(ROOT)}")


# ── Data ──────────────────────────────────────────────────────────────────────
meta = pd.read_csv(EXPORTS / "country_metadata.csv").set_index("iso3")
SHORT_NAMES = {
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "Türkiye": "Türkiye",
    "Republic of Korea": "South Korea",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "United Arab Emirates": "UAE",
}


def name(iso3):
    n = meta["country_name"].get(iso3, iso3)
    n = SHORT_NAMES.get(n, n)
    return n if len(n) <= 22 else iso3


nodes = pd.read_csv(EXPORTS / "nodes_master.csv")
latest = nodes[nodes["year"] == 2025].copy()


def load_map():
    """Country polygons in the dashboard's 1000×500 equirectangular space."""
    src = (ROOT / "dashboard" / "lib" / "worldGeoSvg.ts").read_text()
    shapes = {}
    for iso3, d in re.findall(r'"iso3": "([A-Z]{3})",\s*"name": "[^"]*",\s*"d": "([^"]*)"', src):
        if iso3 in ("ATA", "ATF"):
            continue
        rings = []
        for ring in re.findall(r"M([^MZ]*)Z", d):
            pts = np.array(re.findall(r"(-?[\d.]+) (-?[\d.]+)", ring), dtype=float)
            if len(pts) >= 3:
                rings.append(pts)
        shapes[iso3] = rings
    geo = (ROOT / "dashboard" / "lib" / "countryGeo.ts").read_text()
    centroids = {
        iso: ((float(lng) + 180) / 360 * 1000, (90 - float(lat)) / 180 * 500)
        for iso, lat, lng in re.findall(r'(\w{3}): \{ iso3: "\w{3}", name: "[^"]*", lat: (-?[\d.]+), lng: (-?[\d.]+)', geo)
    }
    return shapes, centroids


def map_axes(fig):
    ax = fig.add_axes([0, 0.1, 1, 0.78])
    ax.set_xlim(0, 1000)
    ax.set_ylim(420, 20)   # crop the empty southern ocean
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_countries(ax, shapes, colors, default=LAND):
    polys, fills = [], []
    for iso3, rings in shapes.items():
        for r in rings:
            polys.append(r)
            fills.append(colors.get(iso3, default))
    ax.add_collection(PolyCollection(polys, facecolors=fills, edgecolors=SURFACE, linewidths=0.3))


# ── 1. Global migrant stock ───────────────────────────────────────────────────
def fig_global_stock():
    ns = pd.read_csv(EXPORTS / "network_summary.csv").sort_values("year")
    x = np.arange(len(ns))
    y = ns["total_migrant_stock"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x[:-1], y[:-1], color=SERIES[0], lw=2, marker="o", ms=5)
    ax.plot(x[-2:], y[-2:], color=SERIES[0], lw=2, ls=(0, (3, 2)))
    ax.plot(x[-1], y[-1], marker="o", ms=6, mfc=SURFACE, mec=SERIES[0], mew=2)
    ax.fill_between(x, y, color=SERIES[0], alpha=0.07, lw=0)
    for i in (0, len(x) - 1):
        ax.annotate(f"{y[i] / 1e6:.0f}M", (x[i], y[i]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, YEAR_LABELS)
    ax.yaxis.set_major_formatter(millions)
    ax.set_ylim(0, y.max() * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Global bilateral migrant stock more than doubled since 1990")
    subtitle(ax, "People living outside their country of birth, summed over all corridors")
    footnote(fig, FOOT_EXTRAP + " Source: UN DESA International Migrant Stock 2024.")
    save(fig, "01_global_migrant_stock.png")


# ── 2. Destination continent shares ───────────────────────────────────────────
def fig_continent_shares():
    m = pd.read_csv(PROCESSED / "migration_long.csv")
    m = m[m["origin_iso3"] != m["dest_iso3"]]
    m["continent"] = m["dest_iso3"].map(meta["continent"])
    share = m.pivot_table(index="year", columns="continent", values="migrant_stock", aggfunc="sum")
    share = share.div(share.sum(axis=1), axis=0) * 100
    order = share.loc[2025].sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(YEARS))
    ends = []
    for i, cont in enumerate(order):
        vals = share[cont].reindex(YEARS).to_numpy()
        ax.plot(x, vals, color=SERIES[i], lw=2, marker="o", ms=4.5, label=cont)
        ends.append([vals[-1], vals[-1], cont])
    # Nudge end labels apart so near-equal series stay readable
    ends.sort(key=lambda e: -e[0])
    for k in range(1, len(ends)):
        ends[k][1] = min(ends[k][1], ends[k - 1][1] - 1.8)
    for val, ypos, cont in ends:
        ax.annotate(f"{cont}  {val:.0f}%", (x[-1], val), xytext=(x[-1] + 0.25, ypos), textcoords="data",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks(x, YEAR_LABELS)
    ax.set_xlim(-0.3, len(x) - 0.3)
    ax.set_ylabel("Share of global migrant stock (%)")
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Where migrants live, by continent of destination")
    subtitle(ax, "Share of the global bilateral migrant stock")
    footnote(fig, FOOT_EXTRAP)
    fig.subplots_adjust(right=0.8, bottom=0.2)
    save(fig, "02_destination_continent_share.png")


# ── 3. Top corridors ──────────────────────────────────────────────────────────
def fig_top_corridors():
    e = pd.read_csv(EXPORTS / "network_edges.csv")
    top = e[e["year"] == 2025].nlargest(15, "weight").iloc[::-1]
    labels = [f"{name(o)} → {name(d)}" for o, d in zip(top["origin_iso3"], top["dest_iso3"])]

    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(labels, top["weight"], color=SERIES[0], height=0.68, edgecolor=SURFACE, linewidth=1.5)
    for yi, v in enumerate(top["weight"]):
        ax.text(v, yi, f"  {v / 1e6:.1f}M", va="center", fontsize=8.5, color=INK)
    ax.xaxis.set_major_formatter(millions)
    ax.set_xlim(0, top["weight"].max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Largest migration corridors, 2025*")
    subtitle(ax, "Migrant stock: people born in the origin living in the destination")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "03_top_corridors_2025.png")


# ── 4. PageRank choropleth ────────────────────────────────────────────────────
def fig_pagerank_map(shapes):
    pr = latest.set_index("iso3")["pagerank"].dropna()
    pr = pr[pr > 0]
    norm = LogNorm(vmin=pr.quantile(0.05), vmax=pr.max())
    colors = {iso: SEQ(norm(np.clip(v, norm.vmin, norm.vmax))) for iso, v in pr.items()}

    fig = plt.figure(figsize=(10, 5.6))
    ax = map_axes(fig)
    draw_countries(ax, shapes, colors)
    fig.text(0.01, 0.97, "Migration attractors: weighted PageRank, 2025*", fontsize=13, weight="bold", va="top")
    fig.text(0.01, 0.925, "Darker countries draw more migrants, directly and through the countries that feed them",
             fontsize=9.5, color=INK_2, va="top")
    cax = fig.add_axes([0.66, 0.1, 0.3, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.set_label("PageRank (log scale)", fontsize=8.5, color=INK_2)
    top5 = ", ".join(name(i) for i in pr.nlargest(5).index)
    fig.text(0.01, 0.1, f"Top 5: {top5}", fontsize=9, color=INK, va="bottom")
    footnote(fig, "Grey: no data. " + FOOT_EXTRAP)
    save(fig, "04_pagerank_map_2025.png")


# ── 5. Corridor map ───────────────────────────────────────────────────────────
def fig_corridor_map(shapes, centroids):
    e = pd.read_csv(EXPORTS / "network_edges.csv")
    top = e[e["year"] == 2025].nlargest(30, "weight")
    lw = np.interp(np.log(top["weight"]), [np.log(top["weight"].min()), np.log(top["weight"].max())], [0.8, 3.6])

    fig = plt.figure(figsize=(10, 5.6))
    ax = map_axes(fig)
    draw_countries(ax, shapes, {})
    for (o, d), w in zip(top[["origin_iso3", "dest_iso3"]].itertuples(index=False), lw):
        if o not in centroids or d not in centroids:
            continue
        (x1, y1), (x2, y2) = centroids[o], centroids[d]
        dist = np.hypot(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, min(y1, y2) - min(90, dist * 0.28)
        path = MplPath([(x1, y1), (cx, cy), (x2, y2)], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
        ax.add_patch(PathPatch(path, fc="none", ec=SERIES[0], lw=w, alpha=0.75, capstyle="round"))
        ax.plot(x2, y2, "o", ms=3.2, color=SERIES[0], mec=SURFACE, mew=0.6)
    fig.text(0.01, 0.97, "The 30 largest migration corridors, 2025*", fontsize=13, weight="bold", va="top")
    fig.text(0.01, 0.925, "Line width ∝ migrant stock (log scale); dots mark destinations",
             fontsize=9.5, color=INK_2, va="top")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "05_corridor_map_2025.png")


# ── 6. Centrality rankings ────────────────────────────────────────────────────
def fig_centrality():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, col, title, color in [
        (axes[0], "pagerank", "PageRank: destinations", SERIES[0]),
        (axes[1], "betweenness_centrality", "Betweenness: bridges", SERIES[1]),
    ]:
        top = latest.nlargest(15, col).iloc[::-1]
        ax.barh([name(i) for i in top["iso3"]], top[col], color=color, height=0.68, edgecolor=SURFACE, linewidth=1.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=11, pad=10)
    fig.suptitle("Most central countries in the 2025* migration network", x=0.01, ha="left",
                 fontsize=13, weight="bold", y=0.99)
    footnote(fig, FOOT_EXTRAP)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "06_centrality_rankings_2025.png")


# ── 7. Immigrants vs emigrants ────────────────────────────────────────────────
def fig_in_out():
    d = latest[(latest["in_strength"] > 0) & (latest["out_strength"] > 0)]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(d["in_strength"], d["out_strength"], s=26, color=SERIES[0], alpha=0.75,
               edgecolor=SURFACE, linewidth=0.8)
    lo, hi = 1e2, 1e8
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(3e7, 1.6e7, "equal in & out", rotation=45, color=INK_2, fontsize=8.5, ha="center", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("Immigrant stock (in-strength)")
    ax.set_ylabel("Emigrant stock (out-strength)")
    label = set(d.nlargest(6, "in_strength")["iso3"]) | set(d.nlargest(6, "out_strength")["iso3"])
    # Greedy vertical repel in screen space so clustered labels don't overlap
    fig.canvas.draw()
    pts = d[d["iso3"].isin(label)].copy()
    pts[["px", "py"]] = ax.transData.transform(pts[["in_strength", "out_strength"]].to_numpy())
    placed = []
    for r in pts.sort_values("py", ascending=False).itertuples():
        width = 6.5 * len(name(r.iso3)) * fig.dpi / 100
        ty = r.py + 4
        for (px, py, w) in placed:
            if abs(r.px - px) < max(w, width) and abs(ty - py) < 13:
                ty = py - 13
        placed.append((r.px, ty, width))
        to_pt = 72 / fig.dpi
        ax.annotate(name(r.iso3), (r.in_strength, r.out_strength), xytext=(4, (ty - r.py) * to_pt),
                    textcoords="offset points", fontsize=8.5, color=INK, va="bottom",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5, shrinkA=0, shrinkB=3) if ty < r.py - 2 else None)
    ax.text(0.97, 0.04, "net receivers →", transform=ax.transAxes, ha="right", color=INK_2, fontsize=9)
    ax.text(0.03, 0.96, "← net senders", transform=ax.transAxes, ha="left", va="top", color=INK_2, fontsize=9)
    ax.set_title("Receivers and senders, 2025*")
    subtitle(ax, "Each dot is a country; both axes on a log scale")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "07_immigrants_vs_emigrants_2025.png")


# ── 8. Modularity ─────────────────────────────────────────────────────────────
def fig_modularity():
    mod = pd.read_csv(EXPORTS / "modularity_scores.csv")
    piv = mod.pivot(index="year", columns="algorithm", values="modularity_q").reindex(YEARS)
    x = np.arange(len(YEARS))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for algo, color, ls, lab in [("louvain", SERIES[0], "-", "Louvain"), ("leiden", SERIES[1], (0, (4, 3)), "Leiden")]:
        ax.plot(x, piv[algo], color=color, lw=2, ls=ls, marker="o", ms=5, label=lab)
    lv = piv["louvain"]
    ax.annotate(f"{lv.iloc[0]:.3f}", (x[0], lv.iloc[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(f"{lv.iloc[-1]:.3f}", (x[-1], lv.iloc[-1]), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, YEAR_LABELS)
    ax.set_ylim(0.5, 0.63)
    ax.set_ylabel("Modularity Q")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.set_title("Migration is becoming less regionally compartmentalised")
    subtitle(ax, "Weighted modularity of the best community partition; lower = more integrated network")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "08_modularity_over_time.png")


# ── 9. Temporal drift ─────────────────────────────────────────────────────────
def fig_drift():
    d = pd.read_csv(EXPORTS / "temporal_drift.csv")
    labels = [f"{a}→{str(b)[2:]}{'*' if b == 2025 else ''}" for a, b in zip(d["year_start"], d["year_end"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, d["jaccard_similarity"], color=SERIES[0], width=0.6, edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(d["jaccard_similarity"]):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Jaccard similarity")
    ax.grid(axis="x", visible=False)
    ax.set_title("How much communities change between snapshots")
    subtitle(ax, "Overlap of matched Louvain communities in consecutive years (1 = identical)")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "09_community_stability.png")


# ── 10. Regression coefficients ───────────────────────────────────────────────
def fig_regression():
    c = pd.read_csv(EXPORTS / "regression_coefficients.csv")
    preds = [("log_population", "log population"), ("log_gdp_per_capita", "log GDP per capita"),
             ("climate_vulnerability", "Climate vulnerability")]
    models = [("baseline", "Baseline"), ("full", "Full"), ("cultural_gravity", "Cultural gravity")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4), sharey=True)
    ypos = np.arange(len(models))
    for ax, (pred, plabel) in zip(axes, preds):
        ax.axvline(0, color=INK_2, lw=1)
        for k, (direction, color, off) in enumerate([("inflows", SERIES[0], 0.13), ("outflows", SERIES[1], -0.13)]):
            for j, (mkey, _) in enumerate(models):
                row = c[(c["model"] == f"{mkey}_{direction}") & (c["predictor"] == pred)]
                if row.empty:
                    continue
                r = row.iloc[0]
                yy = ypos[j] + off
                ax.plot([r.ci_lower_95, r.ci_upper_95], [yy, yy], color=color, lw=2, solid_capstyle="round")
                ax.plot(r.coefficient, yy, "o", ms=6, color=color, mec=SURFACE, mew=1.2)
        ax.set_title(plabel, fontsize=10.5, pad=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [m[1] for m in models])
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, ms=6) for c in SERIES[:2]]
    fig.text(0.01, 0.97, "What predicts migrant stock: coefficients with 95% CIs", fontsize=13, weight="bold", va="top")
    fig.text(0.01, 0.89, "Pooled OLS on log stock, standard errors clustered by country. "
             "Missing intervals: predictor not in that model.", color=INK_2, fontsize=9.5, va="top")
    fig.legend(handles, ["Immigrant stock", "Emigrant stock"], loc="upper right", bbox_to_anchor=(0.99, 0.97), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    save(fig, "10_regression_coefficients.png")


# ── 11. Model fit ─────────────────────────────────────────────────────────────
def fig_model_fit():
    comp = pd.read_csv(EXPORTS / "regression_model_comparison.csv").set_index("model")
    groups = [("baseline", "Baseline\n(3 predictors)"), ("full", "Full\n(7)"), ("cultural_gravity", "Cultural gravity\n(11)")]
    x = np.arange(len(groups))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (direction, color, lab) in enumerate([("inflows", SERIES[0], "Immigrant stock"), ("outflows", SERIES[1], "Emigrant stock")]):
        vals = [comp.loc[f"{g}_{direction}", "r_squared"] for g, _ in groups]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=color, label=lab, edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [g[1] for g in groups])
    ax.set_ylim(0, 0.95)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("R²")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Extra covariates add little explanatory power")
    subtitle(ax, "R² of each OLS specification (larger models have fewer complete observations)")
    save(fig, "11_regression_model_fit.png")


# ── 12. Feature importance ────────────────────────────────────────────────────
def fig_feature_importance():
    p = EXPORTS / "edge_feature_importance.csv"
    if not p.exists():
        print("  - skipped feature importance (run notebooks/06 first)")
        return
    fi = pd.read_csv(p).nlargest(15, "importance").iloc[::-1]
    pretty = {"log_pop": "log population", "log_gdp": "log GDP per capita",
              "gdp_log_diff": "log GDP gap (origin − dest)", "pop_log_diff": "log population gap",
              "same_continent": "same continent", "same_income_group": "same income group",
              "same_dominant_religion": "same dominant religion", "cultural_openness_diff": "cultural openness gap",
              "log_dist": "distance (log)", "contig": "shared border", "comlang_off": "common official language",
              "comlang_ethno": "common spoken language", "colony": "colonial link", "comcol": "same former coloniser",
              "col45": "colonial link after 1945", "smctry": "were one country",
              "gdp_ppp_log_diff": "log GDP (PPP) gap", "political_stability_diff": "political stability gap",
              "life_expectancy_diff": "life expectancy gap", "youth_share_diff": "under-15 share gap",
              "emigrants_lag": "diaspora abroad, last snapshot", "immigrants_lag": "immigrants, last snapshot",
              "log_gdp_ppp": "log GDP per capita (PPP)", "log_pop_density": "log population density",
              "log_area": "log land area", "refugees_abroad_per_1k": "refugees abroad per 1,000",
              "refugees_hosted_per_1k": "refugees hosted per 1,000", "idps_per_1k": "internally displaced per 1,000"}

    def label(f):
        if f in pretty:
            return pretty[f]
        side, _, rest = f.partition("_")
        if side in ("origin", "dest"):
            return f"{'Destination' if side == 'dest' else 'Origin'}: {pretty.get(rest, rest.replace('_', ' '))}"
        return f.replace("_", " ")

    labels = [label(f) for f in fi["feature"]]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(labels, fi["importance"], color=SERIES[0], height=0.68, edgecolor=SURFACE, linewidth=1.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_title("What drives the corridor-size model")
    n_total = len(pd.read_csv(p))
    subtitle(ax, f"Permutation importance on held-out corridors (top 15 of {n_total} features)")
    save(fig, "12_model_feature_importance.png")


# ── 13. Drivers: what high-migration countries share ─────────────────────────
def fig_drivers():
    p = EXPORTS / "drivers_correlations.csv"
    if not p.exists():
        print("  - skipped drivers (run notebooks/07 first)")
        return
    c = pd.read_csv(p)
    piv = {o: c[c["outcome"] == o].set_index("factor") for o in ("immigrant_share", "emigrant_rate")}
    imm, emi = piv["immigrant_share"], piv["emigrant_rate"]
    # Keep factors significant for at least one outcome, ordered by the immigration correlation
    keep = imm.index[(imm["p_partial"] < 0.05) | (emi.reindex(imm.index)["p_partial"] < 0.05)]
    order = imm.loc[keep, "rho_partial"].sort_values().index
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.4, 0.34 * len(order) + 1.6))
    ax.axvline(0, color=INK_2, lw=0.8)
    for df, color, off, lab in [(imm, SERIES[0], 0.15, "Immigrants as % of population"),
                                (emi, SERIES[1], -0.15, "Emigrants as % of native-born")]:
        d = df.reindex(order)
        sig = d["p_partial"] < 0.05
        ax.hlines(y + off, d["ci_low"], d["ci_high"], color=color, lw=1.2, alpha=0.6)
        ax.scatter(d["rho_partial"][sig], (y + off)[sig.to_numpy()], s=40, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=lab)
        ax.scatter(d["rho_partial"][~sig], (y + off)[~sig.to_numpy()], s=36, facecolor=SURFACE,
                   edgecolor=color, linewidth=1.3, zorder=3)
    labels = [imm.loc[f, "label"] + (" †" if isinstance(imm.loc[f, "note"], str) and imm.loc[f, "note"] else "")
              for f in order]
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.85, 0.85)
    ax.set_xlabel("Partial Spearman correlation (income and population held fixed), 95% bootstrap CI")
    ax.legend(loc="lower right")
    ax.set_title("What high-immigration and high-emigration countries have in common")
    subtitle(ax, "2020, countries with ≥ 1 million residents. Hollow = not significant at 5%")
    footnote(fig, "† Partly a mechanical consequence of migration. Correlation, not causation.")
    save(fig, "13_migration_drivers.png")


# ── 14. Migration hump ────────────────────────────────────────────────────────
def fig_hump():
    p = EXPORTS / "country_migration_profile.csv"
    if not p.exists():
        print("  - skipped migration hump (run notebooks/07 first)")
        return
    prof = pd.read_csv(p)
    pts = prof[(prof["year"] == 2020) & (prof["emigrant_rate"] > 0)].dropna(subset=["gdp_pc_ppp"])
    curve = pd.read_csv(EXPORTS / "migration_hump_curve.csv")
    hump = pd.read_csv(EXPORTS / "migration_hump.csv").set_index("spec")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pts["gdp_pc_ppp"], pts["emigrant_rate"], s=22, color=SERIES[0], alpha=0.45,
               edgecolor=SURFACE, linewidth=0.6, label="Countries, 2020")
    bins = curve[curve["kind"] == "bin_median"]
    ax.plot(bins["gdp_pc_ppp"], bins["emigrant_rate"], color=SERIES[1], lw=2, marker="o", ms=5,
            label="Median by income band, 1990–2020")
    fit = curve[curve["kind"] == "fit"]
    ax.plot(fit["gdp_pc_ppp"], fit["emigrant_rate"], color=INK, lw=1.3, ls=(0, (4, 3)), label="Quadratic fit")
    for iso in ("SYR", "ALB", "MEX", "IND", "USA", "PHL", "NER", "DEU"):
        r = pts[pts["iso3"] == iso]
        if len(r):
            ax.annotate(name(iso), (r["gdp_pc_ppp"].iloc[0], r["emigrant_rate"].iloc[0]), xytext=(5, 3),
                        textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k" if v >= 1000 else f"${v:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("GDP per capita, PPP (constant 2021 $, log scale)")
    ax.set_ylabel("Emigrants, % of native-born (log scale)")
    ax.legend(loc="lower right")
    pooled, within = hump.loc["pooled"], hump.loc["within_country"]
    ax.set_title("Does development reduce emigration?")
    subtitle(ax, f"Hump term p = {pooled['p_log_gdp_sq']:.2f} across countries; "
                 f"within countries the curve is {'U-shaped' if within['b_log_gdp_sq'] > 0 else 'humped'} "
                 f"(p = {within['p_log_gdp_sq']:.2f})")
    save(fig, "14_migration_hump.png")


# ── 15–16. Structural gravity ────────────────────────────────────────────────
def fig_gravity():
    p = EXPORTS / "gravity_ppml.csv"
    if not p.exists():
        print("  - skipped gravity (run notebooks/07 first)")
        return
    g = pd.read_csv(p)
    x = np.arange(len(YEARS))
    ppml = g[g["estimator"] == "ppml"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    terms = ["colony", "comcol", "contig", "comlang_ethno", "smctry"]
    for term, color in zip(terms, SERIES):
        d = ppml[ppml["term"] == term].set_index("year").reindex(YEARS)
        ax.plot(x, d["pct_effect"], color=color, lw=2, marker="o", ms=4.5, label=d["label"].dropna().iloc[0])
    ax.set_xticks(x, YEAR_LABELS)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    ax.set_ylabel("Extra migrant stock when the tie exists")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Colonial ties are fading; language and borders are not")
    subtitle(ax, "PPML gravity with origin and destination fixed effects, all country pairs incl. zeros")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "15_gravity_ties.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for est, color, ls, lab in [("ppml", SERIES[0], "-", "PPML (zero corridors kept)"),
                                ("ols_log", SERIES[1], (0, (4, 3)), "OLS on log stock (zeros dropped)")]:
        d = g[(g["estimator"] == est) & (g["term"] == "log_dist")].set_index("year").reindex(YEARS)
        ax.fill_between(x, d["ci_lower_95"], d["ci_upper_95"], color=color, alpha=0.12, lw=0)
        ax.plot(x, d["coefficient"], color=color, lw=2, ls=ls, marker="o", ms=4.5, label=lab)
    ax.set_xticks(x, YEAR_LABELS)
    ax.set_ylabel("Distance elasticity")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    ax.set_title("Distance deters migration at least as much as in 1990")
    subtitle(ax, "% change in migrant stock for a 1% increase in distance, 95% CI clustered by country pair")
    footnote(fig, FOOT_EXTRAP)
    save(fig, "16_distance_elasticity.png")


# ── 17. Model ablation ────────────────────────────────────────────────────────
def fig_ablation():
    p = EXPORTS / "edge_model_ablation.csv"
    if not p.exists():
        print("  - skipped ablation (run notebooks/06 first)")
        return
    a = pd.read_csv(p).iloc[::-1]
    colors = [SERIES[0] if s.startswith("Served") else BLUE_RAMP[1] for s in a["feature_set"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(a["feature_set"], a["r2"], color=colors, height=0.62, edgecolor=SURFACE, linewidth=1.5)
    for b, v, n in zip(bars, a["r2"], a["n_features"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}  ({n} features)", va="center", fontsize=8.5)
    ax.set_xlim(0.6, max(a["r2"]) + 0.08)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Held-out R² on log migrant stock (corridors never seen in training)")
    ax.set_title("Geography is the biggest single gain in accuracy")
    subtitle(ax, "Same corridor split and model settings for every feature set")
    save(fig, "17_model_ablation.png")


# ── 18. Displacement forecast back-tests ─────────────────────────────────────
def fig_displacement_backtest():
    p = EXPORTS / "displacement_backtest.csv"
    if not p.exists():
        print("  - skipped displacement back-test (run notebooks/08 first)")
        return
    bt = pd.read_csv(p)
    crises = bt[bt["method"] == "model"].sort_values("onset")["crisis"].tolist()
    methods = [("model", "This model", SERIES[0]), ("diaspora_only", "Follow the diaspora", SERIES[1]),
               ("distance_only", "Nearest countries", BLUE_RAMP[1])]
    y = np.arange(len(crises))
    h = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for k, (m, label, color) in enumerate(methods):
        vals = bt[bt["method"] == m].set_index("crisis").reindex(crises)["overlap"]
        ax.barh(y + (1 - k) * h, vals, height=h, color=color, edgecolor=SURFACE, linewidth=1.2,
                label=f"{label} (mean {vals.mean():.0%})")
    ax.set_yticks(y, crises)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of displaced people forecast to the right country")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")
    ax.set_title("Forecasting where refugees go, before each crisis began")
    subtitle(ax, "Trained only on UNHCR data from years before onset; compared with actual destinations 1–4 years later")
    save(fig, "18_displacement_backtest.png")


# ── 19. Corridor gaps ─────────────────────────────────────────────────────────
def fig_corridor_gaps():
    p = EXPORTS / "corridor_potential.csv"
    if not p.exists():
        print("  - skipped corridor gaps (run notebooks/09 first)")
        return
    cp = pd.read_csv(p)
    cp = cp[np.maximum(cp["migrant_stock"], cp["expected"]) >= 20_000]
    fig, ax = plt.subplots(figsize=(8, 6))
    for status, color, label in [("as_expected", GRID, "Within 4× of expected"),
                                 ("under", SERIES[1], "At least 4× smaller"),
                                 ("over", SERIES[0], "At least 4× larger")]:
        d = cp[cp["status"] == status]
        ax.scatter(d["expected"].clip(lower=50), d["migrant_stock"].clip(lower=50), s=12, color=color,
                   alpha=0.9 if status != "as_expected" else 0.6, edgecolor="none", label=label)
    lim = (50, 2e7)
    ax.plot(lim, lim, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    # The largest excesses sit close together, so label them in a column with leader lines
    over = cp[cp["status"] == "over"].nlargest(6, "gap").sort_values("migrant_stock", ascending=False)
    for k, r in enumerate(over.itertuples()):
        ax.annotate(f"{name(r.origin_iso3)} → {name(r.dest_iso3)}", (r.expected, r.migrant_stock),
                    xytext=(1.5e3, 9e6 / 1.5 ** k), textcoords="data", fontsize=7.5, color=INK_2, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.5, alpha=0.6))
    for r in cp[cp["status"] == "under"].nsmallest(5, "gap").itertuples():
        ax.annotate(f"{name(r.origin_iso3)} → {name(r.dest_iso3)}", (max(r.expected, 50), max(r.migrant_stock, 50)),
                    xytext=(5, 2), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    fmt = plt.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Expected migrant stock (gravity model)")
    ax.set_ylabel("Actual migrant stock, 2020")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.07))
    ax.set_title("Corridors far larger or smaller than geography predicts")
    subtitle(ax, "PPML gravity with origin and destination fixed effects; corridors with ≥ 20k actual or expected")
    save(fig, "19_corridor_gaps.png")


def main():
    print(f"Writing figures to {OUT.relative_to(ROOT)}/")
    shapes, centroids = load_map()
    fig_global_stock()
    fig_continent_shares()
    fig_top_corridors()
    fig_pagerank_map(shapes)
    fig_corridor_map(shapes, centroids)
    fig_centrality()
    fig_in_out()
    fig_modularity()
    fig_drift()
    fig_regression()
    fig_model_fit()
    fig_feature_importance()
    fig_drivers()
    fig_hump()
    fig_gravity()
    fig_ablation()
    fig_displacement_backtest()
    fig_corridor_gaps()


if __name__ == "__main__":
    sys.exit(main())
