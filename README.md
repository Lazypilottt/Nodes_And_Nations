# Nodes & Nations

**A network analysis of global migration, 1990–2025.**

[![CI](https://github.com/Lazypilottt/Nodes_And_Nations/actions/workflows/ci.yml/badge.svg)](https://github.com/Lazypilottt/Nodes_And_Nations/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Nodes & Nations treats the world's countries as nodes and bilateral migrant
stock as weighted, directed edges. Using 35 years of UN DESA data, it builds
eight network snapshots, measures which countries act as hubs and bridges,
detects regional migration blocs and how they shift over time, tests which
socio-economic factors explain migration, and trains a model that predicts
corridor size. An interactive dashboard puts all of it on a map, alongside a
displacement scenario simulator.

---

## Key findings

| | 1990 | 2025* |
| --- | --- | --- |
| Bilateral migrant stock | 140.8 M | 293.2 M |
| Active corridors | 8,006 | 9,042 |
| Louvain modularity *Q* | 0.604 | 0.544 |

<sub>*2025 is extrapolated from the 2020→2024 UN DESA trend.</sub>

- **The network is becoming more integrated.** Modularity declines steadily
  from 0.604 (1990) to 0.544 (2025): migration is less confined to regional
  blocs than it used to be. Louvain and Leiden agree closely in every year.
- **Most countries keep their bloc.** After tracking communities across years,
  only five countries (Aruba, Venezuela, Colombia, Eritrea, Marshall Islands)
  switch community in at least four of the seven transitions.
- **Size and income dominate.** In pooled OLS with country-clustered standard
  errors, log population and log GDP per capita are the strongest predictors of
  immigrant stock, and climate vulnerability has a significant negative effect.
  Population, GDP and conflict alone explain R² ≈ 0.69 of immigrant stock and
  0.77 of emigrant stock.
- **Modelled cultural indices add little.** Adding the integration-policy and
  cultural-openness indices does not significantly improve the inflow model
  (see the [data caveats](#data-sources--licensing)).
- **The largest corridor** in 2025 is Mexico → United States (≈11.3 M),
  followed by Afghanistan → Iran, Syria → Türkiye and Russia ↔ Ukraine.

<p align="center">
  <img src="Graphs/05_corridor_map_2025.png" alt="Map of the 30 largest migration corridors in 2025" width="85%">
</p>
<p align="center">
  <img src="Graphs/01_global_migrant_stock.png" alt="Global migrant stock 1990–2025" width="49%">
  <img src="Graphs/08_modularity_over_time.png" alt="Network modularity 1990–2025" width="49%">
</p>

All figures live in [`Graphs/`](Graphs/) and are regenerated from the
pipeline outputs with `python scripts/make_figures.py`.

---

## Repository layout

```
api/            FastAPI service for the corridor-size model
dashboard/      Next.js dashboard (see dashboard/README.md)
data/
  raw/          Source datasets (UN DESA, ND-GAIN, Henley, UCDP, modelled indices)
  processed/    Cleaned long-format migration table and country-year factors panel
  exports/      Analysis outputs consumed by the dashboard
  loader.py     Shared paths and loaders
  features.py   Dyad feature engineering shared by training and the API
models/         Trained edge-weight model
notebooks/      Pipeline steps 01–06 (plain Python scripts)
scripts/
  build_pipeline.py   Runs every pipeline step in order
  make_figures.py     Regenerates the figures in Graphs/
Graphs/         Static figures (PNG) generated from the exports
docs/           Project report
```

## Quick start

Requirements: Python 3.9+ (3.11 recommended) and Node.js 18.17+.

```bash
git clone https://github.com/Lazypilottt/Nodes_And_Nations.git
cd Nodes_And_Nations

python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: the exports are committed, so this is only needed after changes
python scripts/build_pipeline.py
python scripts/make_figures.py

cd dashboard
npm install
npm run dev                           # http://localhost:3000
```

## The pipeline

Each step is a standalone script in `notebooks/` and can be run on its own.
`scripts/build_pipeline.py` runs them in order (`--skip-graph` skips 02,
`--skip-train` skips 06).

| Step | Script | What it does |
| --- | --- | --- |
| 01 | `01_data_collection_cleaning.py` | Parses the UN DESA origin–destination matrix (both sexes), extrapolates 2025, and builds a country-year panel of covariates (World Bank GDP, population and unemployment; UNDP education; UCDP conflict; Henley visa openness; ND-GAIN climate vulnerability; modelled religion and integration indices). API responses are cached in `data/raw/cache/`. |
| 02 | `02_network_construction_centrality.py` | Builds one weighted directed graph per snapshot year and computes degree, strength, betweenness and weighted PageRank, plus network-level summaries. |
| 03 | `03_community_detection.py` | Runs Louvain, Leiden and Girvan–Newman, compares them (NMI/ARI), aligns communities across years with Hungarian matching, and measures temporal drift (Jaccard) and boundary nodes. |
| 04 | `04_regression_analysis.py` | Fits baseline, full and cultural-gravity OLS models for log immigrant and emigrant stock (pooled with country-clustered SEs, and year by year), with VIF diagnostics. |
| 05 | `05_export_finalization.py` | Writes `nodes_master.csv` (country-year) and `migration_full_flat.csv` (corridor-level), plus metadata, community labels and summary statistics. |
| 06 | `06_edge_weight_predictor.py` | Trains a gradient-boosting model of log corridor stock on all corridors, with a train/test split grouped by corridor. |

### Main outputs (`data/exports/`)

| File | Grain | Contents |
| --- | --- | --- |
| `nodes_master.csv` | country × year | Centrality, community, boundary score, socio-economic and cultural covariates |
| `migration_full_flat.csv` | corridor × year (top 500/year) | Corridor weight with origin and destination attributes and dyadic indicators |
| `network_edges.csv` | corridor × year (top 500/year) | Origin, destination, weight |
| `modularity_scores.csv` | year × algorithm | Modularity, community count, NMI/ARI agreement |
| `boundary_nodes.csv` | country | Community switches across the seven transitions |
| `regression_*.csv`, `vif_analysis.csv` | model × predictor | Coefficients, standard errors, fit statistics |
| `edge_model_metrics.json` | — | Held-out performance of the corridor model |

The complete corridor table (all ~9,000 corridors per year) is
`data/processed/migration_long.csv`.

## Corridor-size prediction API

```bash
python -m uvicorn api.predict:app --reload --port 8001
```

```bash
curl -X POST localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"origin_iso3": "IND", "dest_iso3": "USA", "year": 2020}'
```

Returns the predicted `log1p` stock, the predicted stock, and any input
features that were missing for that pair. You can also pass a precomputed
`features` object. `GET /health` reports whether the model is loaded. Interactive
docs are at `http://localhost:8001/docs`.

The model is a histogram gradient-boosting regressor over 31 origin,
destination and dyadic features. On corridors it never saw during training it
reaches **R² ≈ 0.73** on log stock (see `data/exports/edge_model_metrics.json`).
Because the split is grouped by corridor, this measures generalisation to new
country pairs rather than interpolation between years of known ones.

## Dashboard

See [dashboard/README.md](dashboard/README.md). In short: an interactive map
with corridor, centrality, community and religion layers; a country drill-down;
a displacement scenario simulator; distribution breakdowns; centrality,
modularity and regression views; and a two-country comparator.

## Data sources & licensing

| Data | Source | Notes |
| --- | --- | --- |
| Bilateral migrant stock | [UN DESA, International Migrant Stock 2024](https://www.un.org/development/desa/pd/content/international-migrant-stock) | Both sexes, mid-year, 1990–2024. 2025 extrapolated. |
| GDP per capita, population, unemployment | [World Bank WDI](https://data.worldbank.org/) (API) | CC BY 4.0 |
| Education index | UNDP Human Development Reports | |
| Conflict intensity | UCDP/PRIO Armed Conflict Dataset | |
| Visa openness | Henley Passport Index | |
| Climate vulnerability | ND-GAIN Country Index | |
| Religion profile | **Modelled** — `data/raw/generate_cultural_datasets.py` | Regional baselines with exact figures for about 30 countries, informed by Pew Research. |
| Integration & cultural-openness indices | **Modelled** — `data/raw/generate_cultural_datasets.py` | MIPEX-*style* scores from regional archetypes plus a linear time trend. Not official MIPEX data. |

> **Caveat.** The religion and integration indices are synthetic approximations
> built for this project, not measured data. Treat any result that depends on
> them (cultural-gravity regressions, the religion/integration breakdowns, the
> kinship and integration options in the simulator) as illustrative.
> Replacing them with sourced data is the most valuable contribution you could
> make; see [CONTRIBUTING.md](CONTRIBUTING.md).

The source code is released under the [MIT License](LICENSE). Third-party
datasets in `data/raw/` remain subject to their providers' terms; cite the
original sources when reusing them.

## Limitations

- Migrant **stock** is not **flow**: a corridor's value counts people living
  abroad at a point in time, not movements in a period.
- 2025 values are a one-year geometric extrapolation from 2024 and are marked
  `is_extrapolated` in `migration_long.csv`.
- Girvan–Newman runs on a reduced subgraph (edges covering 80% of total
  weight) for tractability, so its modularity is not comparable to
  Louvain/Leiden.
- The displacement simulator is a hand-tuned scenario tool, not a validated
  forecasting model.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as
described in [SECURITY.md](SECURITY.md).

## Citation

If you use this project, please cite it using [CITATION.cff](CITATION.cff)
(GitHub shows a "Cite this repository" button) and cite UN DESA for the
underlying migration data.

## License

[MIT](LICENSE) © Lazypilottt
