# Nodes & Nations — Dashboard

A Next.js app for exploring the migration network produced by the Python
pipeline in the repository root.

## Running

```bash
npm install
npm run dev          # http://localhost:3000
```

Production build:

```bash
npm run build && npm start
```

The app reads `../data/exports/` and `../data/processed/migration_long.csv`
at request time, so run `python scripts/build_pipeline.py` from the
repository root first if those files are missing.

## Views

| View | What it shows |
| --- | --- |
| **Overview** | World map with four layers (corridors, PageRank centrality, network communities, dominant religion). Click a country for its largest inbound/outbound corridors from the full dataset. |
| **Crisis simulator** | Gravity-routing model of where people displaced from a chosen country would go, by crisis type, severity and receiving-state policy. Shows projected arrivals and capacity strain (arrivals per 1,000 residents). |
| **Distributions** | Global migrant stock over time and its split by destination continent, income group, religion, integration-policy tier and network community. |
| **Network** | Centrality rankings, modularity over time, boundary states, and OLS regression results with clustered standard errors. |
| **Compare** | Two countries side by side, with the bilateral stock in both directions from 1990 to 2025. |

The year selector in the header (with play/pause) drives every view.

## API routes

| Route | Method | Description |
| --- | --- | --- |
| `/api/dataset` | GET | Nodes, top corridors, network summary, modularity, boundary nodes, regression results and per-year distributions. |
| `/api/country?iso3=IND&year=2020&limit=10` | GET | Largest inbound and outbound corridors for one country. |
| `/api/corridor?a=IND&b=ARE` | GET | Bilateral stock in both directions for every snapshot year. |
| `/api/simulate` | POST | Runs the displacement simulation. Body: `epicenter_iso3`, `scenario` (`conflict`, `climate`, `cultural_kinship`, `visa`, `economic`), `intensity` (10–100), `displaced_scale` (10k–20M), `border_policy` (`universal`, `kinship_first`, `cultural_kinship`, `integration_havens`, `strict_border`). |

Parsed CSVs are cached in memory and refreshed automatically when the files
change on disk.

## Project layout

```
components/   UI: WorldMap, Header, Overview, CrisisSimulator, Distributions, NetworkView, CorridorComparator, ui primitives
lib/          theme (colours, formatters), types, country geodata, chart defaults
lib/server/   server-only data loading and aggregation
pages/api/    API routes
```

## About the simulator

The simulator is an illustrative scenario tool, not a forecast. Destination
shares come from a hand-tuned gravity score combining distance decay,
existing diaspora, income, cultural/religious affinity and the chosen border
policy. Higher severity steepens distance decay, concentrating arrivals in
nearby states. Use it for exploring "what if" questions, not for planning.
