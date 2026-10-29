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
| **Overview** | Animated world map with four layers (corridors, PageRank centrality, network communities, dominant religion). Click a country for its largest corridors, then open its brief. |
| **Displacement planner** | Where people displaced from a chosen country would go, how many each host receives, and the pressure that puts on it (arrivals per 1,000 residents, and relative to the refugees it already hosts). The default engine is fitted on UNHCR data 2000–2024; close any border to see where people go instead. A back-test panel shows how well the model would have predicted seven past crises. A hand-tuned "what-if" engine remains for exploring crisis types and reception policies. |
| **Country brief** | One page per country: immigrant and emigrant trends, main corridors, percentile against all countries on nine measured factors, crisis exposure in both directions, and diaspora gaps. Copy a link, download a CSV, or print / save as PDF. |
| **Flows & gaps** | Directed chord diagram of migration between continents or 20 world regions, stacked regional trends, and the corridor-gap explorer (actual vs gravity-model expected stock, filterable by country). |
| **Drivers** | What high-immigration and high-emigration countries have in common (correlations with ~30 measured factors, raw or holding income and size fixed, plus a top-vs-bottom-fifth profile); the migration hump; PPML gravity estimates over time; and how much each data source improves corridor prediction. |
| **Network** | Centrality rankings, modularity over time, boundary states, and OLS regression results with clustered standard errors. |
| **Distributions** | Global migrant stock over time and its split by destination continent, income group, religion, integration-policy tier and network community. |
| **Compare** | Two countries side by side, with the bilateral stock in both directions from 1990 to 2025. |

The year selector in the header (with play/pause) drives every view. The URL
keeps the view, year and country (`/?view=brief&year=2020&country=DEU`), so
any screen can be bookmarked or shared.

## Theme and motion

Light, dark and system themes (switch in the header; the choice is remembered).
Neutral colours are CSS variables (`styles/globals.css`) mapped onto Tailwind's
`zinc` and `white`, so components need no `dark:` variants; charts and the map
read colours from `lib/themeContext.tsx`. Motion uses framer-motion (view
transitions, sliding tab and year indicators, count-up numbers, corridors that
draw in and flow) and is reduced automatically when the operating system asks
for reduced motion.

## API routes

| Route | Method | Description |
| --- | --- | --- |
| `/api/dataset` | GET | Nodes, top corridors, network summary, modularity, boundary nodes, regression results, per-year distributions, region-to-region flow matrices, the driver analysis and the displacement back-tests. |
| `/api/country?iso3=IND&year=2020&limit=10` | GET | Largest inbound and outbound corridors for one country. |
| `/api/corridor?a=IND&b=ARE` | GET | Bilateral stock in both directions for every snapshot year. |
| `/api/brief?iso3=DEU` | GET | Latest corridors, diaspora and immigration gaps, forecast destinations if the country were the epicentre of a crisis, and current crises most likely to reach it. |
| `/api/potential?iso3=IND&role=any&limit=15` | GET | Corridors at least 4× smaller (`under`) or larger (`over`) than the gravity model expects, plus a scatter sample. `iso3` and `role` (`origin`, `dest`, `any`) are optional. |
| `/api/simulate` | POST | Displacement forecast. Body: `engine` (`model` default, or `scenario`), `epicenter_iso3`, `displaced_scale` (10k–20M), `closed` (array of ISO3 codes whose borders are closed). For `scenario` also: `scenario` (`conflict`, `climate`, `cultural_kinship`, `visa`, `economic`), `intensity` (10–100), `border_policy` (`universal`, `kinship_first`, `cultural_kinship`, `integration_havens`, `strict_border`). |

Parsed CSVs are cached in memory and refreshed automatically when the files
change on disk.

## Project layout

```
components/   UI: WorldMap, Header, Overview, DisplacementPlanner, CountryBrief, Flows, Drivers,
              NetworkView, Distributions, CorridorComparator, ui primitives
lib/          theme (colours, formatters), theme context, types, country geodata, chart defaults
lib/server/   server-only data loading and aggregation
pages/api/    API routes
```

## About the displacement forecast

The default engine reads `data/exports/displacement_shares.csv`, written by
`notebooks/08_displacement_forecast.py`: the average of two destination-choice
models fitted on UNHCR refugee, asylum-seeker and protection data, with
distance, shared borders, language, colonial ties, the existing diaspora, the
host's income, size, stability, conflict and current refugee load, and (in the
second model) where the origin's refugees were last year. Closing a border
removes that country and redistributes its share. Pressure levels are arrivals
per 1,000 residents: Low (< 1.2), Moderate, High (≥ 4), Critical (≥ 12).

The "what-if" engine is a hand-tuned gravity score for exploring crisis types
and reception policies. Land neighbours, common language and colonial ties come
from CEPII GeoDist and are weighted by the PPML estimates in
`notebooks/07_migration_drivers.py`, but the scenario weights are not
validated. Use it for "what if" questions, not for planning.
