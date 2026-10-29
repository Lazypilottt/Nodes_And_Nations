# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Displacement planner** (`notebooks/08_displacement_forecast.py`): a
  destination-choice model fitted on UNHCR origin × host data 2000–2024
  (two PPML models with origin × year fixed effects, averaged), back-tested on
  seven crises (Syria, South Sudan, Venezuela, Myanmar, Afghanistan, Ukraine,
  Sudan). It places 52% of displaced people correctly on average, against 45%
  for diaspora-only and 23% for nearest-country baselines. The dashboard
  planner uses it by default, supports closing borders, and shows pressure on
  each host and the back-test evidence.
- **Corridor gaps** (`notebooks/09_corridor_potential.py`): corridors at least
  4× larger or smaller than a PPML gravity model expects, with an explorer in
  the new Flows & gaps view and `/api/potential`.
- **Country briefs**: a one-page profile per country with peer percentiles,
  crisis exposure and diaspora gaps; copy link, CSV download, print / PDF;
  `/api/brief`.
- **Flows view**: directed chord diagram of migration between continents and
  20 regions, and stacked regional trends.
- **Dark mode** (light / dark / system), shareable URLs for view, year and
  country, and animation throughout (view transitions, sliding indicators,
  count-up numbers, flowing corridors, pulsing crisis epicentre), reduced when
  the system asks for less motion.
- `ppml_fe` supports a single set of fixed effects and returns fitted values.
- Figures 18–19.
- **CEPII GeoDist geography** (`data/raw/build_geography.py`): distance,
  shared border, common language, colonial ties and "were one country" for
  every country pair, mapped to current ISO codes, plus landlocked status and
  land area.
- **23 more World Bank indicators** (GDP per capita PPP, growth, inflation,
  remittances, trade, resource rents, farm employment, female labour
  participation, Gini, density, urbanisation, age structure, fertility, life
  expectancy, internet and mobile use, tourism, homicide) and the four
  **Worldwide Governance Indicators**, cached one file per indicator.
- **UNHCR displacement**: refugees and asylum seekers by origin and by host,
  and internally displaced people, per 1,000 residents.
- **Migration drivers** (`notebooks/07_migration_drivers.py`): correlates of
  immigration and emigration rates with bootstrap CIs, top/bottom-fifth
  profiles, a migration-hump test, and PPML structural gravity by year.
- **PPML estimator** (`data/gravity.py`) with origin and destination fixed
  effects and pair-clustered standard errors, plus an OLS-on-logs comparison.
- Corridor model: ablation table by feature group, a 1990–2015 → 2020
  forecast check, and network features (each endpoint's diaspora in the
  previous snapshot, excluding the corridor itself).
- Development regression model using measured factors only.
- Dashboard **Drivers** view; simulator results list each destination's ties.
- Figures 13–17; unit tests in `tests/`, run in CI.

### Changed
- Corridor model: held-out R² 0.72 → 0.83. The served model no longer uses
  the modelled religion and integration indices.
- Snapshot values use the exact year when published, otherwise the mean
  within ±2 years. 2025 uses 2021–2024 data instead of extrapolating 2020.
- The crisis simulator became the displacement planner; its hand-tuned engine
  remains as the "what-if" option. Neighbours include real land borders, and common language
  and colonial ties are weighted by the PPML estimates. The "Religious kin"
  policy is now "Cultural kin".
- `migration_full_flat.csv` gains the CEPII pair attributes; the new country
  indicators live in `nodes_master.csv` only.

### Fixed
- World Bank downloads that failed part-way were silently kept, which left
  GDP per capita 33% complete. Downloads now retry and fail loudly.
- **Migrant stock used female-only figures.** The UN DESA parser kept the last
  column for each year, which in Table 1 is the female block. It now reads the
  both-sexes totals. All downstream exports were regenerated.
- 2025 values are now extrapolated from the observed 2024 release (2020→2024
  trend) instead of the older 2015→2020 trend.
- Leiden modularity was reported unweighted (≈0.10); it now uses the same
  weighted definition as Louvain (≈0.55).
- Boundary nodes compared arbitrary community IDs across years, flagging every
  country. Communities are now matched across years (Hungarian algorithm)
  before counting switches; community IDs are stable over time as a result.
- `religious_diversity_index` (a linear rescale of fractionalisation) was
  removed from the regression, eliminating perfect collinearity.
- Pooled regressions now use standard errors clustered by country.
- Duplicate `country_name_x` / `country_name_y` columns in the master exports.
- Edge-weight predictor: fixed a syntax error that prevented training; now
  trains on all corridors with a corridor-grouped train/test split.
- Prediction API: crashed on Python 3.9 (`str | None`), and built features with
  different names from training so every ISO-based request failed. Training and
  serving now share `data/features.py`.
- Dashboard: country names never rendered; simulator "telemetry" showed
  hard-coded values; the severity slider had no effect; regression table marked
  every coefficient as significant; 38 corridors were drawn to (0, 0) because
  32 territories lacked coordinates; ⌘/Ctrl-scroll zoomed the whole page.

### Changed
- Dashboard redesigned with a minimal neutral theme; distribution charts use
  the full corridor table instead of the top-500 export.
- Upgraded Next.js 13.5 (end of life) to 14.2.
- API routes cache parsed CSVs in memory instead of re-reading them per request.
- Consolidated Python dependencies into `requirements.txt`; the development
  environment now targets Python 3.11 (3.9 is end-of-life).
- Moved the project report to `docs/`.

### Added
- `scripts/make_figures.py` and twelve regenerated figures in `Graphs/`
  (replacing the old dashboard screenshots, which showed the pre-fix data).
- `edge_feature_importance.csv` export from the corridor model.
- Country drill-down (`/api/country`) and corridor time series (`/api/corridor`).
- Prediction API `GET /health` endpoint.
- LICENSE, Code of Conduct, contributing guide, security policy, citation file,
  issue/PR templates and CI workflow.

### Removed
- `venv/`, `__pycache__/` and other generated files from version control.
- Unused dashboard components and API routes.
