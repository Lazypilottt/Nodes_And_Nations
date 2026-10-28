# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
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
