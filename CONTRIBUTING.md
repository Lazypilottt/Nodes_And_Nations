# Contributing to Nodes & Nations

Thanks for your interest in improving the project. This guide explains how to
set up a development environment and what we look for in a pull request.

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** using the bug report issue template. Include the command you
  ran, the full error output and your OS / Python / Node versions.
- **Propose a feature or methodological change** with the feature request
  template. For changes to the analysis (new covariates, different community
  algorithms, model changes), explain the motivation and expected effect on
  the results.
- **Improve the data** — especially replacing the modelled religion and
  integration indices with sourced data (see README → Data sources).
- **Fix documentation**, typos and examples.

## Development setup

### Python pipeline and API

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_pipeline.py  # regenerate data/processed and data/exports
python -m pytest -q tests         # unit tests
python -m uvicorn api.predict:app --reload --port 8001
```

### Dashboard

```bash
cd dashboard
npm install
npm run dev        # http://localhost:3000
npm run typecheck  # must pass before opening a PR
npm run build      # must pass before opening a PR
```

The dashboard reads `../data/exports` and `../data/processed` directly, so
regenerate those with the pipeline if you change how they are produced.

## Pull request checklist

- [ ] The change is focused; unrelated refactors go in a separate PR.
- [ ] Python files compile (`python -m py_compile <file>`), `python -m pytest -q tests`
      passes, and any pipeline step you touched runs end to end.
- [ ] `npm run typecheck` and `npm run build` pass in `dashboard/`.
- [ ] If outputs in `data/exports/` changed, the PR description summarises how
      (e.g. "modularity Q for 2025 moves from 0.589 to 0.571").
- [ ] Documentation (README, docstrings) is updated for any user-facing change.
- [ ] An entry is added under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).

## Style

- **Python**: PEP 8, 4-space indentation, type hints on new functions,
  docstrings explaining *why* a step exists. Keep each notebook script
  runnable on its own.
- **TypeScript/React**: functional components, colours and formatters from
  `dashboard/lib/theme.ts` rather than hard-coded values, Tailwind utility
  classes for styling.
- **Commits**: short imperative subject line (`Fix Leiden modularity weighting`),
  with a body explaining the reasoning when it isn't obvious.

## Data and ethics

This project deals with displacement and identity-related attributes
(religion, origin). When contributing:

- Cite the source and license of any new dataset in the README.
- Clearly label modelled, synthetic or extrapolated values as such — in the
  data, the README and the dashboard.
- Avoid framing that treats migrants or specific countries as threats.
