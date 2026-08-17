"""
build_pipeline.py
Run the full Nodes_and_Nations ETL + training pipeline end-to-end.

Steps (default):
  1. Phase 1 data collection & cleaning  -> notebooks/01_data_collection_cleaning.py
  2. Phase 2 network construction         -> notebooks/02_network_construction_centrality.py
  3. Phase 3 train edge-weight predictor  -> notebooks/06_edge_weight_predictor.py

Usage:
    python scripts/build_pipeline.py         # run full pipeline
    python scripts/build_pipeline.py --skip-train
    python scripts/build_pipeline.py --skip-graph

The script sets PYTHONPATH so the `data` package is importable when running notebooks as scripts.
It stops on the first failing step and prints helpful diagnostics.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB01 = ROOT / 'notebooks' / '01_data_collection_cleaning.py'
NB02 = ROOT / 'notebooks' / '02_network_construction_centrality.py'
NB03 = ROOT / 'notebooks' / '03_community_detection.py'
NB04 = ROOT / 'notebooks' / '04_regression_analysis.py'
NB05 = ROOT / 'notebooks' / '05_powerbi_exports.py'
NB06 = ROOT / 'notebooks' / '06_edge_weight_predictor.py'

DEFAULT_ENV = os.environ.copy()
# Ensure project root is on PYTHONPATH so 'data' package can be imported by notebooks
DEFAULT_ENV['PYTHONPATH'] = str(ROOT)


def run_step(cmd, env=None, cwd=None):
    print('\n' + '='*70)
    print(f"Running: {' '.join(cmd)}")
    print('='*70)
    try:
        subprocess.run(cmd, check=True, env=env or DEFAULT_ENV, cwd=cwd or str(ROOT))
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        return False


def summarize_outputs():
    """Print short summary of key produced artifacts."""
    exports = ROOT / 'data' / 'exports'
    processed = ROOT / 'data' / 'processed'
    models = ROOT / 'models'

    print('\n' + '#' * 60)
    print('Pipeline summary (files present)')
    print('#' * 60)

    def present(p):
        return p.exists()

    items = [
        (processed / 'migration_long.csv', 'migration_long.csv'),
        (processed / 'factors_panel.csv', 'factors_panel.csv'),
        (processed / 'factors_panel_enriched.csv', 'factors_panel_enriched.csv'),
        (exports / 'network_edges.csv', 'network_edges.csv'),
        (exports / 'centrality_metrics.csv', 'centrality_metrics.csv'),
        (exports / 'nodes_master.csv', 'nodes_master.csv'),
        (exports / 'migration_full_flat.csv', 'migration_full_flat.csv'),
        (models / 'edge_weight_predictor.joblib', 'models/edge_weight_predictor.joblib'),
        (exports / 'edge_model_metrics.json', 'edge_model_metrics.json'),
    ]

    for path, label in items:
        print(f"{label:35} : {'YES' if present(path) else 'NO '}")

    print('#' * 60 + '\n')


def main():
    parser = argparse.ArgumentParser(description='Run notebooks in notebooks/ in numeric order. Use --skip-train / --skip-graph to opt-out.')
    parser.add_argument('--skip-train', action='store_true', help='Skip notebooks whose filename starts with 06 (training)')
    parser.add_argument('--skip-graph', action='store_true', help='Skip notebooks whose filename starts with 02 (graph)')
    parser.add_argument('--notebooks-dir', default=str(ROOT / 'notebooks'), help='Path to notebooks directory')
    args = parser.parse_args()

    nb_dir = Path(args.notebooks_dir)
    if not nb_dir.exists():
        print(f"Notebooks directory not found: {nb_dir}")
        sys.exit(2)

    # Discover notebooks with a leading numeric prefix (e.g. 01_, 02_..)
    notebooks = []
    for p in sorted(nb_dir.iterdir()):
        if p.is_file() and p.suffix == '.py':
            name = p.name
            # accept files that start with a 2-digit numeric prefix
            if len(name) >= 2 and name[:2].isdigit():
                notebooks.append(p)

    if not notebooks:
        print('No numbered notebooks found in', nb_dir)
        sys.exit(2)

    print('Planned run order:')
    for p in notebooks:
        print('  ', p.name)

    for nb in notebooks:
        # Respect skip flags by detecting filename prefixes
        prefix = nb.name[:2]
        if args.skip_graph and prefix == '02':
            print(f"Skipping {nb.name} due to --skip-graph")
            continue
        if args.skip_train and prefix == '06':
            print(f"Skipping {nb.name} due to --skip-train")
            continue

        ok = run_step([sys.executable, str(nb)])
        if not ok:
            print(f"Notebook {nb.name} failed — aborting pipeline.")
            sys.exit(1)

    summarize_outputs()
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
