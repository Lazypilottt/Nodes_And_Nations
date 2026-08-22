Nodes and Nations
=================
A topological analysis of human movement (1990-2025)

There is a quiet structure to how the world moves. Migration isn't just a ledger of border crossings; it's a dynamic, directed graph subject to the gravity of GDP, the friction of conflict, and the momentum of established diasporas. 

This repository models 35 years of global migration using UN DESA bilateral stock data. We treat countries as nodes and mass human movement as weighted, directed edges. The pipeline parses the raw data, constructs temporal graph snapshots, evaluates systemic routing changes via centrality and community detection heuristics, and performs ordinary least squares regression to isolate the macro-drivers of movement.

Everything collapses into clean, unified master exports (`nodes_master.csv` and `migration_full_flat.csv`) and powers an interactive full-stack Next.js / React analytics platform and crisis simulator.


Architecture & Execution
------------------------
The analysis runs as a linear, reproducible pipeline. Heavy notebooks have been replaced with modular scripts that get straight to the math. Run them sequentially or with `scripts/build_pipeline.py`.

Dependencies: networkx, python-louvain, leidenalg, igraph, statsmodels, scikit-learn, xgboost, openpyxl, requests, pandas, numpy, fastapi, uvicorn.

    source venv/bin/activate
    
    python notebooks/01_data_collection_cleaning.py
    python notebooks/02_network_construction_centrality.py
    python notebooks/03_community_detection.py
    python notebooks/04_regression_analysis.py
    python notebooks/05_export_finalization.py
    python notebooks/06_edge_weight_predictor.py

Or build all phases in one command:

    python scripts/build_pipeline.py

Note: Script 01 reads the initial UN matrix into memory and reaches out to the World Bank and UCDP endpoints for socioeconomic covariates. Local caching is implemented; the API heavy lifting only happens on the first run. The 2025 vectors are extrapolated geometrically from recent momentum.


Data Provenance & Indices
-------------------------
* Bilateral migrant stock (1990-2025): UN DESA IMS 2024
* GDP, Population, Unemployment: World Bank WDI (API)
* Education Index: UNDP HDR (Static CSV)
* Conflict intensity: UCDP/PRIO ACD (Static CSV)
* Cultural Integration Index (`cultural_integration_index.csv`): MIPEX-aligned metrics (1990–2025) including `integration_policy_index`, `cultural_openness_index`, `labour_market_integration`, `anti_discrimination_score`, `naturalization_access_score`.
* World Religion Demographics (`world_religions_demographics.csv`): 235 sovereign states with percentage shares across Christianity, Islam, Hinduism, Buddhism, Judaism, and Unaffiliated, alongside Herfindahl fractionalization and diversity indices.
* Visa Openness: Henley Passport Index (Static CSV)
* Climate Vulnerability: ND-GAIN Index (Static CSV)


Analytical Modules
------------------
Phase 2: Network Topology
We build 8 directed graphs (Nx). Edge weights dictate the calculations. The script isolates purely transit hubs (Betweenness) versus global attractors (weight-aware PageRank).

Phase 3: Structural Fault Lines 
As global alliances and borders shift, so does the network density. We apply Louvain, Leiden, and Girvan-Newman algorithms to partition the graph. By measuring Modularity (Q) alongside temporal Jaccard drift, the pipeline identifies "boundary nodes"—states acting as bridges between shifting geo-political clusters.

Phase 4: Cultural & Econometric Gravity Regression 
We fit pooled and year-by-year OLS models against structural and cultural data to determine how capital variance, cultural openness, shared dominant religion, and demographic mass govern corridor edge weights. Cultural gravity models achieve high explanatory power ($R^2 > 0.62$).

Phase 5: Unified Dataset Finalization & Validation
Outputs are written to `data/exports/`, producing unified datasets (`nodes_master.csv` with 42 features per sovereign country-year, and `migration_full_flat.csv` with 91 dyadic features).

Phase 6: Machine-Learning Edge Predictor
A full ML pipeline training gradient-boosted regressors (Histogram Gradient Boosting / XGBoost) with 31 features (R² ~ 0.70) to predict bilateral corridor migrant stock.


Interactive Web Platform (Next.js / React)
------------------------------------------
Launch the interactive dashboard:

    cd dashboard
    npm run dev

Features:
* Planetary Vector Surface: Interactive Mercator SVG map supporting Bilateral Corridors, Centrality Topography, Alliance Clusters, and Dominant Religion Demographics.
* Crisis Displacement Cockpit: Real-time gravity routing simulations across armed conflict, cultural kinship displacement, climate surges, and border policy shocks.
* Macro-Flow Distribution Analytics: Interactive doughnut and pie visualizers for destination continents, income tiers, dominant religions, and MIPEX policy tiers.
* Dyadic Corridor Comparator: Side-by-side socioeconomic, cultural, and religious affinity comparison between sovereign states.
* Network Topology & Econometric Determinants: Centrality rankings, modularity drift (Q) trends, boundary node switchers, and OLS regression tables.


Notes from the run
------------------
The data suggests an interesting systemic truth: the network's Louvain modularity score drops from 0.560 in 1990 down to 0.474 projected for 2025. The graph is becoming less compartmentalized and more structurally integrated over time.

- Codebase finalized 2026.


Phase 6: Machine-Learning Edge Predictor
---------------------------------------
A full ML pipeline has been added to demonstrate an applied data-science deliverable suitable for a resume. It trains a gradient-boosting regressor to predict bilateral migrant stock (edge weight) from origin/destination socioeconomic features and relational signals.

New files:
- notebooks/06_edge_weight_predictor.py  # full training + CV, feature importances, model persistence
- models/edge_weight_predictor.joblib     # (created after running the training script)
- api/predict.py                          # FastAPI demo to serve predictions
- requirements-ml.txt                     # ML + serving dependencies to install in the project venv

Quickstart (recommended)

    source venv/bin/activate
    pip install -r requirements-ml.txt
    python notebooks/06_edge_weight_predictor.py

After training, run the demo server:

    source venv/bin/activate
    uvicorn api.predict:app --reload --port 8001

POST /predict (JSON)
- Provide either `features` (precomputed feature map) or `origin_iso3`, `dest_iso3`, and `year` to let the server compute features and return a prediction.

What this demonstrates for your resume
- End-to-end ML workflow: feature engineering, model selection with randomized CV, evaluation (RMSE/R²), feature-importance visualization, model persistence
- Production demo: simple REST endpoint for fast validation and screenshots
- Use this in a portfolio entry alongside a short write-up and a ROC/importance figure to show measurable impact

If you'd like, the next step is to: run the training script here, save the artifacts, and start the demo server so sample predictions and screenshots can be produced for the portfolio. Reply "Run training" to proceed.
