Nodes and Nations
=================
A topological analysis of human movement (1990-2025)

There is a quiet structure to how the world moves. Migration isn't just a ledger of border crossings; it's a dynamic, directed graph subject to the gravity of GDP, the friction of conflict, and the momentum of established diasporas. 

This repository models 35 years of global migration using UN DESA bilateral stock data. We treat countries as nodes and mass human movement as weighted, directed edges. The pipeline parses the raw data, constructs temporal graph snapshots, evaluates systemic routing changes via centrality and community detection heuristics, and performs ordinary least squares regression to isolate the macro-drivers of movement.

Everything ultimately collapses into a relational schema engineered for Power BI.


Architecture & Execution
------------------------
The analysis runs as a linear pipeline. Heavy notebooks have been avoided in favor of functional scripts that get straight to the math. Run them sequentially.

Dependencies: networkx, python-louvain, leidenalg, igraph, statsmodels, scikit-learn, openpyxl, requests, pandas, numpy.

    source venv/bin/activate
    
    python notebooks/01_data_collection_cleaning.py
    python notebooks/02_network_construction_centrality.py
    python notebooks/03_community_detection.py
    python notebooks/04_regression_analysis.py
    python notebooks/05_powerbi_exports.py

Note: Script 01 reads the initial UN matrix into memory and reaches out to the World Bank and UCDP endpoints for socioeconomic covariates. Local caching is implemented; the API heavy lifting only happens on the first run. The 2025 vectors are extrapolated geometrically from recent momentum.


Data Provenance
---------------
* Bilateral migrant stock (1990-2020): UN DESA IMS 2024
* GDP, Population, Unemployment: World Bank WDI (API)
* Education Index: UNDP HDR (Static CSV)
* Conflict intensity: UCDP/PRIO ACD (Static CSV)

Optional manual vectors:
To expand the regression matrix, you can place `henley_passport_index.csv` and `ndgain_country_index.csv` in the `data/raw/` directory. The ingestion scripts will automatically merge them into the panel sequence if present.


Analytical Modules
------------------
Phase 2: Network Topology
We build 8 directed graphs (Nx). Edge weights dictate the calculations. The script isolates purely transit hubs (Betweenness) versus global attractors (weight-aware PageRank).

Phase 3: Structural Fault Lines 
As global alliances and borders shift, so does the network density. We apply Louvain, Leiden, and Girvan-Newman algorithms to partition the graph. By measuring Modularity (Q) alongside temporal Jaccard drift, the pipeline identifies "boundary nodes"—states acting as bridges between shifting geo-political clusters.

Phase 4: Regression 
We fit pooled OLS models against the structural data to determine if capital variance, demographic mass, or localized instability form the root causality of the edge weights modeled in Phase 2. A baseline three-factor model consistently explains over 60% of the variance in edge weight.

Phase 5: Aggregation
Network scale limits visualization rendering. The terminal script limits the graph to the top 500 edges by weight per year. All outputs are written to `data/exports/`, ready for direct insertion into a star-schema dashboard. See `powerbi_manifest.txt` for relationship mapping.


Notes from the run
------------------
The data suggests an interesting systemic truth: the network's Louvain modularity score drops from 0.560 in 1990 down to 0.474 projected for 2025. The graph is becoming less compartmentalized and more structurally integrated over time.

- Codebase finalized 2026.