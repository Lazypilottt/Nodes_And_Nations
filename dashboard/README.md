# Nodes & Nations — Planetary Migration & Crisis Simulator Dashboard

A high-performance interactive web application and geopolitical shock simulation platform for global human migration topology (1990–2025).

---

## 🌟 Key Features

### 1. 🌍 Interactive Planetary Vector Map & Flight Beams
- **Canvas Particle Flight Arcs**: Animated luminous photon particles traveling along great-circle migration corridors between 235 sovereign nations.
- **Multi-Mode Overlays**:
  - `✈️ Migration Corridors`: Visualizes top bilateral human movement corridors with dynamic width and directional arcs.
  - `💠 Centrality Nodes`: Renders glowing node sizes proportional to PageRank attractor mass and Betweenness transit bridge scores.
  - `🧩 Alliance Clusters`: Colors nations dynamically by their Louvain / Leiden community alliance memberships across time.
  - `⚔️ Crisis Radar Vectors`: Displays pulsing red shockwaves radiating from crisis epicenters with recipient heat rays.
- **Interactive Controls**: Zoom (+/−), Pan dragging, Reset center, Corridor density filters, and country hover tooltips.

### 2. ⚔️ War Room & Geopolitical Shock Simulator
- **Gravity-Routing Displacement Engine**:
  - Mathematical model factoring historical diaspora kinship, geographic contiguity friction ($1/\text{dist}^{0.65}$), economic gravity ($\text{GDP}^{0.28}$), and border absorption policies.
- **Preset Real-World Crises**:
  - 🇺🇦 Ukraine Crisis (2022) — 6.5M displaced
  - 🇸🇾 Syrian Conflict (2015) — 5.2M displaced
  - 🇸🇩 Sudan Civil War (2023) — 3.4M displaced
  - 🇦🇫 Afghanistan Exodus (2021) — 2.8M displaced
  - 🌊 Bangladesh Coastal Climate Surge — 2.5M displaced
  - 🇻🇪 Venezuela Economic Collapse — 4.2M displaced
- **Real-Time Analytics Deck**:
  - Interactive **Donut & Pie Charts** for continent, UN region, and income tier displacement shares.
  - **Top 10 Recipient States Table** with estimated influx numbers and Local Capacity Stress ratings (Critical, High, Moderate, Manageable).

### 3. 📊 Interactive Macro-Flow Analytics (Pie & Donut Charts)
- **Destination Continents** (Donut with interactive legend and percentage calculation).
- **Income Tiers** (High income vs Middle vs Low income destination shares).
- **Community Cluster Mass** (Share of global migration absorbed by each geopolitical alliance).
- **Top 8 Corridors Bar Chart**.

### 4. 🌐 Centralities, Modularity Drift, & Econometric Regression
- **Modularity Trend ($Q$) Line Chart**: Visualizes the $20.5\%$ decline in modularity from $0.596$ (1990) down to $0.474$ (2025), showing systemic global integration.
- **Centrality Leaderboards**: Sort by PageRank, Betweenness, In-Strength, or Out-Strength.
- **Boundary States Matrix**: Identifies geopolitical switchers (e.g. Turkey, Cyprus, Ukraine, Kazakhstan).
- **Econometric Determinants Table**: OLS regression models with coefficients, $t$-stats, and $p$-values ($R^2 \approx 0.61 - 0.75$).

### 5. 🔀 Bilateral Corridor Comparator
- Side-by-side comparative inspection of any two sovereign nations (GDP per capita, population, visa freedom, climate vulnerability, and historical flow trajectory from 1990 to 2025).

### 6. ⏱️ Time-Machine Timeline Player
- Smooth scrubber player spanning 1990 $\to$ 2025 with Play, Pause, and step controls that dynamically recomputes all topological networks.

---

## 🚀 Quick Start

1. Start dev server:
   ```bash
   cd dashboard
   npm run dev
   ```
2. Open [http://localhost:3000](http://localhost:3000) in your browser.
