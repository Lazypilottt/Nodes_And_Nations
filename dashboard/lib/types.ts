export type Year = 1990 | 1995 | 2000 | 2005 | 2010 | 2015 | 2020 | 2025;
export const SNAPSHOT_YEARS: Year[] = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025];

export type View = 'overview' | 'planner' | 'brief' | 'flows' | 'drivers' | 'topology' | 'distributions' | 'comparator';
export const VIEWS: View[] = ['overview', 'planner', 'brief', 'flows', 'drivers', 'topology', 'distributions', 'comparator'];
export type MapMode = 'corridors' | 'centrality' | 'communities' | 'religion' | 'simulation';

export interface NodeRow {
  iso3: string;
  year: number;
  country_name: string | null;
  continent: string | null;
  un_region: string | null;
  income_group: string | null;
  in_strength: number | null;
  out_strength: number | null;
  pagerank: number | null;
  betweenness_centrality: number | null;
  louvain_community: number | null;
  community_label: string | null;
  boundary_score: number | null;
  is_boundary_node: boolean | null;
  gdp_per_capita: number | null;
  population: number | null;
  unemployment: number | null;
  conflict_intensity: number | null;
  climate_vulnerability: number | null;
  visa_openness_index: number | null;
  dominant_religion: string | null;
  integration_policy_index: number | null;
  cultural_openness_index: number | null;
  gdp_pc_ppp: number | null;
  political_stability: number | null;
  urban_pct: number | null;
  youth_share: number | null;
  internet_users: number | null;
  remittances_gdp: number | null;
  life_expectancy: number | null;
  homicide_rate: number | null;
  refugees_hosted_per_1k: number | null;
  refugees_abroad_per_1k: number | null;
}

export interface EdgeRow {
  origin_iso3: string;
  dest_iso3: string;
  year: number;
  weight: number;
  same_continent: number | null;
}

export interface Share {
  label: string;
  value: number;
}

export interface YearDistribution {
  total_stock: number;
  intra_continental_share: number;
  high_income_share: number;
  continent: Share[];
  income: Share[];
  religion: Share[];
  mipex: Share[];
  community: Share[];
}

export interface DriverCorrelation {
  outcome: 'immigrant_share' | 'emigrant_rate';
  factor: string;
  label: string;
  category: string;
  note: string | null;
  n: number;
  rho: number;
  p_value: number;
  rho_partial: number;
  p_partial: number;
  ci_low: number;
  ci_high: number;
}

export interface DriverProfile {
  outcome: 'immigrant_share' | 'emigrant_rate';
  factor: string;
  label: string;
  category: string;
  median_top: number | null;
  median_bottom: number | null;
  median_all: number | null;
}

export interface CountryProfile {
  iso3: string;
  country_name: string | null;
  continent: string | null;
  year: number;
  immigrant_share: number | null;
  emigrant_rate: number | null;
  gdp_pc_ppp: number | null;
}

export interface GravityTerm {
  estimator: 'ppml' | 'ols_log';
  year: number;
  term: string;
  label: string;
  coefficient: number;
  std_error: number;
  p_value: number;
  ci_lower_95: number;
  ci_upper_95: number;
  pct_effect: number | null;
}

export interface Drivers {
  correlations: DriverCorrelation[];
  profile: DriverProfile[];
  countries: CountryProfile[];
  hump: Record<string, any>[];
  hump_curve: { kind: 'fit' | 'bin_median'; gdp_pc_ppp: number; emigrant_rate: number }[];
  gravity: GravityTerm[];
  gravity_fit: Record<string, any>[];
  model_ablation: { feature_set: string; n_features: number; r2: number; rmse_log: number }[];
}

export interface RegionFlows {
  labels: string[];
  matrix: number[][];
}

export interface BacktestRow {
  origin_iso3: string;
  crisis: string;
  onset: number;
  eval_year: number;
  displaced_eval_year: number;
  method: 'model' | 'diaspora_only' | 'distance_only';
  overlap: number;
  top5_hits: number;
  train_years: string;
}

export interface BacktestShare {
  origin_iso3: string;
  crisis: string;
  dest_iso3: string;
  actual_share: number;
  model_share: number;
  diaspora_share: number;
}

export interface Displacement {
  backtest: BacktestRow[];
  backtest_shares: BacktestShare[];
  model: Record<string, any>[];
  meta: Record<string, any> | null;
}

export interface Dataset {
  nodes: NodeRow[];
  edges: EdgeRow[];
  network_summary: Record<string, any>[];
  modularity_scores: Record<string, any>[];
  boundary_nodes: Record<string, any>[];
  regression_coefficients: Record<string, any>[];
  regression_comparison: Record<string, any>[];
  distributions: Record<number, YearDistribution>;
  model_metrics: Record<string, any> | null;
  drivers: Drivers;
  region_flows: Record<number, { continent: RegionFlows; region: RegionFlows }>;
  displacement: Displacement;
}
