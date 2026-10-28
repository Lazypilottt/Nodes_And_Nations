export type Year = 1990 | 1995 | 2000 | 2005 | 2010 | 2015 | 2020 | 2025;
export const SNAPSHOT_YEARS: Year[] = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025];

export type View = 'overview' | 'simulator' | 'distributions' | 'topology' | 'comparator';
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
}
