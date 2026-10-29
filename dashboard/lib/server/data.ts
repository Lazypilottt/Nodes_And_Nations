import fs from 'fs';
import path from 'path';
import { DATA_ROOT, Row, readDataCSV } from '../csv';
import { INCOME_ORDER, MIPEX_TIERS, mipexTier } from '../theme';
import type { Dataset, Displacement, Drivers, EdgeRow, NodeRow, RegionFlows, Share, YearDistribution } from '../types';

const NODE_FIELDS: (keyof NodeRow)[] = [
  'iso3', 'year', 'country_name', 'continent', 'un_region', 'income_group',
  'in_strength', 'out_strength', 'pagerank', 'betweenness_centrality',
  'louvain_community', 'community_label', 'boundary_score', 'is_boundary_node',
  'gdp_per_capita', 'population', 'unemployment', 'conflict_intensity',
  'climate_vulnerability', 'visa_openness_index', 'dominant_religion',
  'integration_policy_index', 'cultural_openness_index',
  'gdp_pc_ppp', 'political_stability', 'urban_pct', 'youth_share', 'internet_users', 'remittances_gdp',
  'life_expectancy', 'homicide_rate', 'refugees_hosted_per_1k', 'refugees_abroad_per_1k',
];

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (v === null || v === undefined ? null : String(v));

export interface Migration {
  origin: string;
  dest: string;
  year: number;
  stock: number;
}

let migrationCache: { rows: Row[]; data: Migration[] } | null = null;

/** Full bilateral stock table (all corridors, all years), excluding self-loops. */
export function migrationLong(): Migration[] {
  const rows = readDataCSV('processed/migration_long.csv');
  if (migrationCache && migrationCache.rows === rows) return migrationCache.data;
  const data: Migration[] = [];
  for (const r of rows) {
    const origin = str(r.origin_iso3);
    const dest = str(r.dest_iso3);
    const stock = num(r.migrant_stock);
    if (!origin || !dest || origin === dest || !stock || stock <= 0) continue;
    data.push({ origin, dest, year: Number(r.year), stock });
  }
  migrationCache = { rows, data };
  return data;
}

export function countryMeta(): Map<string, Row> {
  return new Map(readDataCSV('exports/country_metadata.csv').map((r) => [String(r.iso3), r]));
}

export function nodes(): NodeRow[] {
  return readDataCSV('exports/nodes_master.csv').map((r) => {
    const out: Record<string, unknown> = {};
    for (const f of NODE_FIELDS) out[f] = r[f] ?? null;
    out.country_name = r.country_name ?? r.country_name_x ?? null;
    return out as unknown as NodeRow;
  });
}

function toShares(acc: Map<string, number>, order?: string[]): Share[] {
  const shares = Array.from(acc, ([label, value]) => ({ label, value }));
  if (order) {
    const rank = (l: string) => (order.indexOf(l) === -1 ? order.length : order.indexOf(l));
    return shares.sort((a, b) => rank(a.label) - rank(b.label) || b.value - a.value);
  }
  return shares.sort((a, b) => b.value - a.value);
}

function add(acc: Map<string, number>, key: string | null, v: number) {
  const k = key ?? 'Unknown';
  acc.set(k, (acc.get(k) ?? 0) + v);
}

function distributions(nodeRows: NodeRow[]): Record<number, YearDistribution> {
  const meta = countryMeta();
  const nodeByKey = new Map(nodeRows.map((n) => [`${n.iso3}|${n.year}`, n]));
  const byYear: Record<number, {
    total: number; intra: number; high: number;
    continent: Map<string, number>; income: Map<string, number>; religion: Map<string, number>;
    mipex: Map<string, number>; community: Map<string, number>;
  }> = {};

  for (const m of migrationLong()) {
    const y = (byYear[m.year] ??= {
      total: 0, intra: 0, high: 0,
      continent: new Map(), income: new Map(), religion: new Map(), mipex: new Map(), community: new Map(),
    });
    const dMeta = meta.get(m.dest);
    const oMeta = meta.get(m.origin);
    const dNode = nodeByKey.get(`${m.dest}|${m.year}`);
    const dCont = str(dMeta?.continent);
    const dInc = str(dMeta?.income_group);

    y.total += m.stock;
    if (dCont && dCont === str(oMeta?.continent)) y.intra += m.stock;
    if (dInc === 'High income') y.high += m.stock;
    add(y.continent, dCont, m.stock);
    add(y.income, dInc, m.stock);
    add(y.religion, dNode?.dominant_religion ?? null, m.stock);
    add(y.mipex, mipexTier(dNode?.integration_policy_index), m.stock);
    add(y.community, dNode?.community_label ?? null, m.stock);
  }

  const out: Record<number, YearDistribution> = {};
  for (const [year, y] of Object.entries(byYear)) {
    out[Number(year)] = {
      total_stock: y.total,
      intra_continental_share: y.total ? y.intra / y.total : 0,
      high_income_share: y.total ? y.high / y.total : 0,
      continent: toShares(y.continent),
      income: toShares(y.income, INCOME_ORDER),
      religion: toShares(y.religion),
      mipex: toShares(y.mipex, MIPEX_TIERS),
      community: toShares(y.community),
    };
  }
  return out;
}

function readJSON(rel: string): Record<string, any> | null {
  const p = path.join(DATA_ROOT, rel);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return null;
  }
}

function drivers(): Drivers {
  const rows = (f: string) => readDataCSV(`exports/${f}`) as any[];
  return {
    correlations: rows('drivers_correlations.csv'),
    profile: rows('drivers_profile.csv').map((r) => ({
      outcome: r.outcome, factor: r.factor, label: r.label, category: r.category,
      median_top: num(r.median_top), median_bottom: num(r.median_bottom), median_all: num(r.median_all),
    })),
    countries: rows('country_migration_profile.csv').map((r) => ({
      iso3: r.iso3, country_name: str(r.country_name), continent: str(r.continent), year: Number(r.year),
      immigrant_share: num(r.immigrant_share), emigrant_rate: num(r.emigrant_rate), gdp_pc_ppp: num(r.gdp_pc_ppp),
    })),
    hump: rows('migration_hump.csv'),
    hump_curve: rows('migration_hump_curve.csv'),
    gravity: rows('gravity_ppml.csv'),
    gravity_fit: rows('gravity_fit.csv'),
    model_ablation: rows('edge_model_ablation.csv'),
  };
}

/** Origin-region × destination-region migrant stock for each snapshot year (for the chord diagram). */
function regionFlows(): Dataset['region_flows'] {
  const meta = countryMeta();
  const levels = { continent: 'continent', region: 'un_region' } as const;
  const acc: Record<number, Record<keyof typeof levels, Map<string, number>>> = {};
  for (const m of migrationLong()) {
    const o = meta.get(m.origin);
    const d = meta.get(m.dest);
    if (!o || !d) continue;
    const y = (acc[m.year] ??= { continent: new Map(), region: new Map() });
    for (const [level, col] of Object.entries(levels) as [keyof typeof levels, string][]) {
      const k = `${o[col]}|${d[col]}`;
      y[level].set(k, (y[level].get(k) ?? 0) + m.stock);
    }
  }
  const toMatrix = (pairs: Map<string, number>): RegionFlows => {
    const totals = new Map<string, number>();
    for (const [k, v] of pairs) for (const part of k.split('|')) totals.set(part, (totals.get(part) ?? 0) + v);
    const labels = Array.from(totals.keys()).filter((l) => l && l !== 'null').sort();
    const idx = new Map(labels.map((l, i) => [l, i]));
    const matrix = labels.map(() => labels.map(() => 0));
    for (const [k, v] of pairs) {
      const [a, b] = k.split('|');
      if (idx.has(a) && idx.has(b)) matrix[idx.get(a)!][idx.get(b)!] += v;
    }
    return { labels, matrix };
  };
  const out: Dataset['region_flows'] = {};
  for (const [year, y] of Object.entries(acc)) {
    out[Number(year)] = { continent: toMatrix(y.continent), region: toMatrix(y.region) };
  }
  return out;
}

function displacement(): Displacement {
  return {
    backtest: readDataCSV('exports/displacement_backtest.csv') as any[],
    backtest_shares: readDataCSV('exports/displacement_backtest_shares.csv') as any[],
    model: readDataCSV('exports/displacement_model.csv'),
    meta: readJSON('exports/displacement_meta.json'),
  };
}

let datasetCache: { key: string; value: Dataset } | null = null;

export function buildDataset(): Dataset {
  const files = [
    'exports/nodes_master.csv', 'exports/network_edges.csv', 'exports/network_summary.csv',
    'exports/modularity_scores.csv', 'exports/boundary_nodes.csv', 'exports/regression_coefficients.csv',
    'exports/regression_model_comparison.csv', 'exports/country_metadata.csv', 'processed/migration_long.csv',
    'exports/drivers_correlations.csv', 'exports/country_migration_profile.csv', 'exports/gravity_ppml.csv',
    'exports/edge_model_ablation.csv', 'exports/edge_model_metrics.json',
    'exports/displacement_backtest.csv', 'exports/displacement_meta.json',
  ];
  const key = files
    .map((f) => {
      const p = path.join(DATA_ROOT, f);
      return fs.existsSync(p) ? fs.statSync(p).mtimeMs : 0;
    })
    .join('|');
  if (datasetCache && datasetCache.key === key) return datasetCache.value;

  const meta = countryMeta();
  const nodeRows = nodes();

  const edges: EdgeRow[] = readDataCSV('exports/network_edges.csv')
    .map((r) => {
      const o = String(r.origin_iso3);
      const d = String(r.dest_iso3);
      const oc = str(meta.get(o)?.continent);
      const dc = str(meta.get(d)?.continent);
      return {
        origin_iso3: o,
        dest_iso3: d,
        year: Number(r.year),
        weight: num(r.weight) ?? 0,
        same_continent: oc && dc ? Number(oc === dc) : null,
      };
    })
    .sort((a, b) => a.year - b.year || b.weight - a.weight);

  const boundary = readDataCSV('exports/boundary_nodes.csv')
    .map((r): Row => ({ ...r, country_name: str(meta.get(String(r.iso3))?.country_name) ?? r.iso3 }))
    .sort((a, b) => (num(b.boundary_score) ?? 0) - (num(a.boundary_score) ?? 0));

  const value: Dataset = {
    nodes: nodeRows,
    edges,
    network_summary: readDataCSV('exports/network_summary.csv'),
    modularity_scores: readDataCSV('exports/modularity_scores.csv'),
    boundary_nodes: boundary,
    regression_coefficients: readDataCSV('exports/regression_coefficients.csv'),
    regression_comparison: readDataCSV('exports/regression_model_comparison.csv'),
    distributions: distributions(nodeRows),
    model_metrics: readJSON('exports/edge_model_metrics.json'),
    drivers: drivers(),
    region_flows: regionFlows(),
    displacement: displacement(),
  };
  datasetCache = { key, value };
  return value;
}
