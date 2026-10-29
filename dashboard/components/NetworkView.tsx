import React, { useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { lineGrid } from '../lib/charts';
import { useChartKey } from '../lib/themeContext';
import { getCountryGeo } from '../lib/countryGeo';
import { ACCENT, categorical, fmtCompact, fmtInt, fmtNum } from '../lib/theme';
import { SNAPSHOT_YEARS, type Dataset, type NodeRow, type Year } from '../lib/types';
import { Card, Empty, Segmented, Stat, Table, td, th } from './ui';

type Tab = 'centrality' | 'communities' | 'regression';
type Metric = 'pagerank' | 'betweenness_centrality' | 'in_strength' | 'out_strength';

const METRICS: { value: Metric; label: string }[] = [
  { value: 'pagerank', label: 'PageRank' },
  { value: 'betweenness_centrality', label: 'Betweenness' },
  { value: 'in_strength', label: 'Immigrants' },
  { value: 'out_strength', label: 'Emigrants' },
];

const PREDICTOR_LABELS: Record<string, string> = {
  const: 'Intercept',
  log_population: 'log Population',
  log_gdp_per_capita: 'log GDP per capita',
  conflict_intensity: 'Conflict intensity',
  unemployment: 'Unemployment',
  education_index: 'Education index',
  visa_openness_index: 'Visa openness',
  climate_vulnerability: 'Climate vulnerability',
  cultural_openness_index: 'Cultural openness',
  integration_policy_index: 'Integration policy',
  religious_fractionalization: 'Religious fractionalisation',
  religiosity_importance_score: 'Religiosity',
  youth_share: 'Population under 15 (%)',
  urban_pct: 'Urban population (%)',
  internet_users: 'Internet users (%)',
  political_stability: 'Political stability',
  resource_rents: 'Natural-resource rents (% GDP)',
  landlocked: 'Landlocked',
};

function stars(p: number) {
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  return '';
}

function modelLabel(m: string) {
  const [kind, dir] = [m.replace(/_(inflows|outflows)$/, ''), m.endsWith('inflows') ? 'inflows' : 'outflows'];
  const k =
    kind === 'cultural_gravity' ? 'Cultural gravity' : kind === 'development' ? 'Development' : kind === 'full' ? 'Full' : 'Baseline';
  return `${k} · ${dir}`;
}

export default function NetworkView({ data, year, nodes }: { data: Dataset; year: Year; nodes: NodeRow[] }) {
  const [tab, setTab] = useState<Tab>('centrality');
  return (
    <div className="space-y-6">
      <Segmented
        ariaLabel="Network analysis section"
        value={tab}
        onChange={setTab}
        options={[
          { value: 'centrality', label: 'Centrality' },
          { value: 'communities', label: 'Communities' },
          { value: 'regression', label: 'Regression' },
        ]}
      />
      {tab === 'centrality' && <Centrality nodes={nodes} year={year} />}
      {tab === 'communities' && <Communities data={data} year={year} />}
      {tab === 'regression' && <Regression data={data} />}
    </div>
  );
}

function Centrality({ nodes, year }: { nodes: NodeRow[]; year: Year }) {
  const [metric, setMetric] = useState<Metric>('pagerank');
  const ranked = useMemo(
    () => [...nodes].sort((a, b) => (Number(b[metric]) || 0) - (Number(a[metric]) || 0)).slice(0, 20),
    [nodes, metric]
  );

  return (
    <Card
      title={`Most central countries, ${year}`}
      subtitle="PageRank: weighted attractiveness as a destination. Betweenness: how often a country bridges shortest paths."
      actions={<Segmented size="xs" ariaLabel="Ranking metric" value={metric} onChange={setMetric} options={METRICS} />}
      bodyClassName="px-2 pb-2"
    >
      <Table>
        <thead>
          <tr>
            <th className={th}>#</th>
            <th className={th}>Country</th>
            <th className={`${th} text-right`}>PageRank</th>
            <th className={`${th} text-right`}>Betweenness</th>
            <th className={`${th} text-right`}>Immigrants</th>
            <th className={`${th} text-right`}>Emigrants</th>
            <th className={th}>Community</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100">
          {ranked.map((n, i) => (
            <tr key={n.iso3} className="hover:bg-zinc-50">
              <td className={`${td} tabular-nums text-zinc-400`}>{i + 1}</td>
              <td className={`${td} font-medium text-zinc-900`}>{getCountryGeo(n.iso3).flag} {n.country_name ?? n.iso3}</td>
              <td className={`${td} text-right tabular-nums ${metric === 'pagerank' ? 'font-semibold text-zinc-900' : ''}`}>{fmtNum(n.pagerank, 4)}</td>
              <td className={`${td} text-right tabular-nums ${metric === 'betweenness_centrality' ? 'font-semibold text-zinc-900' : ''}`}>{fmtNum(n.betweenness_centrality, 4)}</td>
              <td className={`${td} text-right tabular-nums ${metric === 'in_strength' ? 'font-semibold text-zinc-900' : ''}`}>{fmtCompact(n.in_strength)}</td>
              <td className={`${td} text-right tabular-nums ${metric === 'out_strength' ? 'font-semibold text-zinc-900' : ''}`}>{fmtCompact(n.out_strength)}</td>
              <td className={`${td} text-xs`}>
                {n.louvain_community != null && (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: categorical(n.louvain_community) }} />
                    {(n.community_label ?? '').replace(/^Cluster \d+:\s*/, '')}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </Card>
  );
}

function Communities({ data, year }: { data: Dataset; year: Year }) {
  const chartKey = useChartKey();
  const q = (algo: string, y: number) => {
    const m = data.modularity_scores.find((r) => r.algorithm === algo && Number(r.year) === y);
    return m ? Number(m.modularity_q) : null;
  };
  const nComms = data.modularity_scores.find((r) => r.algorithm === 'louvain' && Number(r.year) === year)?.n_communities;
  const first = q('louvain', SNAPSHOT_YEARS[0]);
  const now = q('louvain', year);
  const boundary = data.boundary_nodes.filter((b) => b.is_boundary_node);

  const chart = {
    labels: SNAPSHOT_YEARS,
    datasets: [
      {
        label: 'Louvain',
        data: SNAPSHOT_YEARS.map((y) => q('louvain', y)),
        borderColor: ACCENT,
        backgroundColor: ACCENT,
        tension: 0.3,
        pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 5 : 3)),
      },
      {
        label: 'Leiden',
        data: SNAPSHOT_YEARS.map((y) => q('leiden', y)),
        borderColor: '#0d9488',
        backgroundColor: '#0d9488',
        borderDash: [4, 4],
        tension: 0.3,
        pointRadius: 3,
      },
    ],
  };
  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'top', align: 'end' } },
    scales: { x: { grid: { display: false } }, y: { grid: lineGrid, suggestedMin: 0.45, suggestedMax: 0.65 } },
  };

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      <Card
        className="lg:col-span-2"
        title="Modularity over time"
        subtitle="Higher Q means migration is more concentrated within regional blocs; lower Q means a more integrated global network."
      >
        <div className="mb-5 grid grid-cols-3 gap-4">
          <Stat label={`Q in ${SNAPSHOT_YEARS[0]}`} value={fmtNum(first, 3)} />
          <Stat
            label={`Q in ${year}`}
            value={fmtNum(now, 3)}
            hint={first && now ? `${now >= first ? '+' : ''}${(((now - first) / first) * 100).toFixed(1)}% vs ${SNAPSHOT_YEARS[0]}` : undefined}
            tone="accent"
          />
          <Stat label="Communities" value={nComms ?? '—'} hint={`Louvain, ${year}`} />
        </div>
        <div className="h-60">
          <Line key={chartKey} data={chart} options={options} />
        </div>
      </Card>

      <Card
        title="Boundary states"
        subtitle="Countries that switched community in at least half of the seven transitions between snapshots"
        bodyClassName="px-2 pb-2"
      >
        {boundary.length === 0 ? (
          <Empty>No boundary states</Empty>
        ) : (
          <div className="max-h-[360px] overflow-y-auto">
            <Table>
              <thead className="sticky top-0 bg-white">
                <tr>
                  <th className={th}>Country</th>
                  <th className={`${th} text-right`}>Switches</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {boundary.map((b) => (
                  <tr key={b.iso3}>
                    <td className={td}>{getCountryGeo(b.iso3).flag} {b.country_name}</td>
                    <td className={`${td} text-right tabular-nums`}>{b.n_changes} / {b.total_periods}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>
        )}
      </Card>
    </div>
  );
}

function Regression({ data }: { data: Dataset }) {
  const models = data.regression_comparison;
  const [model, setModel] = useState<string>(
    models.find((m) => m.model === 'cultural_gravity_inflows')?.model ?? models[0]?.model ?? ''
  );
  const summary = models.find((m) => m.model === model);
  const coefs = data.regression_coefficients.filter((c) => c.model === model);

  if (models.length === 0) return <Card><Empty>No regression results found.</Empty></Card>;

  return (
    <Card
      title="What predicts migrant stock?"
      subtitle={`OLS on log ${summary?.dependent === 'log_out_strength' ? 'emigrant' : 'immigrant'} stock per country-year. Standard errors clustered by country.`}
      actions={
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-xs text-zinc-800 outline-none focus:border-blue-500"
        >
          {models.map((m) => (
            <option key={m.model} value={m.model}>{modelLabel(m.model)}</option>
          ))}
        </select>
      }
    >
      <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="R²" value={fmtNum(summary?.r_squared, 3)} tone="accent" />
        <Stat label="Adjusted R²" value={fmtNum(summary?.adj_r2, 3)} />
        <Stat label="Observations" value={fmtInt(summary?.n_obs)} />
        <Stat label="AIC" value={fmtInt(summary?.aic)} />
      </div>
      <div className="-mx-3">
        <Table>
          <thead>
            <tr>
              <th className={th}>Predictor</th>
              <th className={`${th} text-right`}>Coefficient</th>
              <th className={`${th} text-right`}>Std. error</th>
              <th className={`${th} text-right`}>95% CI</th>
              <th className={`${th} text-right`}>p-value</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {coefs.map((c) => {
              const p = Number(c.p_value);
              const sig = p < 0.05;
              return (
                <tr key={c.predictor} className={sig ? '' : 'text-zinc-400'}>
                  <td className={`${td} ${sig ? 'font-medium text-zinc-900' : 'text-zinc-400'}`}>
                    {PREDICTOR_LABELS[c.predictor] ?? c.predictor}
                  </td>
                  <td className={`${td} text-right tabular-nums ${sig ? 'text-zinc-900' : 'text-zinc-400'}`}>
                    {fmtNum(Number(c.coefficient), 4)}
                    <span className="ml-0.5 inline-block w-5 text-left text-blue-600">{stars(p)}</span>
                  </td>
                  <td className={`${td} text-right tabular-nums ${sig ? '' : 'text-zinc-400'}`}>{fmtNum(Number(c.std_error), 4)}</td>
                  <td className={`${td} text-right tabular-nums ${sig ? '' : 'text-zinc-400'}`}>
                    [{fmtNum(Number(c.ci_lower_95), 3)}, {fmtNum(Number(c.ci_upper_95), 3)}]
                  </td>
                  <td className={`${td} text-right tabular-nums ${sig ? '' : 'text-zinc-400'}`}>
                    {p < 0.0001 ? '< 0.0001' : p.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      </div>
      <p className="mt-4 text-[11px] text-zinc-500">* p &lt; 0.05 · ** p &lt; 0.01 · *** p &lt; 0.001. Non-significant predictors are greyed out.</p>
    </Card>
  );
}
