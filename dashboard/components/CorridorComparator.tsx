import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import useSWR from 'swr';
import { lineGrid } from '../lib/charts';
import { useChartKey } from '../lib/themeContext';
import { COUNTRY_LIST, getCountryGeo } from '../lib/countryGeo';
import { fetcher } from '../lib/fetcher';
import { ACCENT, fmtCompact, fmtInt, fmtNum } from '../lib/theme';
import { SNAPSHOT_YEARS, type NodeRow, type Year } from '../lib/types';
import { Card, Field, Stat, selectClass } from './ui';

const B_COLOR = '#d97706';

interface CorridorResponse {
  a: string;
  b: string;
  series: { year: number; a_to_b: number; b_to_a: number }[];
}

const ROWS: { label: string; get: (n?: NodeRow) => React.ReactNode }[] = [
  { label: 'Population', get: (n) => fmtCompact(n?.population) },
  { label: 'GDP per capita', get: (n) => (n?.gdp_per_capita ? `$${fmtInt(n.gdp_per_capita)}` : '—') },
  { label: 'Unemployment', get: (n) => (n?.unemployment != null ? `${fmtNum(n.unemployment, 1)}%` : '—') },
  { label: 'Immigrant stock', get: (n) => fmtCompact(n?.in_strength) },
  { label: 'Emigrant stock', get: (n) => fmtCompact(n?.out_strength) },
  { label: 'PageRank', get: (n) => fmtNum(n?.pagerank, 4) },
  { label: 'Visa openness', get: (n) => fmtNum(n?.visa_openness_index, 0) },
  { label: 'Climate vulnerability', get: (n) => fmtNum(n?.climate_vulnerability, 3) },
  { label: 'Dominant religion', get: (n) => n?.dominant_religion ?? '—' },
  { label: 'Integration policy', get: (n) => fmtNum(n?.integration_policy_index, 2) },
  { label: 'Network community', get: (n) => (n?.community_label ?? '—').replace(/^Cluster \d+:\s*/, '') },
];

export default function CorridorComparator({ nodesByIso, year }: { nodesByIso: Record<string, NodeRow>; year: Year }) {
  const chartKey = useChartKey();
  const [a, setA] = useState('IND');
  const [b, setB] = useState('ARE');
  const { data, error } = useSWR<CorridorResponse>(a !== b ? `/api/corridor?a=${a}&b=${b}` : null, fetcher, {
    keepPreviousData: true,
  });

  const ga = getCountryGeo(a);
  const gb = getCountryGeo(b);
  const na = nodesByIso[a];
  const nb = nodesByIso[b];
  const point = data?.series.find((s) => s.year === year);
  const aToB = point?.a_to_b ?? 0;
  const bToA = point?.b_to_a ?? 0;

  const chart = {
    labels: SNAPSHOT_YEARS,
    datasets: [
      {
        label: `${ga.name} → ${gb.name}`,
        data: SNAPSHOT_YEARS.map((y) => data?.series.find((s) => s.year === y)?.a_to_b ?? 0),
        borderColor: ACCENT,
        backgroundColor: ACCENT,
        tension: 0.3,
        pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 5 : 2.5)),
      },
      {
        label: `${gb.name} → ${ga.name}`,
        data: SNAPSHOT_YEARS.map((y) => data?.series.find((s) => s.year === y)?.b_to_a ?? 0),
        borderColor: B_COLOR,
        backgroundColor: B_COLOR,
        tension: 0.3,
        pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 5 : 2.5)),
      },
    ],
  };
  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', align: 'end' },
      tooltip: { callbacks: { label: (ctx: any) => ` ${ctx.dataset.label}: ${fmtInt(ctx.raw)}` } },
    },
    scales: { x: { grid: { display: false } }, y: { grid: lineGrid, beginAtZero: true, ticks: { callback: (v: number) => fmtCompact(v, 0) } } },
  };

  return (
    <div className="space-y-6">
      <Card>
        <div className="grid grid-cols-1 items-end gap-4 sm:grid-cols-[1fr_auto_1fr]">
          <Field label="Country A">
            <select className={selectClass} value={a} onChange={(e) => setA(e.target.value)}>
              {COUNTRY_LIST.map((c) => <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>)}
            </select>
          </Field>
          <button
            onClick={() => {
              setA(b);
              setB(a);
            }}
            aria-label="Swap countries"
            className="h-[38px] rounded-lg border border-zinc-200 px-3 text-sm text-zinc-600 hover:bg-zinc-50"
          >
            ⇄
          </button>
          <Field label="Country B">
            <select className={selectClass} value={b} onChange={(e) => setB(e.target.value)}>
              {COUNTRY_LIST.map((c) => <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>)}
            </select>
          </Field>
        </div>
        {a === b && <p className="mt-3 text-xs text-amber-600">Pick two different countries.</p>}
        {error && <p className="mt-3 text-xs text-red-600">{error.message}</p>}
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Bilateral migrant stock over time" subtitle="People born in one country living in the other">
          <div className="mb-5 grid grid-cols-3 gap-4">
            <Stat label={`${ga.iso3} → ${gb.iso3}, ${year}`} value={fmtCompact(aToB)} tone="accent" />
            <Stat label={`${gb.iso3} → ${ga.iso3}, ${year}`} value={fmtCompact(bToA)} />
            <Stat
              label="Net flow"
              value={aToB === bToA ? '0' : `${fmtCompact(Math.abs(aToB - bToA))}`}
              hint={aToB === bToA ? 'Balanced' : `towards ${aToB > bToA ? gb.name : ga.name}`}
            />
          </div>
          <div className="h-64">
            <Line key={chartKey} data={chart} options={options} />
          </div>
        </Card>

        <Card title={`Profiles, ${year}`} bodyClassName="px-5 pb-3 pt-2">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[11px] uppercase tracking-wide text-zinc-500">
                <th className="py-2 text-left font-medium" />
                <th className="py-2 text-right font-medium">{ga.flag} {ga.iso3}</th>
                <th className="py-2 text-right font-medium">{gb.flag} {gb.iso3}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {ROWS.map((r) => (
                <tr key={r.label}>
                  <td className="py-2 pr-2 text-xs text-zinc-500">{r.label}</td>
                  <td className="max-w-[110px] truncate py-2 pl-2 text-right tabular-nums text-zinc-900">{r.get(na)}</td>
                  <td className="max-w-[110px] truncate py-2 pl-2 text-right tabular-nums text-zinc-900">{r.get(nb)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
}
