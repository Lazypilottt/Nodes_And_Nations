import { motion } from 'framer-motion';
import React, { useMemo, useState } from 'react';
import { Line } from 'react-chartjs-2';
import useSWR from 'swr';
import { lineGrid } from '../lib/charts';
import { COUNTRY_LIST, getCountryGeo } from '../lib/countryGeo';
import { fetcher } from '../lib/fetcher';
import { ACCENT, fmtCompact, fmtNum, fmtPct } from '../lib/theme';
import { useChartKey } from '../lib/themeContext';
import { SNAPSHOT_YEARS, type Dataset, type NodeRow, type Year } from '../lib/types';
import { GapTable } from './Flows';
import { Card, CountUp, Empty, Stat, selectClass } from './ui';

const OUTBOUND = '#d97706';

interface BriefResponse {
  iso3: string;
  year: number;
  inbound: { iso3: string; stock: number }[];
  outbound: { iso3: string; stock: number }[];
  gaps: Record<'diaspora_under' | 'diaspora_over' | 'immigrant_under' | 'immigrant_over', any[]>;
  if_displaced: { iso3: string; share: number }[];
  exposure: { iso3: string; share: number; displaced_abroad: number; expected: number }[];
}

const PEER_FACTORS: { key: keyof NodeRow; label: string; fmt: (v: number) => string; higherIs?: string }[] = [
  { key: 'gdp_pc_ppp', label: 'GDP per capita (PPP)', fmt: (v) => `$${fmtCompact(v, 1)}` },
  { key: 'life_expectancy', label: 'Life expectancy', fmt: (v) => `${v.toFixed(1)} yrs` },
  { key: 'urban_pct', label: 'Urban population', fmt: (v) => `${v.toFixed(0)}%` },
  { key: 'youth_share', label: 'Population under 15', fmt: (v) => `${v.toFixed(0)}%` },
  { key: 'internet_users', label: 'Internet users', fmt: (v) => `${v.toFixed(0)}%` },
  { key: 'political_stability', label: 'Political stability', fmt: (v) => v.toFixed(2) },
  { key: 'homicide_rate', label: 'Homicide rate', fmt: (v) => `${v.toFixed(1)} /100k` },
  { key: 'remittances_gdp', label: 'Remittances received', fmt: (v) => `${v.toFixed(1)}% GDP` },
  { key: 'refugees_hosted_per_1k', label: 'Refugees hosted', fmt: (v) => `${v.toFixed(1)} /1k` },
];

function percentile(values: number[], x: number) {
  if (values.length === 0) return null;
  return values.filter((v) => v < x).length / values.length;
}

function median(values: number[]) {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : null;
}

export default function CountryBrief({
  data,
  year,
  iso3,
  onChange,
}: {
  data: Dataset;
  year: Year;
  iso3: string;
  onChange: (iso3: string) => void;
}) {
  const chartKey = useChartKey();
  const [copied, setCopied] = useState(false);
  const { data: brief } = useSWR<BriefResponse>(`/api/brief?iso3=${iso3}`, fetcher, { keepPreviousData: true });
  const geo = getCountryGeo(iso3);

  const series = useMemo(() => data.nodes.filter((n) => n.iso3 === iso3).sort((a, b) => a.year - b.year), [data.nodes, iso3]);
  const node = series.find((n) => n.year === year);
  const profile = data.drivers.countries.find((c) => c.iso3 === iso3 && c.year === year);
  const yearNodes = useMemo(() => data.nodes.filter((n) => n.year === year), [data.nodes, year]);
  const rank = useMemo(
    () => [...yearNodes].sort((a, b) => (b.pagerank ?? 0) - (a.pagerank ?? 0)).findIndex((n) => n.iso3 === iso3) + 1,
    [yearNodes, iso3]
  );

  const peers = useMemo(
    () =>
      PEER_FACTORS.map((f) => {
        const x = node?.[f.key] as number | null | undefined;
        const all = yearNodes.map((n) => n[f.key] as number | null).filter((v): v is number => v != null && Number.isFinite(v));
        const group = yearNodes
          .filter((n) => n.income_group && n.income_group === node?.income_group)
          .map((n) => n[f.key] as number | null)
          .filter((v): v is number => v != null && Number.isFinite(v));
        return {
          ...f,
          value: x ?? null,
          pct: x != null ? percentile(all, x) : null,
          groupPct: (() => {
            const m = median(group);
            return m != null ? percentile(all, m) : null;
          })(),
        };
      }),
    [node, yearNodes]
  );

  const trend = {
    labels: SNAPSHOT_YEARS,
    datasets: [
      {
        label: 'Immigrants',
        data: SNAPSHOT_YEARS.map((y) => series.find((n) => n.year === y)?.in_strength ?? null),
        borderColor: ACCENT,
        backgroundColor: 'rgba(37, 99, 235, 0.10)',
        fill: true,
        tension: 0.35,
        pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 5 : 2.5)),
      },
      {
        label: 'Emigrants',
        data: SNAPSHOT_YEARS.map((y) => series.find((n) => n.year === y)?.out_strength ?? null),
        borderColor: OUTBOUND,
        backgroundColor: 'rgba(217, 119, 6, 0.08)',
        fill: true,
        tension: 0.35,
        pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 5 : 2.5)),
      },
    ],
  };
  const trendOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top', align: 'end' },
      tooltip: { callbacks: { label: (ctx: any) => ` ${ctx.dataset.label}: ${fmtCompact(ctx.raw)}` } },
    },
    scales: { x: { grid: { display: false } }, y: { grid: lineGrid, ticks: { callback: (v: number) => fmtCompact(v, 0) } } },
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked: the URL in the address bar is already shareable */
    }
  };

  const downloadCsv = () => {
    const rows: (string | number)[][] = [['section', 'item', 'value']];
    const push = (section: string, item: string, value: unknown) => rows.push([section, item, value == null ? '' : String(value)]);
    push('country', 'name', geo.name);
    push('country', 'year', year);
    push('migration', 'immigrants', node?.in_strength);
    push('migration', 'emigrants', node?.out_strength);
    push('migration', 'immigrant_share_pct', profile?.immigrant_share?.toFixed(2));
    push('migration', 'emigrant_rate_pct', profile?.emigrant_rate?.toFixed(2));
    push('network', 'pagerank_rank', rank || '');
    for (const r of brief?.inbound ?? []) push('top_origins', getCountryGeo(r.iso3).name, r.stock);
    for (const r of brief?.outbound ?? []) push('top_destinations', getCountryGeo(r.iso3).name, r.stock);
    for (const p of peers) push('peer_percentile', p.label, p.pct != null ? Math.round(p.pct * 100) : '');
    for (const [k, list] of Object.entries(brief?.gaps ?? {}))
      for (const g of list) push(`gap_${k}`, `${getCountryGeo(g.origin_iso3).name} -> ${getCountryGeo(g.dest_iso3).name}`, `${g.migrant_stock} actual / ${g.expected} expected`);
    for (const r of brief?.if_displaced ?? []) push('if_displaced_share', getCountryGeo(r.iso3).name, r.share.toFixed(4));
    for (const r of brief?.exposure ?? []) push('exposure_expected_arrivals', getCountryGeo(r.iso3).name, Math.round(r.expected));
    const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `nodes-and-nations_${iso3}_${year}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <motion.section
        key={iso3}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-xl border border-zinc-200 bg-white p-5"
      >
        <div className="pointer-events-none absolute -right-6 -top-10 select-none text-[9rem] leading-none opacity-[0.07]">{geo.flag}</div>
        <div className="relative flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Country brief · {year}</div>
            <h1 className="mt-1 flex items-center gap-2 text-2xl font-semibold text-zinc-900">
              <span>{geo.flag}</span>
              {node?.country_name ?? geo.name}
            </h1>
            <p className="mt-1 text-xs text-zinc-500">
              {[node?.un_region ?? geo.un_region, node?.income_group ?? geo.income_group, node?.community_label?.replace(/^Cluster \d+:\s*/, '')]
                .filter(Boolean)
                .join(' · ')}
            </p>
          </div>
          <div className="no-print flex flex-wrap items-center gap-2">
            <select className={`${selectClass} w-56 py-1.5`} value={iso3} onChange={(e) => onChange(e.target.value)} aria-label="Country">
              {COUNTRY_LIST.map((c) => (
                <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>
              ))}
            </select>
            <button onClick={copyLink} className="rounded-lg border border-zinc-200 px-3 py-1.5 text-xs text-zinc-700 hover:bg-zinc-100">
              {copied ? 'Link copied' : 'Copy link'}
            </button>
            <button onClick={downloadCsv} className="rounded-lg border border-zinc-200 px-3 py-1.5 text-xs text-zinc-700 hover:bg-zinc-100">
              Download CSV
            </button>
            <button onClick={() => window.print()} className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700">
              Print / PDF
            </button>
          </div>
        </div>
        <div className="relative mt-5 grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Immigrants" value={<CountUp value={node?.in_strength} format={(n) => fmtCompact(n)} />} tone="accent" />
          <Stat label="Emigrants" value={<CountUp value={node?.out_strength} format={(n) => fmtCompact(n)} />} />
          <Stat
            label="Immigrant share"
            value={<CountUp value={profile?.immigrant_share} format={(n) => `${n.toFixed(1)}%`} />}
            hint="of residents"
          />
          <Stat
            label="Emigration rate"
            value={<CountUp value={profile?.emigrant_rate} format={(n) => `${n.toFixed(1)}%`} />}
            hint="of native-born"
          />
          <Stat label="Network rank" value={rank > 0 ? `#${rank}` : '—'} hint="by PageRank" />
          <Stat label="Refugees hosted" value={node?.refugees_hosted_per_1k != null ? `${fmtNum(node.refugees_hosted_per_1k, 1)} /1k` : '—'} />
        </div>
      </motion.section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Immigrants and emigrants, 1990–2025" subtitle="2025 is extrapolated from the 2024 UN DESA release">
          <div className="h-64">
            {series.length ? <Line key={`${chartKey}-${iso3}`} data={trend} options={trendOptions} /> : <Empty>No data</Empty>}
          </div>
        </Card>
        <Card title="Main corridors" subtitle={`Latest year (${brief?.year ?? '…'})`} delay={0.05}>
          <CorridorList title="Immigrants come from" rows={brief?.inbound} color={ACCENT} onPick={onChange} />
          <div className="h-4" />
          <CorridorList title="Emigrants live in" rows={brief?.outbound} color={OUTBOUND} onPick={onChange} />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="How it compares" subtitle={`Percentile among all countries in ${year}. Grey tick: median of ${node?.income_group ?? 'its income group'}.`} delay={0.05}>
          <ul className="space-y-3">
            {peers.map((p, i) => (
              <li key={String(p.key)} className="text-xs">
                <div className="mb-1 flex justify-between gap-2">
                  <span className="text-zinc-700">{p.label}</span>
                  <span className="tabular-nums text-zinc-500">
                    <span className="font-medium text-zinc-900">{p.value != null ? p.fmt(p.value) : '—'}</span>
                    {p.pct != null && <span className="ml-1.5">{Math.round(p.pct * 100)}th pct</span>}
                  </span>
                </div>
                <div className="relative h-2 rounded-full bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-300">
                  {p.groupPct != null && (
                    <span className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-zinc-500" style={{ left: `${p.groupPct * 100}%` }} />
                  )}
                  {p.pct != null && (
                    <motion.span
                      className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-blue-600 shadow dark:bg-blue-400"
                      initial={{ left: '0%', opacity: 0 }}
                      animate={{ left: `${p.pct * 100}%`, opacity: 1 }}
                      transition={{ duration: 0.7, delay: i * 0.04, ease: [0.22, 1, 0.36, 1] }}
                    />
                  )}
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="Crisis exposure" subtitle="From the displacement model (Displacement planner tab)" delay={0.1}>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">If a crisis displaced people from {geo.name}, they would go to</h3>
          <ul className="space-y-1.5">
            {(brief?.if_displaced ?? []).slice(0, 6).map((r, i) => (
              <li key={r.iso3} className="grid grid-cols-[minmax(0,9rem)_1fr_3rem] items-center gap-2 text-xs">
                <span className="truncate text-zinc-700">{getCountryGeo(r.iso3).flag} {getCountryGeo(r.iso3).name}</span>
                <span className="h-1.5 overflow-hidden rounded-full bg-zinc-100">
                  <motion.span
                    className="block h-full rounded-full bg-red-500/80"
                    initial={{ width: 0 }}
                    animate={{ width: `${(r.share / (brief?.if_displaced[0]?.share || 1)) * 100}%` }}
                    transition={{ duration: 0.6, delay: i * 0.04 }}
                  />
                </span>
                <span className="text-right tabular-nums text-zinc-900">{fmtPct(r.share)}</span>
              </li>
            ))}
          </ul>
          <h3 className="mb-2 mt-5 text-xs font-medium uppercase tracking-wide text-zinc-500">Current displacement most likely to reach {geo.name}</h3>
          {brief && brief.exposure.length === 0 ? (
            <p className="text-xs text-zinc-500">No current displacement crisis sends a meaningful share here.</p>
          ) : (
            <ul className="space-y-1.5">
              {(brief?.exposure ?? []).map((r) => (
                <li key={r.iso3} className="flex items-center justify-between gap-2 text-xs">
                  <span className="truncate text-zinc-700">
                    {getCountryGeo(r.iso3).flag} {getCountryGeo(r.iso3).name}
                    <span className="ml-1.5 text-zinc-400">{fmtCompact(r.displaced_abroad)} displaced abroad</span>
                  </span>
                  <span className="shrink-0 tabular-nums text-zinc-900">{fmtPct(r.share)} share</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title={`${geo.name}’s diaspora: gaps against the gravity model`} subtitle="Where fewer (or more) of its emigrants live than distance and ties predict" bodyClassName="px-3 pb-3 pt-2" delay={0.05}>
          <GapTable rows={brief?.gaps.diaspora_under ?? []} kind="under" onOpenBrief={onChange} />
          <div className="h-3" />
          <GapTable rows={brief?.gaps.diaspora_over ?? []} kind="over" onOpenBrief={onChange} />
        </Card>
        <Card title={`Immigrants in ${geo.name}: gaps against the gravity model`} subtitle="Origins that send fewer (or more) people here than expected" bodyClassName="px-3 pb-3 pt-2" delay={0.1}>
          <GapTable rows={brief?.gaps.immigrant_under ?? []} kind="under" onOpenBrief={onChange} />
          <div className="h-3" />
          <GapTable rows={brief?.gaps.immigrant_over ?? []} kind="over" onOpenBrief={onChange} />
        </Card>
      </div>
    </div>
  );
}

function CorridorList({
  title,
  rows,
  color,
  onPick,
}: {
  title: string;
  rows?: { iso3: string; stock: number }[];
  color: string;
  onPick: (iso3: string) => void;
}) {
  const max = Math.max(...(rows ?? []).map((r) => r.stock), 1);
  return (
    <div>
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">{title}</h3>
      <ul className="space-y-1.5">
        {(rows ?? []).slice(0, 6).map((r, i) => (
          <li key={r.iso3} className="text-xs">
            <div className="flex items-center justify-between gap-2">
              <button className="truncate text-left text-zinc-700 hover:text-blue-600 dark:hover:text-blue-400" onClick={() => onPick(r.iso3)}>
                {getCountryGeo(r.iso3).flag} {getCountryGeo(r.iso3).name}
              </button>
              <span className="tabular-nums text-zinc-900">{fmtCompact(r.stock)}</span>
            </div>
            <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-zinc-100">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: color }}
                initial={{ width: 0 }}
                animate={{ width: `${(r.stock / max) * 100}%` }}
                transition={{ duration: 0.6, delay: i * 0.04 }}
              />
            </div>
          </li>
        ))}
        {!rows && <li className="text-xs text-zinc-400">Loading…</li>}
        {rows?.length === 0 && <li className="text-xs text-zinc-400">No recorded corridors</li>}
      </ul>
    </div>
  );
}

