import { chordDirected, ribbonArrow, type Chord, type ChordGroup } from 'd3-chord';
import { arc as d3arc } from 'd3-shape';
import { AnimatePresence, motion } from 'framer-motion';
import React, { useMemo, useState } from 'react';
import { Line, Scatter } from 'react-chartjs-2';
import useSWR from 'swr';
import { lineGrid } from '../lib/charts';
import { COUNTRY_LIST, getCountryGeo } from '../lib/countryGeo';
import { fetcher } from '../lib/fetcher';
import { categorical, fmtCompact, fmtInt } from '../lib/theme';
import { useChartKey, usePalette } from '../lib/themeContext';
import { SNAPSHOT_YEARS, type Dataset, type RegionFlows, type Year } from '../lib/types';
import { Card, Empty, Segmented, Table, selectClass, td, th } from './ui';

const UNDER = '#ea580c';
const OVER = '#2563eb';

export default function Flows({ data, year, onOpenBrief }: { data: Dataset; year: Year; onOpenBrief: (iso3: string) => void }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <ChordCard data={data} year={year} />
        <StreamCard data={data} year={year} />
      </div>
      <Gaps onOpenBrief={onOpenBrief} />
    </div>
  );
}

// ── Chord diagram ─────────────────────────────────────────────────────────────

function ChordCard({ data, year }: { data: Dataset; year: Year }) {
  const [level, setLevel] = useState<'continent' | 'region'>('continent');
  const [active, setActive] = useState<number | null>(null);
  const flows = data.region_flows[year]?.[level];

  return (
    <Card
      className="lg:col-span-3"
      title={`Who moves where, ${year}`}
      subtitle="Migrant stock between world regions. Ribbons point from origin to destination; loops are moves within a region."
      actions={
        <Segmented
          size="xs"
          ariaLabel="Region level"
          value={level}
          onChange={(l) => {
            setLevel(l);
            setActive(null);
          }}
          options={[
            { value: 'continent', label: 'Continents' },
            { value: 'region', label: 'Regions' },
          ]}
        />
      }
    >
      {!flows ? (
        <Empty>No flow data for {year}</Empty>
      ) : (
        <div className="grid grid-cols-1 items-center gap-4 md:grid-cols-[1fr_14rem]">
          <ChordDiagram flows={flows} active={active} onActive={setActive} level={level} year={year} />
          <ChordSide flows={flows} active={active} />
        </div>
      )}
    </Card>
  );
}

function ChordDiagram({
  flows,
  active,
  onActive,
  level,
  year,
}: {
  flows: RegionFlows;
  active: number | null;
  onActive: (i: number | null) => void;
  level: string;
  year: number;
}) {
  const palette = usePalette();
  const size = 560;
  const outer = size / 2 - (level === 'region' ? 92 : 64);
  const inner = outer - 14;

  const { chords, groups } = useMemo(() => {
    const layout = chordDirected()
      .padAngle(level === 'region' ? 0.02 : 0.04)
      .sortSubgroups((a, b) => b - a)
      .sortChords((a, b) => b - a)(flows.matrix);
    return { chords: Array.from(layout) as Chord[], groups: layout.groups as ChordGroup[] };
  }, [flows, level]);

  const ribbon = useMemo(() => ribbonArrow<Chord, any>().radius(inner - 1).padAngle(1 / inner), [inner]);
  const arcGen = useMemo(() => d3arc<ChordGroup>().innerRadius(inner).outerRadius(outer), [inner, outer]);
  const total = flows.matrix.flat().reduce((a, b) => a + b, 0);

  return (
    <svg viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`} className="mx-auto h-auto w-full max-w-[560px]" role="img" aria-label="Chord diagram of migration between regions" onMouseLeave={() => onActive(null)}>
      <AnimatePresence mode="wait">
        <motion.g key={`${level}-${year}`} initial={{ opacity: 0, rotate: -6 }} animate={{ opacity: 1, rotate: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}>
          <g>
            {chords.map((c, i) => {
              const on = active == null || c.source.index === active || c.target.index === active;
              return (
                <motion.path
                  key={i}
                  d={ribbon(c) as unknown as string}
                  fill={categorical(c.source.index)}
                  stroke={palette.surface}
                  strokeWidth={0.5}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: on ? (active == null ? 0.62 : 0.85) : 0.05 }}
                  transition={{ duration: 0.3, delay: active == null ? Math.min(i * 0.004, 0.4) : 0 }}
                >
                  <title>
                    {`${flows.labels[c.source.index]} → ${flows.labels[c.target.index]}: ${fmtCompact(c.source.value)} migrants`}
                  </title>
                </motion.path>
              );
            })}
          </g>
          {groups.map((g) => {
            const mid = (g.startAngle + g.endAngle) / 2;
            const flip = mid > Math.PI;
            const share = g.value / total;
            return (
              <g key={g.index} onMouseEnter={() => onActive(g.index)} className="cursor-pointer">
                <path
                  d={arcGen(g) ?? undefined}
                  fill={categorical(g.index)}
                  stroke={palette.surface}
                  opacity={active == null || active === g.index ? 1 : 0.35}
                  style={{ transition: 'opacity 0.3s' }}
                />
                {share > 0.004 && (
                  <text
                    transform={`rotate(${(mid * 180) / Math.PI - 90}) translate(${outer + 6}) ${flip ? 'rotate(180)' : ''}`}
                    textAnchor={flip ? 'end' : 'start'}
                    dominantBaseline="middle"
                    fontSize={level === 'region' ? 10 : 12}
                    fill={active == null || active === g.index ? palette.ink : palette.muted}
                    style={{ transition: 'fill 0.3s' }}
                  >
                    {flows.labels[g.index]}
                  </text>
                )}
              </g>
            );
          })}
        </motion.g>
      </AnimatePresence>
    </svg>
  );
}

function ChordSide({ flows, active }: { flows: RegionFlows; active: number | null }) {
  if (active == null) {
    const total = flows.matrix.flat().reduce((a, b) => a + b, 0);
    const within = flows.matrix.reduce((a, row, i) => a + row[i], 0);
    return (
      <div className="space-y-3 text-xs text-zinc-600">
        <p className="text-sm font-medium text-zinc-900">Hover a region</p>
        <p>to see where its emigrants live and where its immigrants come from.</p>
        <p>
          <span className="text-lg font-semibold tabular-nums text-zinc-900">{Math.round((within / total) * 100)}%</span>
          <br />of migrants live in their own {flows.labels.length > 8 ? 'region' : 'continent'}.
        </p>
      </div>
    );
  }
  const label = flows.labels[active];
  const out = flows.matrix[active].map((v, j) => ({ j, v })).filter((x) => x.v > 0).sort((a, b) => b.v - a.v).slice(0, 5);
  const inn = flows.matrix.map((row, i) => ({ j: i, v: row[active] })).filter((x) => x.v > 0).sort((a, b) => b.v - a.v).slice(0, 5);
  const list = (rows: { j: number; v: number }[]) => (
    <ul className="mt-1.5 space-y-1">
      {rows.map((r) => (
        <li key={r.j} className="flex items-center justify-between gap-2">
          <span className="flex min-w-0 items-center gap-1.5 truncate text-zinc-700">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: categorical(r.j) }} />
            <span className="truncate">{flows.labels[r.j]}</span>
          </span>
          <span className="tabular-nums text-zinc-900">{fmtCompact(r.v)}</span>
        </li>
      ))}
    </ul>
  );
  return (
    <motion.div key={active} initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} className="space-y-4 text-xs">
      <p className="flex items-center gap-2 text-sm font-medium text-zinc-900">
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: categorical(active) }} />
        {label}
      </p>
      <div>
        <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Its emigrants live in</div>
        {list(out)}
      </div>
      <div>
        <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Its immigrants come from</div>
        {list(inn)}
      </div>
    </motion.div>
  );
}

// ── Stacked trend ─────────────────────────────────────────────────────────────

function StreamCard({ data, year }: { data: Dataset; year: Year }) {
  const [side, setSide] = useState<'dest' | 'origin'>('dest');
  const chartKey = useChartKey();
  const labels = data.region_flows[SNAPSHOT_YEARS[0]]?.continent.labels ?? [];

  const datasets = labels.map((label, i) => ({
    label,
    data: SNAPSHOT_YEARS.map((y) => {
      const f = data.region_flows[y]?.continent;
      if (!f) return null;
      const k = f.labels.indexOf(label);
      if (k < 0) return 0;
      return side === 'dest' ? f.matrix.reduce((a, row) => a + row[k], 0) : f.matrix[k].reduce((a, b) => a + b, 0);
    }),
    borderColor: categorical(i),
    backgroundColor: `${categorical(i)}cc`,
    fill: true,
    tension: 0.35,
    pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 3.5 : 0)),
    borderWidth: 1,
  }));

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'bottom' },
      tooltip: { callbacks: { label: (ctx: any) => ` ${ctx.dataset.label}: ${fmtCompact(ctx.raw)}` } },
    },
    scales: {
      x: { grid: { display: false } },
      y: { stacked: true, grid: lineGrid, ticks: { callback: (v: number) => fmtCompact(v, 0) } },
    },
  };

  return (
    <Card
      className="lg:col-span-2"
      title="Migrant stock by continent"
      subtitle={side === 'dest' ? 'Where migrants live, 1990–2025' : 'Where migrants were born, 1990–2025'}
      actions={
        <Segmented
          size="xs"
          ariaLabel="Group by"
          value={side}
          onChange={setSide}
          options={[
            { value: 'dest', label: 'Destination' },
            { value: 'origin', label: 'Origin' },
          ]}
        />
      }
    >
      <div className="h-[420px]">
        <Line key={`${chartKey}-${side}`} data={{ labels: SNAPSHOT_YEARS, datasets }} options={options} />
      </div>
    </Card>
  );
}

// ── Corridor gaps ─────────────────────────────────────────────────────────────

interface PotentialResponse {
  counts: { under: number; over: number; total: number };
  under: Gap[];
  over: Gap[];
  scatter: { origin_iso3: string; dest_iso3: string; migrant_stock: number; expected: number; status: string }[];
}
interface Gap {
  origin_iso3: string;
  dest_iso3: string;
  migrant_stock: number;
  expected: number;
  ratio: number;
  gap: number;
  contig: number;
  comlang_ethno: number;
  colony: number;
}

export function GapTable({ rows, kind, onOpenBrief }: { rows: Gap[]; kind: 'under' | 'over'; onOpenBrief?: (iso3: string) => void }) {
  if (rows.length === 0) return <Empty>None</Empty>;
  const name = (iso: string) => (
    <button className="hover:text-blue-600 dark:hover:text-blue-400" onClick={() => onOpenBrief?.(iso)}>
      {getCountryGeo(iso).flag} {getCountryGeo(iso).name}
    </button>
  );
  return (
    <Table>
      <thead>
        <tr>
          <th className={th}>Corridor</th>
          <th className={`${th} text-right`}>Actual</th>
          <th className={`${th} text-right`}>Expected</th>
          <th className={`${th} text-right`}>{kind === 'under' ? 'Shortfall' : 'Excess'}</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-zinc-100">
        {rows.map((r, i) => (
          <motion.tr
            key={`${r.origin_iso3}-${r.dest_iso3}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: i * 0.03 }}
            className="hover:bg-zinc-50"
          >
            <td className={`${td} text-xs`}>
              {name(r.origin_iso3)} <span className="text-zinc-400">→</span> {name(r.dest_iso3)}
            </td>
            <td className={`${td} text-right text-xs tabular-nums`}>{fmtCompact(r.migrant_stock)}</td>
            <td className={`${td} text-right text-xs tabular-nums`}>{fmtCompact(r.expected)}</td>
            <td className={`${td} text-right text-xs font-medium tabular-nums`} style={{ color: kind === 'under' ? UNDER : OVER }}>
              {kind === 'under' ? `${fmtCompact(-r.gap)} · ${(1 / Math.max(r.ratio, 1e-4)).toFixed(0)}× less` : `${fmtCompact(r.gap)} · ${r.ratio.toFixed(0)}× more`}
            </td>
          </motion.tr>
        ))}
      </tbody>
    </Table>
  );
}

function Gaps({ onOpenBrief }: { onOpenBrief: (iso3: string) => void }) {
  const [iso, setIso] = useState('');
  const chartKey = useChartKey();
  const { data, error } = useSWR<PotentialResponse>(`/api/potential?limit=10${iso ? `&iso3=${iso}` : ''}`, fetcher, {
    keepPreviousData: true,
  });

  const scatter = useMemo(() => {
    const pts = data?.scatter ?? [];
    const group = (status: string, color: string, label: string) => ({
      label,
      data: pts
        .filter((p) => p.status === status)
        .map((p) => ({ x: Math.max(p.expected, 10), y: Math.max(p.migrant_stock, 10), o: p.origin_iso3, d: p.dest_iso3 })),
      backgroundColor: `${color}99`,
      borderColor: color,
      pointRadius: 2.5,
      pointHoverRadius: 6,
    });
    return {
      datasets: [
        group('over', OVER, 'Larger than expected'),
        group('under', UNDER, 'Smaller than expected'),
        group('as_expected', '#a1a1aa', 'Within 4× of expected'),
      ],
    };
  }, [data]);

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', align: 'end' },
      tooltip: {
        callbacks: {
          label: (ctx: any) =>
            ` ${getCountryGeo(ctx.raw.o).name} → ${getCountryGeo(ctx.raw.d).name}: ${fmtCompact(ctx.raw.y)} actual, ${fmtCompact(ctx.raw.x)} expected`,
        },
      },
    },
    scales: {
      x: { type: 'logarithmic', grid: lineGrid, title: { display: true, text: 'Expected migrant stock (gravity model)' }, ticks: { callback: (v: number) => ([1e2, 1e3, 1e4, 1e5, 1e6].includes(v) ? fmtCompact(v, 0) : '') } },
      y: { type: 'logarithmic', grid: lineGrid, title: { display: true, text: 'Actual migrant stock, 2020' }, ticks: { callback: (v: number) => ([1e2, 1e3, 1e4, 1e5, 1e6, 1e7].includes(v) ? fmtCompact(v, 0) : '') } },
    },
  };

  return (
    <Card
      delay={0.1}
      title="Corridor gaps: where diasporas are bigger or smaller than they should be"
      subtitle="A gravity model predicts each corridor from distance, borders, language and colonial history, given how much each country sends and receives overall. Large gaps point to barriers (labour-mobility and recognition agreements) or special channels (refugees, recruitment schemes, free movement)."
      actions={
        <select className={`${selectClass} w-56 py-1.5 text-xs`} value={iso} onChange={(e) => setIso(e.target.value)} aria-label="Filter by country">
          <option value="">All countries</option>
          {COUNTRY_LIST.map((c) => (
            <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>
          ))}
        </select>
      }
    >
      {error ? (
        <Empty>Corridor gaps not available. Run notebooks/09_corridor_potential.py.</Empty>
      ) : (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <div>
            <div className="h-[380px]">
              <Scatter key={`${chartKey}-${iso}`} data={scatter} options={options} />
            </div>
            {data && (
              <p className="mt-2 text-[11px] text-zinc-500">
                {fmtInt(data.counts.total)} corridors{iso ? ` involving ${getCountryGeo(iso).name}` : ''}: {fmtInt(data.counts.over)} at least
                4× larger than expected, {fmtInt(data.counts.under)} at least 4× smaller.
              </p>
            )}
          </div>
          <div className="space-y-5">
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide" style={{ color: UNDER }}>Largest shortfalls</h3>
              <GapTable rows={data?.under ?? []} kind="under" onOpenBrief={onOpenBrief} />
            </div>
            <div>
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide" style={{ color: OVER }}>Largest excesses</h3>
              <GapTable rows={data?.over ?? []} kind="over" onOpenBrief={onOpenBrief} />
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
