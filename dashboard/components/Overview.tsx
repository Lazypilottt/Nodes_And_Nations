import React, { useMemo, useState } from 'react';
import useSWR from 'swr';
import { getCountryGeo } from '../lib/countryGeo';
import { fetcher } from '../lib/fetcher';
import { ACCENT, fmtCompact, fmtInt, fmtPct } from '../lib/theme';
import type { Dataset, MapMode, NodeRow, Year } from '../lib/types';
import WorldMap, { type MapArc } from './WorldMap';
import { Card, Segmented, Stat, Table, td, th } from './ui';

const OUTBOUND = '#d97706';

const MODES: { value: MapMode; label: string }[] = [
  { value: 'corridors', label: 'Corridors' },
  { value: 'centrality', label: 'Centrality' },
  { value: 'communities', label: 'Communities' },
  { value: 'religion', label: 'Religion' },
];

interface CountryResponse {
  iso3: string;
  year: number;
  total_inbound: number;
  total_outbound: number;
  inbound: { iso3: string; stock: number }[];
  outbound: { iso3: string; stock: number }[];
}

export default function Overview({
  data,
  year,
  nodesByIso,
}: {
  data: Dataset;
  year: Year;
  nodesByIso: Record<string, NodeRow>;
}) {
  const [mode, setMode] = useState<MapMode>('corridors');
  const [selected, setSelected] = useState<string | null>(null);
  const [limit, setLimit] = useState(25);

  const { data: country } = useSWR<CountryResponse>(
    selected ? `/api/country?iso3=${selected}&year=${year}&limit=${limit}` : null,
    fetcher,
    { keepPreviousData: true }
  );

  const yearEdges = useMemo(() => data.edges.filter((e) => e.year === year), [data.edges, year]);

  const arcs: MapArc[] = useMemo(() => {
    if (mode !== 'corridors') return [];
    if (selected && country?.iso3 === selected) {
      return [
        ...country.inbound.map((c) => ({ origin: c.iso3, dest: selected, value: c.stock, color: ACCENT })),
        ...country.outbound.map((c) => ({ origin: selected, dest: c.iso3, value: c.stock, color: OUTBOUND })),
      ];
    }
    return yearEdges.slice(0, limit).map((e) => ({ origin: e.origin_iso3, dest: e.dest_iso3, value: e.weight }));
  }, [mode, selected, country, yearEdges, limit]);

  const dist = data.distributions[year];
  const ranked = useMemo(() => Object.values(nodesByIso), [nodesByIso]);
  const topDest = ranked.reduce<NodeRow | null>((a, n) => ((n.in_strength ?? 0) > (a?.in_strength ?? -1) ? n : a), null);
  const topOrigin = ranked.reduce<NodeRow | null>((a, n) => ((n.out_strength ?? 0) > (a?.out_strength ?? -1) ? n : a), null);

  return (
    <div className="space-y-6">
      <Card
        title="Migration map"
        subtitle={
          mode === 'corridors'
            ? selected
              ? `Largest corridors into and out of ${getCountryGeo(selected).name}, ${year}`
              : `Top ${limit} bilateral corridors by migrant stock, ${year}. Click a country to explore it.`
            : `${MODES.find((m) => m.value === mode)?.label} view, ${year}`
        }
        actions={
          <div className="flex flex-wrap items-center gap-2">
            {mode === 'corridors' && (
              <Segmented
                ariaLabel="Number of corridors"
                size="xs"
                value={limit}
                onChange={setLimit}
                options={[10, 25, 50].map((n) => ({ value: n, label: n }))}
              />
            )}
            <Segmented ariaLabel="Map layer" value={mode} onChange={setMode} options={MODES} />
          </div>
        }
        bodyClassName="p-3"
      >
        <WorldMap
          mode={mode}
          nodes={nodesByIso}
          arcs={arcs}
          selected={selected}
          onSelect={setSelected}
          className="h-[360px] sm:h-[480px]"
        />
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {selected ? (
          <CountryProfile
            iso3={selected}
            node={nodesByIso[selected]}
            country={country?.iso3 === selected ? country : undefined}
            onClose={() => setSelected(null)}
            rank={
              ranked
                .slice()
                .sort((a, b) => (b.pagerank ?? 0) - (a.pagerank ?? 0))
                .findIndex((n) => n.iso3 === selected) + 1
            }
          />
        ) : (
          <Card
            className="lg:col-span-2"
            title={`Largest corridors, ${year}`}
            subtitle="Bilateral migrant stock (people born in the origin, living in the destination)"
            bodyClassName="px-2 pb-2"
          >
            <Table>
              <thead>
                <tr>
                  <th className={th}>#</th>
                  <th className={th}>Origin</th>
                  <th className={th}>Destination</th>
                  <th className={`${th} text-right`}>Migrant stock</th>
                  <th className={th}>Type</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {yearEdges.slice(0, 10).map((e, i) => {
                  const o = getCountryGeo(e.origin_iso3);
                  const d = getCountryGeo(e.dest_iso3);
                  return (
                    <tr key={`${e.origin_iso3}-${e.dest_iso3}`} className="hover:bg-zinc-50">
                      <td className={`${td} tabular-nums text-zinc-400`}>{i + 1}</td>
                      <td className={td}>
                        <button className="hover:text-blue-600" onClick={() => setSelected(e.origin_iso3)}>
                          {o.flag} {o.name}
                        </button>
                      </td>
                      <td className={td}>
                        <button className="hover:text-blue-600" onClick={() => setSelected(e.dest_iso3)}>
                          {d.flag} {d.name}
                        </button>
                      </td>
                      <td className={`${td} text-right font-medium tabular-nums text-zinc-900`}>{fmtInt(e.weight)}</td>
                      <td className={`${td} text-xs text-zinc-500`}>
                        {e.same_continent === null ? '—' : e.same_continent ? 'Intra-continental' : 'Inter-continental'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
          </Card>
        )}

        <Card title="At a glance" subtitle={`All ${fmtInt(data.network_summary.find((s) => s.year === year)?.n_edges)} corridors, ${year}`}>
          <div className="grid grid-cols-2 gap-x-4 gap-y-5">
            <Stat label="High-income share" value={fmtPct(dist?.high_income_share)} hint="of migrants live in high-income states" />
            <Stat label="Intra-continental" value={fmtPct(dist?.intra_continental_share)} hint="of migrants stay on their continent" />
            <Stat
              label="Top destination"
              value={topDest ? `${getCountryGeo(topDest.iso3).flag} ${topDest.country_name ?? topDest.iso3}` : '—'}
              hint={`${fmtCompact(topDest?.in_strength)} immigrants`}
            />
            <Stat
              label="Top origin"
              value={topOrigin ? `${getCountryGeo(topOrigin.iso3).flag} ${topOrigin.country_name ?? topOrigin.iso3}` : '—'}
              hint={`${fmtCompact(topOrigin?.out_strength)} emigrants`}
            />
          </div>
        </Card>
      </div>
    </div>
  );
}

function CountryProfile({
  iso3,
  node,
  country,
  rank,
  onClose,
}: {
  iso3: string;
  node?: NodeRow;
  country?: CountryResponse;
  rank: number;
  onClose: () => void;
}) {
  const geo = getCountryGeo(iso3);
  const list = (rows: { iso3: string; stock: number }[] | undefined, color: string) => (
    <ul className="space-y-1.5">
      {(rows ?? []).slice(0, 6).map((r) => (
        <li key={r.iso3} className="flex items-center justify-between gap-3 text-sm">
          <span className="truncate text-zinc-700">
            {getCountryGeo(r.iso3).flag} {getCountryGeo(r.iso3).name}
          </span>
          <span className="tabular-nums text-zinc-900" style={{ color }}>{fmtCompact(r.stock)}</span>
        </li>
      ))}
      {!rows && <li className="text-sm text-zinc-400">Loading…</li>}
      {rows?.length === 0 && <li className="text-sm text-zinc-400">No recorded corridors</li>}
    </ul>
  );

  return (
    <Card
      className="lg:col-span-2"
      title={
        <span className="flex items-center gap-2">
          <span className="text-lg">{geo.flag}</span>
          {node?.country_name ?? geo.name}
        </span>
      }
      subtitle={[node?.un_region, node?.income_group, node?.community_label].filter(Boolean).join(' · ')}
      actions={
        <button onClick={onClose} className="rounded-md px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800">
          Clear selection
        </button>
      }
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-4">
        <Stat label="Immigrants" value={fmtCompact(country?.total_inbound ?? node?.in_strength)} tone="accent" />
        <Stat label="Emigrants" value={fmtCompact(country?.total_outbound ?? node?.out_strength)} />
        <Stat label="PageRank rank" value={rank > 0 ? `#${rank}` : '—'} />
        <Stat label="GDP per capita" value={node?.gdp_per_capita ? `$${fmtInt(node.gdp_per_capita)}` : '—'} />
      </div>
      <div className="mt-6 grid grid-cols-1 gap-6 sm:grid-cols-2">
        <div>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Where immigrants come from</h3>
          {list(country?.inbound, ACCENT)}
        </div>
        <div>
          <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">Where emigrants live</h3>
          {list(country?.outbound, OUTBOUND)}
        </div>
      </div>
    </Card>
  );
}
