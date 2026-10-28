import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { COUNTRY_LIST, getCountryGeo } from '../lib/countryGeo';
import {
  INCOME_COLORS, MIPEX_COLORS, MIPEX_TIERS, RELIGION_COLORS, STRAIN_COLORS, categorical, fmtCompact, fmtInt, fmtPct,
  type StrainLevel,
} from '../lib/theme';
import type { NodeRow } from '../lib/types';
import WorldMap, { type SimulationOverlay } from './WorldMap';
import { Card, Field, Segmented, ShareBars, Stat, Table, selectClass, td, th } from './ui';

type Scenario = 'conflict' | 'climate' | 'cultural_kinship' | 'visa' | 'economic';
type Policy = 'universal' | 'kinship_first' | 'cultural_kinship' | 'integration_havens' | 'strict_border';

interface Params {
  epicenter: string;
  scenario: Scenario;
  intensity: number;
  displaced: number;
  policy: Policy;
}

interface Recipient {
  iso3: string;
  country_name: string;
  dist_km: number;
  predicted_influx: number;
  share: number;
  strain_per_1k: number;
  strain_level: StrainLevel;
}

interface SimResult {
  params: { epicenter_iso3: string };
  reference_year: number;
  summary: { total_displaced: number; mean_distance_km: number; top5_share: number; critical_states: number };
  top_recipients: Recipient[];
  breakdowns: Record<'continent' | 'income' | 'religion' | 'mipex', { label: string; value: number }[]>;
}

const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: 'conflict', label: 'Armed conflict' },
  { value: 'climate', label: 'Climate displacement' },
  { value: 'cultural_kinship', label: 'Cultural & religious kinship' },
  { value: 'visa', label: 'Visa / border liberalisation' },
  { value: 'economic', label: 'Economic collapse' },
];

const POLICIES: { value: Policy; label: string; hint: string }[] = [
  { value: 'universal', label: 'Universal', hint: 'No additional routing preference' },
  { value: 'kinship_first', label: 'Diaspora', hint: 'Favours states with an existing diaspora' },
  { value: 'cultural_kinship', label: 'Religious kin', hint: 'Favours states sharing the dominant religion' },
  { value: 'integration_havens', label: 'Integration', hint: 'Favours states with open integration policy' },
  { value: 'strict_border', label: 'Strict', hint: 'Non-neighbouring states admit 70% fewer people' },
];

const PRESETS: (Params & { id: string; title: string })[] = [
  { id: 'ukr', title: 'Ukraine 2022', epicenter: 'UKR', scenario: 'conflict', intensity: 95, displaced: 6_500_000, policy: 'universal' },
  { id: 'syr', title: 'Syria 2015', epicenter: 'SYR', scenario: 'conflict', intensity: 90, displaced: 5_200_000, policy: 'cultural_kinship' },
  { id: 'sdn', title: 'Sudan 2023', epicenter: 'SDN', scenario: 'conflict', intensity: 85, displaced: 3_400_000, policy: 'universal' },
  { id: 'mmr', title: 'Myanmar 2017', epicenter: 'MMR', scenario: 'cultural_kinship', intensity: 90, displaced: 1_200_000, policy: 'cultural_kinship' },
  { id: 'ven', title: 'Venezuela', epicenter: 'VEN', scenario: 'economic', intensity: 80, displaced: 4_200_000, policy: 'universal' },
  { id: 'bgd', title: 'Bangladesh coast', epicenter: 'BGD', scenario: 'climate', intensity: 75, displaced: 2_500_000, policy: 'integration_havens' },
];

const BREAKDOWNS = [
  { value: 'continent', label: 'Continent' },
  { value: 'income', label: 'Income' },
  { value: 'religion', label: 'Religion' },
  { value: 'mipex', label: 'Integration' },
] as const;

const sameParams = (a: Params, b: Params | null) =>
  !!b && a.epicenter === b.epicenter && a.scenario === b.scenario && a.intensity === b.intensity &&
  a.displaced === b.displaced && a.policy === b.policy;

export default function CrisisSimulator({ nodesByIso }: { nodesByIso: Record<string, NodeRow> }) {
  const [params, setParams] = useState<Params>(PRESETS[0]);
  const [lastRun, setLastRun] = useState<Params | null>(null);
  const [result, setResult] = useState<SimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [breakdown, setBreakdown] = useState<(typeof BREAKDOWNS)[number]['value']>('continent');
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (p: Params) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({
          epicenter_iso3: p.epicenter,
          scenario: p.scenario,
          intensity: p.intensity,
          displaced_scale: p.displaced,
          border_policy: p.policy,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error ?? 'Simulation failed');
      setResult(body);
      setLastRun(p);
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message ?? 'Simulation failed');
    } finally {
      if (abortRef.current === ctrl) setLoading(false);
    }
  }, []);

  useEffect(() => {
    run(PRESETS[0]);
    return () => abortRef.current?.abort();
  }, [run]);

  const update = (patch: Partial<Params>) => setParams((p) => ({ ...p, ...patch }));
  const loadPreset = (p: Params) => {
    setParams(p);
    run(p);
  };
  const dirty = !sameParams(params, lastRun);

  const overlay: SimulationOverlay | null = useMemo(() => {
    if (!result) return null;
    const recipients: SimulationOverlay['recipients'] = {};
    for (const r of result.top_recipients) recipients[r.iso3] = { influx: r.predicted_influx, level: r.strain_level };
    return { epicenter: result.params.epicenter_iso3, recipients };
  }, [result]);

  const arcs = useMemo(
    () =>
      (result?.top_recipients ?? []).slice(0, 15).map((r) => ({
        origin: result!.params.epicenter_iso3,
        dest: r.iso3,
        value: r.predicted_influx,
        color: STRAIN_COLORS[r.strain_level],
      })),
    [result]
  );

  const colorFor = (label: string, i: number) => {
    if (breakdown === 'income') return INCOME_COLORS[label] ?? '#cbd5e1';
    if (breakdown === 'religion') return RELIGION_COLORS[label] ?? '#cbd5e1';
    if (breakdown === 'mipex') return MIPEX_COLORS[MIPEX_TIERS.indexOf(label)] ?? '#cbd5e1';
    return categorical(i);
  };

  const ranEpicenter = result ? getCountryGeo(result.params.epicenter_iso3) : null;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
      <div className="space-y-6 lg:col-span-7">
        <Card
          title={ranEpicenter ? `${ranEpicenter.flag} ${ranEpicenter.name}: projected destinations` : 'Projected destinations'}
          subtitle="Click any country to make it the epicenter, then run the simulation."
          bodyClassName="p-3"
        >
          <WorldMap
            mode="simulation"
            nodes={nodesByIso}
            arcs={arcs}
            simulation={overlay}
            selected={params.epicenter}
            onSelect={(iso) => iso && update({ epicenter: iso })}
            className="h-[340px] sm:h-[440px]"
          />
        </Card>

        <Card>
          <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-4">
            <Stat label="Displaced" value={fmtCompact(result?.summary.total_displaced)} />
            <Stat
              label="Main destination"
              value={result?.top_recipients[0] ? `${getCountryGeo(result.top_recipients[0].iso3).flag} ${result.top_recipients[0].country_name}` : '—'}
              hint={result?.top_recipients[0] ? `${fmtCompact(result.top_recipients[0].predicted_influx)} people` : undefined}
              tone="accent"
            />
            <Stat label="Top-5 concentration" value={fmtPct(result?.summary.top5_share)} hint="share absorbed by five states" />
            <Stat
              label="Critical strain"
              value={result ? result.summary.critical_states : '—'}
              hint={result ? `mean distance ${fmtInt(result.summary.mean_distance_km)} km` : undefined}
              tone={result && result.summary.critical_states > 0 ? 'danger' : 'default'}
            />
          </div>
        </Card>
      </div>

      <div className="space-y-6 lg:col-span-5">
        <Card title="Scenario" subtitle="Gravity-routing model: distance decay × diaspora × income × affinity × policy">
          <div className="mb-5 flex flex-wrap gap-1.5">
            {PRESETS.map((p) => {
              const active = sameParams(p, params);
              return (
                <button
                  key={p.id}
                  onClick={() => loadPreset(p)}
                  className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                    active ? 'border-blue-600 bg-blue-50 text-blue-700' : 'border-zinc-200 text-zinc-600 hover:border-zinc-300 hover:text-zinc-900'
                  }`}
                >
                  {p.title}
                </button>
              );
            })}
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field label="Epicenter">
              <select className={selectClass} value={params.epicenter} onChange={(e) => update({ epicenter: e.target.value })}>
                {COUNTRY_LIST.map((c) => (
                  <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>
                ))}
              </select>
            </Field>
            <Field label="Crisis type">
              <select className={selectClass} value={params.scenario} onChange={(e) => update({ scenario: e.target.value as Scenario })}>
                {SCENARIOS.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </Field>
            <Field label="People displaced" hint={fmtCompact(params.displaced, 2)}>
              <input
                type="range" min={100_000} max={10_000_000} step={100_000}
                value={params.displaced}
                onChange={(e) => update({ displaced: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </Field>
            <Field label="Severity" hint={`${params.intensity}%`}>
              <input
                type="range" min={10} max={100} step={5}
                value={params.intensity}
                onChange={(e) => update({ intensity: Number(e.target.value) })}
                className="w-full accent-blue-600"
              />
            </Field>
          </div>
          <p className="mt-2 text-[11px] text-zinc-500">Higher severity concentrates displacement in nearby states.</p>

          <div className="mt-5">
            <div className="mb-1.5 text-xs font-medium text-zinc-600">Receiving-state policy</div>
            <Segmented size="xs" value={params.policy} onChange={(v) => update({ policy: v })} options={POLICIES} ariaLabel="Border policy" />
            <p className="mt-1.5 text-[11px] text-zinc-500">{POLICIES.find((p) => p.value === params.policy)?.hint}</p>
          </div>

          <div className="mt-5 flex items-center justify-between gap-3">
            <span className="text-xs text-zinc-500">
              {error ? <span className="text-red-600">{error}</span> : dirty ? 'Parameters changed' : result ? `Reference year ${result.reference_year}` : ''}
            </span>
            <button
              onClick={() => run(params)}
              disabled={loading || !dirty}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-500"
            >
              {loading && <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />}
              {loading ? 'Running…' : 'Run simulation'}
            </button>
          </div>
        </Card>

        <Card
          title="Where people go"
          actions={<Segmented size="xs" value={breakdown} onChange={setBreakdown} options={BREAKDOWNS} ariaLabel="Breakdown" />}
        >
          {result ? (
            <ShareBars data={result.breakdowns[breakdown]} colorFor={colorFor} format={fmtCompact} limit={6} />
          ) : (
            <div className="h-40 animate-pulse rounded-lg bg-zinc-100" />
          )}
        </Card>

        <Card title="Top receiving states" bodyClassName="px-2 pb-2">
          <Table>
            <thead>
              <tr>
                <th className={th}>State</th>
                <th className={`${th} text-right`}>Arrivals</th>
                <th className={`${th} text-right`}>Per 1k residents</th>
                <th className={th}>Strain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {(result?.top_recipients ?? []).slice(0, 8).map((r) => (
                <tr key={r.iso3}>
                  <td className={`${td} max-w-[160px] truncate`}>{getCountryGeo(r.iso3).flag} {r.country_name}</td>
                  <td className={`${td} text-right font-medium tabular-nums text-zinc-900`}>{fmtInt(r.predicted_influx)}</td>
                  <td className={`${td} text-right tabular-nums`}>{r.strain_per_1k.toFixed(1)}</td>
                  <td className={td}>
                    <span className="inline-flex items-center gap-1.5 text-xs" style={{ color: STRAIN_COLORS[r.strain_level] }}>
                      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: STRAIN_COLORS[r.strain_level] }} />
                      {r.strain_level}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card>
      </div>
    </div>
  );
}
