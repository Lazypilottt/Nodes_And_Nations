import { AnimatePresence, motion } from 'framer-motion';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { COUNTRY_LIST, getCountryGeo } from '../lib/countryGeo';
import { INCOME_COLORS, STRAIN_COLORS, categorical, fmtCompact, fmtInt, fmtPct, type StrainLevel } from '../lib/theme';
import type { BacktestRow, Dataset, NodeRow } from '../lib/types';
import { Card, CountUp, Empty, Field, Segmented, ShareBars, Stat, selectClass } from './ui';
import WorldMap, { type SimulationOverlay } from './WorldMap';

type Engine = 'model' | 'scenario';
type Scenario = 'conflict' | 'climate' | 'cultural_kinship' | 'visa' | 'economic';
type Policy = 'universal' | 'kinship_first' | 'cultural_kinship' | 'integration_havens' | 'strict_border';

interface Params {
  engine: Engine;
  epicenter: string;
  displaced: number;
  closed: string[];
  scenario: Scenario;
  intensity: number;
  policy: Policy;
}

interface Recipient {
  iso3: string;
  country_name: string;
  continent: string | null;
  dist_km: number;
  ties: string[];
  existing_diaspora: number;
  predicted_influx: number;
  share: number;
  strain_per_1k: number;
  strain_level: StrainLevel;
  current_refugees: number | null;
  vs_current_refugees: number | null;
}

interface SimResult {
  params: { epicenter_iso3: string; engine: Engine };
  reference_year: number;
  summary: { total_displaced: number; mean_distance_km: number; top5_share: number; critical_states: number };
  top_recipients: Recipient[];
  n_destinations: number;
  breakdowns: Record<'continent' | 'income', { label: string; value: number }[]>;
}

const PRESETS: { id: string; title: string; epicenter: string; displaced: number; closed?: string[] }[] = [
  { id: 'sdn', title: 'Sudan', epicenter: 'SDN', displaced: 3_000_000 },
  { id: 'ukr', title: 'Ukraine', epicenter: 'UKR', displaced: 6_500_000, closed: ['RUS', 'BLR'] },
  { id: 'hti', title: 'Haiti', epicenter: 'HTI', displaced: 600_000 },
  { id: 'mmr', title: 'Myanmar', epicenter: 'MMR', displaced: 1_200_000 },
  { id: 'afg', title: 'Afghanistan', epicenter: 'AFG', displaced: 2_000_000 },
  { id: 'ven', title: 'Venezuela', epicenter: 'VEN', displaced: 4_000_000 },
  { id: 'twn', title: 'Taiwan (hypothetical)', epicenter: 'TWN', displaced: 1_000_000, closed: ['CHN'] },
];

const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: 'conflict', label: 'Armed conflict' },
  { value: 'climate', label: 'Climate displacement' },
  { value: 'cultural_kinship', label: 'Cultural & religious kinship' },
  { value: 'visa', label: 'Visa / border liberalisation' },
  { value: 'economic', label: 'Economic collapse' },
];

const POLICIES: { value: Policy; label: string }[] = [
  { value: 'universal', label: 'Universal' },
  { value: 'kinship_first', label: 'Diaspora' },
  { value: 'cultural_kinship', label: 'Cultural kin' },
  { value: 'integration_havens', label: 'Integration' },
  { value: 'strict_border', label: 'Strict' },
];

const METHOD_LABELS: Record<BacktestRow['method'], string> = {
  model: 'This model',
  diaspora_only: 'Follow the diaspora',
  distance_only: 'Nearest countries',
};

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

export default function DisplacementPlanner({ data, nodesByIso }: { data: Dataset; nodesByIso: Record<string, NodeRow> }) {
  const [params, setParams] = useState<Params>({
    engine: 'model',
    epicenter: 'SDN',
    displaced: 3_000_000,
    closed: [],
    scenario: 'conflict',
    intensity: 80,
    policy: 'universal',
  });
  const [result, setResult] = useState<SimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [breakdown, setBreakdown] = useState<'continent' | 'income'>('continent');
  const abortRef = useRef<AbortController | null>(null);
  const debounced = useDebounced(params, 250);

  useEffect(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setLoading(true);
    setError(null);
    fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: ctrl.signal,
      body: JSON.stringify({
        engine: debounced.engine,
        epicenter_iso3: debounced.epicenter,
        displaced_scale: debounced.displaced,
        closed: debounced.closed,
        scenario: debounced.scenario,
        intensity: debounced.intensity,
        border_policy: debounced.policy,
      }),
    })
      .then(async (r) => {
        const body = await r.json();
        if (!r.ok) throw new Error(body?.error ?? 'Forecast failed');
        setResult(body);
      })
      .catch((e) => {
        if (e?.name !== 'AbortError') setError(e?.message ?? 'Forecast failed');
      })
      .finally(() => {
        if (abortRef.current === ctrl) setLoading(false);
      });
    return () => ctrl.abort();
  }, [debounced]);

  const update = (patch: Partial<Params>) => setParams((p) => ({ ...p, ...patch }));
  const toggleClosed = (iso: string) =>
    setParams((p) => ({ ...p, closed: p.closed.includes(iso) ? p.closed.filter((c) => c !== iso) : [...p.closed, iso] }));

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

  const epi = getCountryGeo(params.epicenter);
  const top = result?.top_recipients[0];

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-blue-500/20 bg-gradient-to-r from-blue-600/[0.07] via-transparent to-teal-500/[0.06] px-5 py-4"
      >
        <h1 className="text-base font-semibold text-zinc-900">Where will people go when a crisis forces them out?</h1>
        <p className="mt-1 max-w-3xl text-sm text-zinc-600">
          A forecast for humanitarian agencies, host governments and donors in the first weeks of a crisis, when aid must be
          pre-positioned before arrivals are counted. The model is fitted on 25 years of UNHCR data and back-tested on seven
          past crises. Close a border to see where people would go instead.
        </p>
      </motion.div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-7">
          <Card
            title={`${epi.flag} ${epi.name}: projected destinations`}
            subtitle="Click any country to make it the epicentre."
            actions={
              <AnimatePresence>
                {loading && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center gap-1.5 text-[11px] text-zinc-500"
                  >
                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-zinc-300 border-t-blue-600" />
                    Updating
                  </motion.span>
                )}
              </AnimatePresence>
            }
            bodyClassName="p-3"
          >
            <WorldMap
              mode="simulation"
              nodes={nodesByIso}
              arcs={arcs}
              simulation={overlay}
              closed={params.closed}
              selected={params.epicenter}
              onSelect={(iso) => iso && update({ epicenter: iso, closed: params.closed.filter((c) => c !== iso) })}
              className="h-[340px] sm:h-[440px]"
            />
          </Card>

          <Card delay={0.05}>
            <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-4">
              <Stat label="Displaced" value={<CountUp value={result?.summary.total_displaced} format={(n) => fmtCompact(n, 2)} />} />
              <Stat
                label="Main destination"
                value={top ? `${getCountryGeo(top.iso3).flag} ${top.country_name}` : '—'}
                hint={top ? `${fmtCompact(top.predicted_influx)} people · ${fmtPct(top.share)}` : undefined}
                tone="accent"
              />
              <Stat
                label="Top-5 concentration"
                value={<CountUp value={result ? result.summary.top5_share * 100 : null} format={(n) => `${Math.round(n)}%`} />}
                hint="share absorbed by five states"
              />
              <Stat
                label="States at critical strain"
                value={<CountUp value={result?.summary.critical_states} format={(n) => fmtInt(n)} />}
                hint="≥ 12 arrivals per 1,000 residents"
                tone={result && result.summary.critical_states > 0 ? 'danger' : 'default'}
              />
            </div>
          </Card>

          <Card
            delay={0.1}
            title="Arrivals and pressure by destination"
            subtitle="Pressure is arrivals per 1,000 residents. “× hosted” compares arrivals with the refugees each country already hosts."
            bodyClassName="px-5 pb-5 pt-3"
          >
            {error ? (
              <Empty>{error}</Empty>
            ) : result ? (
              <ArrivalBars rows={result.top_recipients.slice(0, 12)} />
            ) : (
              <div className="h-64 animate-pulse rounded-lg bg-zinc-100" />
            )}
          </Card>
        </div>

        <div className="space-y-6 lg:col-span-5">
          <Card title="Crisis" delay={0.05}>
            <div className="mb-4">
              <Segmented
                ariaLabel="Forecast engine"
                value={params.engine}
                onChange={(engine) => update({ engine })}
                options={[
                  { value: 'model', label: 'Evidence-based forecast' },
                  { value: 'scenario', label: 'What-if scenario' },
                ]}
              />
              <p className="mt-1.5 text-[11px] text-zinc-500">
                {params.engine === 'model'
                  ? 'Fitted on UNHCR refugee and asylum data, 2000–2024, and back-tested on real crises.'
                  : 'Hand-tuned routing for exploring crisis types and reception policies. Not validated.'}
              </p>
            </div>

            <div className="mb-4 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => {
                const active = params.epicenter === p.epicenter;
                return (
                  <motion.button
                    key={p.id}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => update({ epicenter: p.epicenter, displaced: p.displaced, closed: p.closed ?? [] })}
                    className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                      active
                        ? 'border-blue-600 bg-blue-600/10 text-blue-700 dark:text-blue-300'
                        : 'border-zinc-200 text-zinc-600 hover:border-zinc-300 hover:text-zinc-900'
                    }`}
                  >
                    {p.title}
                  </motion.button>
                );
              })}
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Field label="Epicentre">
                <select
                  className={selectClass}
                  value={params.epicenter}
                  onChange={(e) => update({ epicenter: e.target.value, closed: params.closed.filter((c) => c !== e.target.value) })}
                >
                  {COUNTRY_LIST.map((c) => (
                    <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>
                  ))}
                </select>
              </Field>
              <Field label="People displaced" hint={fmtCompact(params.displaced, 2)}>
                <input
                  type="range" min={50_000} max={10_000_000} step={50_000}
                  value={params.displaced}
                  onChange={(e) => update({ displaced: Number(e.target.value) })}
                  className="mt-2 w-full accent-blue-600"
                />
              </Field>
            </div>

            <div className="mt-4">
              <Field label="Closed borders">
                <select
                  className={selectClass}
                  value=""
                  onChange={(e) => e.target.value && toggleClosed(e.target.value)}
                >
                  <option value="">Close a country’s border…</option>
                  {(result?.top_recipients ?? []).slice(0, 15).map((r) => (
                    <option key={r.iso3} value={r.iso3}>{getCountryGeo(r.iso3).flag} {r.country_name} (projected {fmtCompact(r.predicted_influx)})</option>
                  ))}
                  <option disabled>──────────</option>
                  {COUNTRY_LIST.filter((c) => c.iso3 !== params.epicenter && !params.closed.includes(c.iso3)).map((c) => (
                    <option key={c.iso3} value={c.iso3}>{c.flag} {c.name}</option>
                  ))}
                </select>
              </Field>
              <div className="mt-2 flex min-h-[1.75rem] flex-wrap gap-1.5">
                <AnimatePresence initial={false}>
                  {params.closed.map((iso) => (
                    <motion.button
                      key={iso}
                      layout
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      onClick={() => toggleClosed(iso)}
                      className="inline-flex items-center gap-1 rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-700 hover:bg-zinc-200"
                      aria-label={`Reopen ${getCountryGeo(iso).name}`}
                    >
                      {getCountryGeo(iso).flag} {getCountryGeo(iso).name}
                      <span className="text-zinc-400">✕</span>
                    </motion.button>
                  ))}
                </AnimatePresence>
                {params.closed.length === 0 && <span className="text-[11px] text-zinc-400">All borders open</span>}
              </div>
            </div>

            <AnimatePresence initial={false}>
              {params.engine === 'scenario' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-4 grid grid-cols-1 gap-4 border-t border-zinc-100 pt-4 sm:grid-cols-2">
                    <Field label="Crisis type">
                      <select className={selectClass} value={params.scenario} onChange={(e) => update({ scenario: e.target.value as Scenario })}>
                        {SCENARIOS.map((s) => (
                          <option key={s.value} value={s.value}>{s.label}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Severity" hint={`${params.intensity}%`}>
                      <input
                        type="range" min={10} max={100} step={5}
                        value={params.intensity}
                        onChange={(e) => update({ intensity: Number(e.target.value) })}
                        className="mt-2 w-full accent-blue-600"
                      />
                    </Field>
                  </div>
                  <div className="mt-4">
                    <div className="mb-1.5 text-xs font-medium text-zinc-600">Receiving-state policy</div>
                    <Segmented size="xs" value={params.policy} onChange={(v) => update({ policy: v })} options={POLICIES} ariaLabel="Border policy" />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>

          <Backtest data={data} epicenter={params.epicenter} />

          <Card
            delay={0.15}
            title="Where people go"
            actions={
              <Segmented
                size="xs"
                value={breakdown}
                onChange={setBreakdown}
                options={[
                  { value: 'continent', label: 'Continent' },
                  { value: 'income', label: 'Income' },
                ]}
                ariaLabel="Breakdown"
              />
            }
          >
            {result ? (
              <ShareBars
                data={result.breakdowns[breakdown]}
                colorFor={(label, i) => (breakdown === 'income' ? INCOME_COLORS[label] ?? '#a1a1aa' : categorical(i))}
                format={fmtCompact}
                limit={6}
              />
            ) : (
              <div className="h-40 animate-pulse rounded-lg bg-zinc-100" />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function ArrivalBars({ rows }: { rows: Recipient[] }) {
  const max = Math.max(...rows.map((r) => r.predicted_influx), 1);
  return (
    <ul className="space-y-2">
      <AnimatePresence initial={false}>
        {rows.map((r, i) => (
          <motion.li
            key={r.iso3}
            layout
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35, delay: i * 0.025 }}
            className="grid grid-cols-[minmax(0,7rem)_1fr_auto] items-center gap-3 text-xs sm:grid-cols-[minmax(0,10rem)_1fr_auto]"
          >
            <div className="min-w-0">
              <div className="truncate font-medium text-zinc-800">{getCountryGeo(r.iso3).flag} {r.country_name}</div>
              <div className="truncate text-[10px] text-zinc-400">{r.ties.length ? r.ties.join(' · ') : `${fmtInt(r.dist_km)} km`}</div>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-zinc-100">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: STRAIN_COLORS[r.strain_level] }}
                initial={{ width: 0 }}
                animate={{ width: `${(r.predicted_influx / max) * 100}%` }}
                transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
            <div className="text-right tabular-nums sm:w-44">
              <span className="font-semibold text-zinc-900">{fmtCompact(r.predicted_influx)}</span>
              <span className="ml-1.5 text-zinc-500">{r.strain_per_1k.toFixed(1)}/1k</span>
              {r.vs_current_refugees != null && r.vs_current_refugees >= 0.1 && (
                <span className="ml-1.5 hidden text-zinc-400 sm:inline">×{r.vs_current_refugees.toFixed(1)} hosted</span>
              )}
            </div>
          </motion.li>
        ))}
      </AnimatePresence>
      <li className="flex flex-wrap gap-3 pt-2 text-[11px] text-zinc-500">
        {(Object.keys(STRAIN_COLORS) as StrainLevel[]).map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: STRAIN_COLORS[k] }} />
            {k}
          </span>
        ))}
      </li>
    </ul>
  );
}

function Backtest({ data, epicenter }: { data: Dataset; epicenter: string }) {
  const bt = data.displacement.backtest;
  const crises = useMemo(() => Array.from(new Set(bt.map((r) => r.origin_iso3))), [bt]);
  const [chosen, setChosen] = useState<string>(crises.includes(epicenter) ? epicenter : crises[0] ?? '');
  useEffect(() => {
    if (crises.includes(epicenter)) setChosen(epicenter);
  }, [epicenter, crises]);

  if (bt.length === 0) {
    return (
      <Card title="How reliable is this?">
        <Empty>Run notebooks/08_displacement_forecast.py to back-test the model.</Empty>
      </Card>
    );
  }

  const mean = (m: BacktestRow['method']) => {
    const rows = bt.filter((r) => r.method === m);
    return rows.reduce((a, r) => a + Number(r.overlap), 0) / Math.max(1, rows.length);
  };
  const methods: BacktestRow['method'][] = ['model', 'diaspora_only', 'distance_only'];
  const crisisRows = bt.filter((r) => r.origin_iso3 === chosen);
  const crisis = crisisRows[0];
  const shares = data.displacement.backtest_shares
    .filter((s) => s.origin_iso3 === chosen)
    .sort((a, b) => b.actual_share - a.actual_share)
    .slice(0, 6);
  const maxShare = Math.max(...shares.flatMap((s) => [s.actual_share, s.model_share]), 0.01);

  return (
    <Card
      delay={0.1}
      title="How reliable is this?"
      subtitle="Back-tests: trained only on years before each crisis began, then compared with where people actually went."
    >
      <div className="space-y-2">
        {methods.map((m, i) => (
          <div key={m} className="text-xs">
            <div className="mb-1 flex justify-between">
              <span className={m === 'model' ? 'font-medium text-zinc-900' : 'text-zinc-600'}>{METHOD_LABELS[m]}</span>
              <span className="tabular-nums text-zinc-900">{fmtPct(mean(m))} placed correctly</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-zinc-100">
              <motion.div
                className={`h-full rounded-full ${m === 'model' ? 'bg-blue-600 dark:bg-blue-400' : 'bg-zinc-400'}`}
                initial={{ width: 0 }}
                animate={{ width: `${mean(m) * 100}%` }}
                transition={{ duration: 0.8, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-zinc-500">
        Average over {crises.length} crises of the share of displaced people assigned to the right country.
      </p>

      <div className="mt-5 border-t border-zinc-100 pt-4">
        <div className="mb-3 flex items-center justify-between gap-2">
          <span className="text-xs font-medium text-zinc-700">Back-test detail</span>
          <select className={`${selectClass} w-auto py-1 text-xs`} value={chosen} onChange={(e) => setChosen(e.target.value)}>
            {crises.map((c) => (
              <option key={c} value={c}>{bt.find((r) => r.origin_iso3 === c)?.crisis}</option>
            ))}
          </select>
        </div>
        {crisis && (
          <p className="mb-3 text-[11px] text-zinc-500">
            Trained on {crisis.train_years}, compared with {crisis.eval_year} ({fmtCompact(crisis.displaced_eval_year)} displaced).
            Model: {fmtPct(crisisRows.find((r) => r.method === 'model')?.overlap)} placed correctly, top-5 hits{' '}
            {crisisRows.find((r) => r.method === 'model')?.top5_hits}/5.
          </p>
        )}
        <ul className="space-y-2.5">
          {shares.map((s) => (
            <li key={s.dest_iso3} className="text-xs">
              <div className="mb-1 flex justify-between">
                <span className="text-zinc-700">{getCountryGeo(s.dest_iso3).flag} {getCountryGeo(s.dest_iso3).name}</span>
                <span className="tabular-nums text-zinc-500">
                  actual <span className="font-medium text-zinc-900">{fmtPct(s.actual_share)}</span> · forecast{' '}
                  <span className="font-medium text-blue-600 dark:text-blue-400">{fmtPct(s.model_share)}</span>
                </span>
              </div>
              <div className="relative h-2 rounded-full bg-zinc-100">
                <motion.div
                  key={`${chosen}-${s.dest_iso3}`}
                  className="absolute inset-y-0 left-0 rounded-full bg-zinc-400/70"
                  initial={{ width: 0 }}
                  animate={{ width: `${(s.actual_share / maxShare) * 100}%` }}
                  transition={{ duration: 0.6 }}
                />
                <motion.span
                  key={`${chosen}-${s.dest_iso3}-m`}
                  className="absolute top-1/2 h-3.5 w-1 -translate-y-1/2 rounded-full bg-blue-600 dark:bg-blue-400"
                  initial={{ left: 0, opacity: 0 }}
                  animate={{ left: `calc(${(s.model_share / maxShare) * 100}% - 2px)`, opacity: 1 }}
                  transition={{ duration: 0.6, delay: 0.1 }}
                />
              </div>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[11px] text-zinc-500">Grey bar: actual share. Blue tick: forecast made before the crisis.</p>
      </div>
    </Card>
  );
}
