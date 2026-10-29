import React, { useMemo, useState } from 'react';
import { Line, Scatter } from 'react-chartjs-2';
import { lineGrid } from '../lib/charts';
import { useChartKey, usePalette } from '../lib/themeContext';
import { ACCENT, categorical, fmtCompact, fmtInt, fmtNum } from '../lib/theme';
import {
  SNAPSHOT_YEARS, type Dataset, type DriverCorrelation, type DriverProfile, type Drivers as DriversData, type Year,
} from '../lib/types';
import { Card, Empty, Segmented, Stat, Table, td, th } from './ui';

type Outcome = 'immigrant_share' | 'emigrant_rate';
type Mode = 'partial' | 'raw';

const NEGATIVE = '#ea580c';
const OUTCOMES: { value: Outcome; label: string }[] = [
  { value: 'immigrant_share', label: 'Immigration' },
  { value: 'emigrant_rate', label: 'Emigration' },
];
const OUTCOME_TEXT: Record<Outcome, string> = {
  immigrant_share: 'immigrants as a share of the population',
  emigrant_rate: 'emigrants as a share of everyone born in the country',
};
const TIE_TERMS = ['contig', 'comlang_ethno', 'colony', 'comcol', 'smctry'];

const PCT_FACTORS = new Set([
  'gdp_growth', 'inflation', 'unemployment', 'gini', 'trade_openness', 'resource_rents', 'agri_employment',
  'female_labour_participation', 'remittances_gdp', 'urban_pct', 'youth_share', 'elderly_share', 'internet_users',
]);

function fmtFactor(factor: string, v: number | null): string {
  if (v == null || !Number.isFinite(v)) return '—';
  if (factor === 'log_gdp_pc_ppp') return `$${fmtCompact(Math.exp(v), 1)}`;
  if (factor === 'log_population') return fmtCompact(Math.exp(v), 1);
  if (factor === 'log_pop_density') return `${fmtInt(Math.exp(v))} /km²`;
  if (factor === 'log_area') return `${fmtCompact(Math.exp(v), 0)} km²`;
  if (PCT_FACTORS.has(factor)) return `${fmtNum(v, 1)}%`;
  if (factor.endsWith('_per_1k')) return `${fmtNum(v, 1)} /1k`;
  if (factor === 'mobile_subscriptions') return `${fmtInt(v)} /100`;
  if (factor === 'homicide_rate') return `${fmtNum(v, 1)} /100k`;
  return fmtNum(v, 2);
}

export default function Drivers({ data, year }: { data: Dataset; year: Year }) {
  const d = data.drivers;
  if (!d || d.correlations.length === 0) {
    return (
      <Card>
        <Empty>Driver analysis not found. Run <code>python notebooks/07_migration_drivers.py</code> from the repository root.</Empty>
      </Card>
    );
  }
  return (
    <div className="space-y-6">
      <Headline d={d} metrics={data.model_metrics} year={year} />
      <Correlates d={d} />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Hump d={d} year={year} />
        <Gravity d={d} year={year} />
      </div>
      <Ablation d={d} metrics={data.model_metrics} />
    </div>
  );
}

function ppml(d: DriversData, year: number, term: string) {
  return d.gravity.find((g) => g.estimator === 'ppml' && Number(g.year) === year && g.term === term);
}

function Headline({ d, metrics, year }: { d: DriversData; metrics: Record<string, any> | null; year: Year }) {
  const original = d.model_ablation.find((r) => r.feature_set.startsWith('Original'));
  const dist = ppml(d, year, 'log_dist');
  const dist1990 = ppml(d, SNAPSHOT_YEARS[0], 'log_dist');
  const colony = ppml(d, year, 'colony');
  const colony1990 = ppml(d, SNAPSHOT_YEARS[0], 'colony');
  return (
    <section className="grid grid-cols-2 gap-6 rounded-xl border border-zinc-200 bg-white p-5 lg:grid-cols-4">
      <Stat
        label="Corridor model R²"
        value={fmtNum(metrics?.r2, 3)}
        hint={original ? `was ${fmtNum(original.r2, 3)} before the new data` : undefined}
        tone="accent"
      />
      <Stat
        label={`Distance elasticity, ${year}`}
        value={fmtNum(dist?.coefficient, 2)}
        hint={dist1990 ? `${fmtNum(dist1990.coefficient, 2)} in ${SNAPSHOT_YEARS[0]}` : undefined}
      />
      <Stat
        label={`Colonial link, ${year}`}
        value={colony?.pct_effect != null ? `+${fmtInt(colony.pct_effect)}%` : '—'}
        hint={colony1990?.pct_effect != null ? `+${fmtInt(colony1990.pct_effect)}% in ${SNAPSHOT_YEARS[0]}` : undefined}
      />
      <Stat
        label="Forecast check"
        value={fmtNum(metrics?.temporal?.r2, 3)}
        hint={metrics?.temporal ? `R², trained ${metrics.temporal.train_years}, tested ${metrics.temporal.test_year}` : undefined}
      />
    </section>
  );
}

function Correlates({ d }: { d: DriversData }) {
  const [outcome, setOutcome] = useState<Outcome>('immigrant_share');
  const [mode, setMode] = useState<Mode>('partial');

  const rows = useMemo(() => {
    const key = mode === 'partial' ? 'rho_partial' : 'rho';
    return d.correlations
      .filter((c) => c.outcome === outcome)
      .map((c) => ({ ...c, value: Number(c[key]), p: Number(mode === 'partial' ? c.p_partial : c.p_value) }))
      .sort((a, b) => b.value - a.value);
  }, [d, outcome, mode]);

  const profileRows = useMemo(() => {
    const byFactor = new Map(d.profile.filter((p) => p.outcome === outcome).map((p) => [p.factor, p]));
    return [...rows]
      .filter((r) => r.p < 0.05 && r.factor !== 'landlocked')
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 10)
      .map((r) => byFactor.get(r.factor))
      .filter((p): p is DriverProfile => !!p);
  }, [d, rows, outcome]);

  const n = rows[0]?.n;
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
      <Card
        className="lg:col-span-3"
        title="What high-migration countries have in common"
        subtitle={`Spearman correlation with ${OUTCOME_TEXT[outcome]}, 2020, countries with at least 1 million residents${n ? ` (n ≈ ${n})` : ''}.`}
        actions={
          <div className="flex flex-wrap gap-2">
            <Segmented size="xs" ariaLabel="Outcome" value={outcome} onChange={setOutcome} options={OUTCOMES} />
            <Segmented
              size="xs"
              ariaLabel="Correlation type"
              value={mode}
              onChange={setMode}
              options={[
                { value: 'partial', label: 'Income & size held fixed' },
                { value: 'raw', label: 'Raw' },
              ]}
            />
          </div>
        }
      >
        <DivergingBars rows={rows} showCI={mode === 'partial'} />
        <p className="mt-4 text-[11px] leading-relaxed text-zinc-500">
          Blue: more of the factor goes with more migration. Orange: with less. Grey bars are not significant at 5%.
          {mode === 'partial' && ' Whiskers are 95% bootstrap intervals. Holding income (GDP per capita, PPP) and population fixed shows what matters beyond being rich or small.'}
          {' '}† partly a mechanical consequence of migration rather than a cause. Correlation is not causation.
        </p>
      </Card>

      <Card
        className="lg:col-span-2"
        title={`Typical ${outcome === 'immigrant_share' ? 'high-immigration' : 'high-emigration'} country`}
        subtitle="Medians for the top and bottom fifth of countries, for the ten strongest significant correlates"
        bodyClassName="px-2 pb-2"
      >
        {profileRows.length === 0 ? (
          <Empty>No significant correlates</Empty>
        ) : (
          <Table>
            <thead>
              <tr>
                <th className={th}>Factor</th>
                <th className={`${th} text-right`}>Bottom 20%</th>
                <th className={`${th} text-right`}>Top 20%</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {profileRows.map((p) => (
                <tr key={p.factor}>
                  <td className={`${td} max-w-[180px] truncate text-xs`} title={p.label}>{p.label}</td>
                  <td className={`${td} text-right text-xs tabular-nums`}>{fmtFactor(p.factor, p.median_bottom)}</td>
                  <td className={`${td} text-right text-xs font-medium tabular-nums text-zinc-900`}>{fmtFactor(p.factor, p.median_top)}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}

function DivergingBars({ rows, showCI }: { rows: (DriverCorrelation & { value: number; p: number })[]; showCI: boolean }) {
  const pos = (v: number) => `${50 + Math.max(-1, Math.min(1, v)) * 50}%`;
  return (
    <ul className="space-y-1">
      {rows.map((r) => {
        const sig = r.p < 0.05;
        const color = !sig ? 'rgb(var(--zinc-300))' : r.value >= 0 ? ACCENT : NEGATIVE;
        const left = Math.min(50, 50 + r.value * 50);
        const width = Math.abs(r.value) * 50;
        return (
          <li
            key={r.factor}
            className="grid grid-cols-[minmax(0,11rem)_1fr_3rem] items-center gap-3 text-xs"
            title={`${r.label}: ρ = ${r.value.toFixed(2)}${showCI ? ` [${Number(r.ci_low).toFixed(2)}, ${Number(r.ci_high).toFixed(2)}]` : ''}, p = ${r.p.toPrecision(2)}, n = ${r.n}${r.note ? ` — ${r.note}` : ''}`}
          >
            <span className={`truncate ${sig ? 'text-zinc-800' : 'text-zinc-400'}`}>
              {r.label}
              {r.note ? <span className="ml-0.5 text-zinc-400">†</span> : null}
            </span>
            <span className="relative h-4">
              <span className="absolute inset-y-0 left-1/2 w-px bg-zinc-300" />
              <span className="absolute top-1 h-2 rounded-sm" style={{ left: `${left}%`, width: `${width}%`, backgroundColor: color }} />
              {showCI && sig && (
                <span
                  className="absolute top-[7px] h-px bg-zinc-500"
                  style={{ left: pos(Number(r.ci_low)), width: `${(Number(r.ci_high) - Number(r.ci_low)) * 50}%` }}
                />
              )}
            </span>
            <span className={`text-right tabular-nums ${sig ? 'font-medium text-zinc-900' : 'text-zinc-400'}`}>
              {r.value >= 0 ? '+' : ''}{r.value.toFixed(2)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Hump({ d, year }: { d: DriversData; year: Year }) {
  const chartKey = useChartKey();
  const palette = usePalette();
  const points = d.countries.filter((c) => c.year === year && c.gdp_pc_ppp && c.emigrant_rate && c.emigrant_rate > 0);
  const fit = d.hump_curve.filter((c) => c.kind === 'fit');
  const bins = d.hump_curve.filter((c) => c.kind === 'bin_median');
  const pooled = d.hump.find((h) => h.spec === 'pooled');
  const within = d.hump.find((h) => h.spec === 'within_country');

  const chart = {
    datasets: [
      {
        label: 'Countries',
        data: points.map((c) => ({ x: c.gdp_pc_ppp, y: c.emigrant_rate, name: c.country_name ?? c.iso3 })),
        backgroundColor: 'rgba(37, 99, 235, 0.35)',
        borderColor: 'rgba(37, 99, 235, 0.6)',
        pointRadius: 3,
        pointHoverRadius: 6,
        order: 3,
      },
      {
        label: 'Median by income band (1990–2020)',
        data: bins.map((b) => ({ x: Number(b.gdp_pc_ppp), y: Number(b.emigrant_rate) })),
        backgroundColor: NEGATIVE,
        borderColor: NEGATIVE,
        showLine: true,
        borderWidth: 2,
        pointRadius: 4,
        order: 1,
      },
      {
        label: 'Quadratic fit (pooled)',
        data: fit.map((f) => ({ x: Number(f.gdp_pc_ppp), y: Number(f.emigrant_rate) })),
        borderColor: palette.ink,
        borderDash: [4, 4],
        borderWidth: 1.5,
        showLine: true,
        pointRadius: 0,
        order: 2,
      },
    ],
  };
  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', align: 'end' },
      tooltip: {
        filter: (ctx: any) => ctx.datasetIndex !== 2,
        callbacks: {
          label: (ctx: any) =>
            ctx.raw.name
              ? ` ${ctx.raw.name}: ${fmtNum(ctx.raw.y, 1)}% emigrated · $${fmtCompact(ctx.raw.x, 1)}`
              : ` Median ${fmtNum(ctx.raw.y, 1)}% at $${fmtCompact(ctx.raw.x, 1)}`,
        },
      },
    },
    scales: {
      x: {
        type: 'logarithmic',
        grid: lineGrid,
        title: { display: true, text: 'GDP per capita, PPP (log scale)' },
        ticks: { callback: (v: number) => ([1000, 3000, 10000, 30000, 100000].includes(v) ? `$${fmtCompact(v, 0)}` : '') },
      },
      y: {
        type: 'logarithmic',
        grid: lineGrid,
        title: { display: true, text: 'Emigrants, % of native-born (log scale)' },
        ticks: { callback: (v: number) => ([0.1, 0.3, 1, 3, 10, 30].includes(v) ? `${v}%` : '') },
      },
    },
  };

  const sigHump = pooled && Number(pooled.p_log_gdp_sq) < 0.05 && Number(pooled.b_log_gdp_sq) < 0;
  return (
    <Card
      title="Does development reduce emigration?"
      subtitle={`The “migration hump”: emigration against income, ${year}. Lines pool all observed years.`}
    >
      <div className="h-72">
        {points.length === 0 ? <Empty>No country data for {year}</Empty> : <Scatter key={chartKey} data={chart} options={options} />}
      </div>
      <p className="mt-4 text-xs leading-relaxed text-zinc-600">
        {sigHump ? (
          <>Across countries, emigration rises with income and peaks around <b>${fmtCompact(Number(pooled!.peak_gdp_pc_ppp), 0)}</b> (PPP) before falling.</>
        ) : (
          <>
            The poorest countries have the lowest emigration rates, and middle-income countries some of the highest, but the
            quadratic hump is <b>not statistically significant</b>
            {pooled ? ` (p = ${Number(pooled.p_log_gdp_sq).toFixed(2)})` : ''}.
          </>
        )}{' '}
        {within && (
          <>
            Within countries over time, the curve is {Number(within.b_log_gdp_sq) > 0 ? 'U-shaped rather than humped' : 'humped'}
            {` (p = ${Number(within.p_log_gdp_sq).toFixed(2)})`}: the cross-country pattern does not show that a given country’s emigration
            rises as it grows, which matches the open debate in the literature.
          </>
        )}
      </p>
    </Card>
  );
}

function trendText(subject: string, first: number | null | undefined, last: number | null | undefined, unit: string): string {
  if (first == null || last == null) return '';
  const f = (v: number) => (unit === '%' ? `+${fmtInt(v)}%` : fmtNum(v, 2));
  return `${subject} was worth ${f(first)} in ${SNAPSHOT_YEARS[0]} and ${f(last)} in 2020.`;
}

function Gravity({ d, year }: { d: DriversData; year: Year }) {
  const chartKey = useChartKey();
  const [view, setView] = useState<'ties' | 'distance'>('ties');
  const years = SNAPSHOT_YEARS;
  const series = (term: string, estimator: 'ppml' | 'ols_log' = 'ppml', key: 'pct_effect' | 'coefficient' = 'pct_effect') =>
    years.map((y) => {
      const r = d.gravity.find((g) => g.estimator === estimator && Number(g.year) === y && g.term === term);
      return r ? Number(r[key]) : null;
    });

  const labels = new Map(d.gravity.map((g) => [g.term, g.label]));
  const pointRadius = years.map((y) => (y === year ? 5 : 2.5));
  const chart =
    view === 'ties'
      ? {
          labels: years,
          datasets: TIE_TERMS.map((t, i) => ({
            label: labels.get(t) ?? t,
            data: series(t),
            borderColor: categorical(i),
            backgroundColor: categorical(i),
            tension: 0.3,
            pointRadius,
          })),
        }
      : {
          labels: years,
          datasets: [
            {
              label: 'PPML (zeros kept)',
              data: series('log_dist', 'ppml', 'coefficient'),
              borderColor: ACCENT,
              backgroundColor: ACCENT,
              tension: 0.3,
              pointRadius,
            },
            {
              label: 'OLS on log stock (zeros dropped)',
              data: series('log_dist', 'ols_log', 'coefficient'),
              borderColor: '#a1a1aa',
              backgroundColor: '#a1a1aa',
              borderDash: [4, 4],
              tension: 0.3,
              pointRadius: 2.5,
            },
          ],
        };
  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', align: 'end' },
      tooltip: {
        callbacks: {
          label: (ctx: any) =>
            view === 'ties' ? ` ${ctx.dataset.label}: +${fmtInt(ctx.raw)}%` : ` ${ctx.dataset.label}: ${fmtNum(ctx.raw, 2)}`,
        },
      },
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        grid: lineGrid,
        title: { display: true, text: view === 'ties' ? 'Extra migrant stock when the tie exists' : 'Elasticity of stock to distance' },
        ticks: { callback: (v: number) => (view === 'ties' ? `+${v}%` : v) },
      },
    },
  };

  return (
    <Card
      title="What makes a corridor large?"
      subtitle="Structural gravity (PPML) with origin and destination fixed effects, every country pair including zero corridors"
      actions={
        <Segmented
          size="xs"
          ariaLabel="Gravity view"
          value={view}
          onChange={setView}
          options={[
            { value: 'ties', label: 'Shared ties' },
            { value: 'distance', label: 'Distance' },
          ]}
        />
      }
    >
      <div className="h-72">
        <Line key={chartKey} data={chart} options={options} />
      </div>
      <p className="mt-4 text-xs leading-relaxed text-zinc-600">
        {view === 'ties'
          ? `Each line is the extra migrant stock between two countries that share the tie, compared with otherwise identical pairs. ${trendText(
              'A colonial link',
              ppml(d, years[0], 'colony')?.pct_effect,
              ppml(d, 2020, 'colony')?.pct_effect,
              '%',
            )} ${trendText('A common language', ppml(d, years[0], 'comlang_ethno')?.pct_effect, ppml(d, 2020, 'comlang_ethno')?.pct_effect, '%')}`
          : `A 1% increase in distance changes migrant stock by this many percent. ${trendText(
              'The elasticity',
              ppml(d, years[0], 'log_dist')?.coefficient,
              ppml(d, 2020, 'log_dist')?.coefficient,
              '',
            )} OLS on logs drops the zero corridors and gives a different answer.`}
        {' '}2025 is extrapolated.
      </p>
    </Card>
  );
}

function Ablation({ d, metrics }: { d: DriversData; metrics: Record<string, any> | null }) {
  const rows = d.model_ablation;
  if (rows.length === 0) return null;
  const max = Math.max(...rows.map((r) => Number(r.r2)), 1);
  return (
    <Card
      title="How much the new data improves corridor prediction"
      subtitle="Held-out R² on log migrant stock for country pairs the model never saw, by feature set (same split, same settings)"
    >
      <ul className="space-y-2.5">
        {rows.map((r) => {
          const served = r.feature_set.startsWith('Served');
          return (
            <li key={r.feature_set}>
              <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
                <span className={served ? 'font-medium text-zinc-900' : 'text-zinc-700'}>
                  {r.feature_set} <span className="text-zinc-400">· {r.n_features} features</span>
                </span>
                <span className="tabular-nums font-medium text-zinc-900">{Number(r.r2).toFixed(3)}</span>
              </div>
              <div className="h-1.5 rounded-full bg-zinc-100">
                <div
                  className="h-1.5 rounded-full"
                  style={{ width: `${(Number(r.r2) / max) * 100}%`, backgroundColor: served ? ACCENT : '#93c5fd' }}
                />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-4 text-[11px] text-zinc-500">
        The served model ({metrics?.n_features ?? '—'} features, tuned) scores R² {fmtNum(metrics?.r2, 3)} and leaves out the modelled
        religion and integration indices, so its predictions rest on measured data only.
      </p>
    </Card>
  );
}
