import React from 'react';
import { Line } from 'react-chartjs-2';
import { lineGrid } from '../lib/charts';
import { useChartKey } from '../lib/themeContext';
import {
  ACCENT, INCOME_COLORS, MIPEX_COLORS, MIPEX_TIERS, RELIGION_COLORS, categorical, fmtCompact, fmtPct,
} from '../lib/theme';
import { SNAPSHOT_YEARS, type Dataset, type Year } from '../lib/types';
import { Card, ShareBars, Stat } from './ui';

export default function Distributions({ data, year }: { data: Dataset; year: Year }) {
  const chartKey = useChartKey();
  const dist = data.distributions[year];
  if (!dist) return <Card><p className="text-sm text-zinc-500">No distribution data for {year}.</p></Card>;

  const series = SNAPSHOT_YEARS.map((y) => data.distributions[y]);
  const trend = {
    labels: SNAPSHOT_YEARS,
    datasets: [
      {
        label: 'Global migrant stock',
        data: series.map((d) => d?.total_stock ?? null),
        borderColor: ACCENT,
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        fill: true,
        tension: 0.3,
        pointRadius: SNAPSHOT_YEARS.map((y) => (y === year ? 5 : 2.5)),
        pointBackgroundColor: ACCENT,
      },
    ],
  };

  const trendOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx: any) => ` ${fmtCompact(ctx.raw, 1)} migrants` } },
    },
    scales: {
      x: { grid: { display: false } },
      y: { grid: lineGrid, ticks: { callback: (v: number) => fmtCompact(v, 0) } },
    },
  };

  const byContinent = (label: string, i: number) => categorical(i);
  const byIncome = (label: string) => INCOME_COLORS[label] ?? '#a1a1aa';
  const byReligion = (label: string) => RELIGION_COLORS[label] ?? '#a1a1aa';
  const byMipex = (label: string) => MIPEX_COLORS[MIPEX_TIERS.indexOf(label)] ?? '#a1a1aa';
  // Match the community colours used on the map (keyed by community ID, not list position)
  const byCommunity = (label: string, i: number) => {
    const id = /^Cluster (\d+)/.exec(label)?.[1];
    return id !== undefined ? categorical(Number(id)) : categorical(i);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Global migrant stock" subtitle="Sum over all bilateral corridors, 1990–2025 (2025 is extrapolated)">
          <div className="h-56">
            <Line key={chartKey} data={trend} options={trendOptions} />
          </div>
        </Card>
        <Card title={`Snapshot ${year}`}>
          <div className="grid grid-cols-1 gap-5">
            <Stat label="Total migrant stock" value={fmtCompact(dist.total_stock)} tone="accent" />
            <Stat label="Living in high-income states" value={fmtPct(dist.high_income_share)} />
            <Stat label="Moved within their continent" value={fmtPct(dist.intra_continental_share)} />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card title="By destination continent" subtitle="Where migrants live">
          <ShareBars data={dist.continent} colorFor={byContinent} format={fmtCompact} />
        </Card>
        <Card title="By destination income group" subtitle="World Bank classification">
          <ShareBars data={dist.income} colorFor={byIncome} format={fmtCompact} />
        </Card>
        <Card title="By destination's dominant religion" subtitle="Modelled religion profile (see data notes)">
          <ShareBars data={dist.religion} colorFor={byReligion} format={fmtCompact} />
        </Card>
        <Card title="By destination integration policy" subtitle="Modelled MIPEX-style index tiers (see data notes)">
          <ShareBars data={dist.mipex} colorFor={byMipex} format={fmtCompact} />
        </Card>
      </div>

      <Card title="By network community" subtitle={`Louvain communities of the ${year} migration graph, labelled by dominant region`}>
        <ShareBars data={dist.community} colorFor={byCommunity} format={fmtCompact} />
      </Card>
    </div>
  );
}
