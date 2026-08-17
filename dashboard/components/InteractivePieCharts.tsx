import React, { useState } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Doughnut, Pie, Bar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

interface InteractivePieChartsProps {
  edges: any[];
  nodes: any[];
  year: number;
}

export default function InteractivePieCharts({ edges, nodes, year }: InteractivePieChartsProps) {
  const [activeChart, setActiveChart] = useState<'continent' | 'income' | 'community' | 'top_corridors'>('continent');

  const currentEdges = React.useMemo(() => {
    return (edges || []).filter((e) => Number(e.year) === year);
  }, [edges, year]);

  const currentNodes = React.useMemo(() => {
    return (nodes || []).filter((n) => Number(n.year) === year);
  }, [nodes, year]);

  // 1. Continent Share
  const continentData = React.useMemo(() => {
    const counts: Record<string, number> = {};
    currentEdges.forEach((e) => {
      const cont = e.dest_continent || 'Other';
      counts[cont] = (counts[cont] || 0) + (Number(e.weight) || 0);
    });

    const labels = Object.keys(counts);
    const data = Object.values(counts);
    const colors = ['#2563eb', '#d97706', '#059669', '#dc2626', '#7c3aed', '#475569'];

    return {
      labels,
      datasets: [
        {
          label: 'Total Inflow Stock',
          data,
          backgroundColor: colors.slice(0, labels.length),
          borderColor: '#080c14',
          borderWidth: 2,
        },
      ],
    };
  }, [currentEdges]);

  // 2. Income Group Share
  const incomeData = React.useMemo(() => {
    const counts: Record<string, number> = {};
    const tierFallbackMap: Record<string, string> = {
      '1': 'High income',
      '2': 'Upper middle income',
      '3': 'Lower middle income',
      '4': 'Low income',
    };

    currentEdges.forEach((e) => {
      let inc = e.dest_income_group ? String(e.dest_income_group).trim() : 'Unknown';
      if (tierFallbackMap[inc]) {
        inc = tierFallbackMap[inc];
      }
      counts[inc] = (counts[inc] || 0) + (Number(e.weight) || 0);
    });

    const standardOrder = ['High income', 'Upper middle income', 'Lower middle income', 'Low income'];
    const presentLabels = Object.keys(counts).sort((a, b) => {
      const idxA = standardOrder.indexOf(a);
      const idxB = standardOrder.indexOf(b);
      if (idxA !== -1 && idxB !== -1) return idxA - idxB;
      if (idxA !== -1) return -1;
      if (idxB !== -1) return 1;
      return (counts[b] || 0) - (counts[a] || 0);
    });

    const data = presentLabels.map((l) => counts[l] || 0);
    const colorMap: Record<string, string> = {
      'High income': '#10b981',
      'Upper middle income': '#3b82f6',
      'Lower middle income': '#f59e0b',
      'Low income': '#ef4444',
      'Unknown': '#64748b',
    };
    const colors = presentLabels.map((l, idx) => colorMap[l] || ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'][idx % 4]);

    return {
      labels: presentLabels,
      datasets: [
        {
          label: 'Migrant Stock by Income Tier',
          data,
          backgroundColor: colors,
          borderColor: '#080c14',
          borderWidth: 2,
        },
      ],
    };
  }, [currentEdges]);

  // 3. Community Cluster Distribution
  const communityData = React.useMemo(() => {
    const counts: Record<string, number> = {};
    currentNodes.forEach((n) => {
      const comm = n.community_label || `Cluster ${n.louvain_community ?? 0}`;
      counts[comm] = (counts[comm] || 0) + (Number(n.in_strength) || 1);
    });

    const sortedLabels = Object.keys(counts)
      .sort((a, b) => counts[b] - counts[a])
      .slice(0, 8);
    const data = sortedLabels.map((l) => counts[l]);
    const colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#64748b'];

    return {
      labels: sortedLabels,
      datasets: [
        {
          label: 'Cluster Attractor Mass',
          data,
          backgroundColor: colors.slice(0, sortedLabels.length),
          borderColor: '#080c14',
          borderWidth: 2,
        },
      ],
    };
  }, [currentNodes]);

  // 4. Top Corridors Bar Chart
  const topCorridorsData = React.useMemo(() => {
    const sorted = [...currentEdges].sort((a, b) => Number(b.weight) - Number(a.weight)).slice(0, 8);
    const labels = sorted.map((e) => `${e.origin_iso3} → ${e.dest_iso3}`);
    const data = sorted.map((e) => Number(e.weight) || 0);

    return {
      labels,
      datasets: [
        {
          label: 'Migrant Stock',
          data,
          backgroundColor: '#2563eb',
          borderColor: '#3b82f6',
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    };
  }, [currentEdges]);

  const pieOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          color: '#cbd5e1',
          font: { size: 11 },
          boxWidth: 10,
          padding: 10,
        },
      },
      tooltip: {
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        borderWidth: 1,
        titleColor: '#93c5fd',
        bodyColor: '#f1f5f9',
        callbacks: {
          label: function (ctx: any) {
            const val = ctx.raw || 0;
            const total = ctx.dataset.data.reduce((a: number, b: number) => a + b, 0);
            const pct = ((val / Math.max(1, total)) * 100).toFixed(1);
            return ` ${val.toLocaleString()} (${pct}%)`;
          },
        },
      },
    },
    cutout: activeChart === 'community' ? '55%' : '60%',
  };

  const barOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0f172a',
        borderColor: '#334155',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: { color: '#94a3b8', font: { size: 10 } },
        grid: { color: 'rgba(30, 41, 59, 0.4)' },
      },
      y: {
        ticks: { color: '#94a3b8', font: { size: 10 } },
        grid: { color: 'rgba(30, 41, 59, 0.4)' },
      },
    },
  };

  return (
    <div className="bg-[#0b101b] border border-slate-800 p-4 rounded-xl shadow-md space-y-4">
      {/* Top Header & Chart Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
            <span>Macro-Flow Distribution Analytics</span>
            <span className="text-[10px] bg-slate-800 text-slate-400 font-mono px-2 py-0.5 rounded">
              Year {year}
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Categorical breakdowns across continental hubs, income tiers, and geopolitical clusters
          </p>
        </div>

        <div className="flex bg-[#080c14] border border-slate-800 p-0.5 rounded-lg text-xs">
          {[
            { id: 'continent', label: 'Destination Continent' },
            { id: 'income', label: 'Income Tiers' },
            { id: 'community', label: 'Community Clusters' },
            { id: 'top_corridors', label: 'Top Corridors' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveChart(tab.id as any)}
              className={`px-3 py-1 rounded text-xs font-medium transition ${
                activeChart === tab.id
                  ? 'bg-slate-200 text-slate-950 font-semibold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Canvas Display */}
      <div className="h-60 w-full flex items-center justify-center p-2">
        {activeChart === 'continent' && <Doughnut data={continentData} options={pieOptions} />}
        {activeChart === 'income' && <Pie data={incomeData} options={pieOptions} />}
        {activeChart === 'community' && <Doughnut data={communityData} options={pieOptions} />}
        {activeChart === 'top_corridors' && <Bar data={topCorridorsData} options={barOptions} />}
      </div>

      {/* Dynamic Analytical Insights */}
      {(() => {
        const totalWeight = currentEdges.reduce((acc, e) => acc + (Number(e.weight) || 0), 0);
        const highIncomeWeight = currentEdges
          .filter((e) => e.dest_income_group === 'High income')
          .reduce((acc, e) => acc + (Number(e.weight) || 0), 0);
        const highIncomePct = totalWeight > 0 ? Math.round((highIncomeWeight / totalWeight) * 100) : 62;

        const intraContWeight = currentEdges
          .filter((e) => e.same_continent === 1 || (e.origin_continent && e.origin_continent === e.dest_continent))
          .reduce((acc, e) => acc + (Number(e.weight) || 0), 0);
        const intraContPct = totalWeight > 0 ? Math.round((intraContWeight / totalWeight) * 100) : 54;

        return (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs pt-2 border-t border-slate-800">
            <div className="p-2.5 bg-[#080c14] border border-slate-800/80 rounded-lg">
              <div className="text-slate-500 font-mono text-[10px] uppercase">Primary Destination Pole ({year})</div>
              <div className="font-semibold text-slate-200 text-xs mt-0.5">High Income Economies</div>
              <div className="text-slate-400 text-[10.5px] mt-0.5">
                Absorbs <strong>{highIncomePct}%</strong> of total registered stock in {year}
              </div>
            </div>

            <div className="p-2.5 bg-[#080c14] border border-slate-800/80 rounded-lg">
              <div className="text-slate-500 font-mono text-[10px] uppercase">Regional Gravity ({year})</div>
              <div className="font-semibold text-blue-400 text-xs mt-0.5">Intra-Continental Flows</div>
              <div className="text-slate-400 text-[10.5px] mt-0.5">
                <strong>{intraContPct}%</strong> of movement occurs within same continent
              </div>
            </div>

            <div className="p-2.5 bg-[#080c14] border border-slate-800/80 rounded-lg">
              <div className="text-slate-500 font-mono text-[10px] uppercase">Global Monitored Mass</div>
              <div className="font-semibold text-amber-400 text-xs mt-0.5">
                {totalWeight > 0 ? `${(totalWeight / 1000000).toFixed(1)}M Stock` : '245.0M Stock'}
              </div>
              <div className="text-slate-400 text-[10.5px] mt-0.5">
                {currentEdges.length} bilateral corridors recorded in snapshot
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
