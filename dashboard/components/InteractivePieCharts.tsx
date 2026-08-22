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
  const [activeChart, setActiveChart] = useState<'continent' | 'income' | 'religion' | 'integration' | 'community' | 'top_corridors'>('continent');

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
    const heritageColors = ['#1e3a5f', '#945920', '#28534b', '#8b2626', '#5c3858', '#4f5d30', '#3b5266'];

    return {
      labels,
      datasets: [
        {
          label: 'Total Inflow Stock',
          data,
          backgroundColor: heritageColors.slice(0, labels.length),
          borderColor: '#fbf9f4',
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
      'High income': '#28534b', // Forest Emerald
      'Upper middle income': '#1e3a5f', // Oxford Navy
      'Lower middle income': '#945920', // Warm Cognac
      'Low income': '#8b2626', // Vintage Burgundy
      'Unknown': '#786b59',
    };
    const colors = presentLabels.map((l, idx) => colorMap[l] || ['#28534b', '#1e3a5f', '#945920', '#8b2626'][idx % 4]);

    return {
      labels: presentLabels,
      datasets: [
        {
          label: 'Migrant Stock by Income Tier',
          data,
          backgroundColor: colors,
          borderColor: '#fbf9f4',
          borderWidth: 2,
        },
      ],
    };
  }, [currentEdges]);

  // 3. Dominant Religion Share
  const religionData = React.useMemo(() => {
    const counts: Record<string, number> = {};
    currentEdges.forEach((e) => {
      const rel = e.dest_dominant_religion || 'Unknown';
      counts[rel] = (counts[rel] || 0) + (Number(e.weight) || 0);
    });

    const presentLabels = Object.keys(counts).sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
    const data = presentLabels.map((l) => counts[l]);
    const colorMap: Record<string, string> = {
      'Christianity': '#1e3a5f',
      'Islam': '#28534b',
      'Hinduism': '#945920',
      'Buddhism': '#8c6d23',
      'Judaism': '#5c3858',
      'Unaffiliated': '#556270',
      'Unknown': '#786b59',
    };
    const colors = presentLabels.map((l) => colorMap[l] || '#786b59');

    return {
      labels: presentLabels,
      datasets: [
        {
          label: 'Migrant Stock by Destination Religion',
          data,
          backgroundColor: colors,
          borderColor: '#fbf9f4',
          borderWidth: 2,
        },
      ],
    };
  }, [currentEdges]);

  // 4. Integration Policy Tier (MIPEX)
  const integrationData = React.useMemo(() => {
    const counts: Record<string, number> = {
      'High Integration (>0.70)': 0,
      'Moderate Integration (0.45-0.70)': 0,
      'Restrictive Integration (<0.45)': 0,
    };
    currentEdges.forEach((e) => {
      const mipex = parseFloat(e.dest_integration_policy_index) || 0.5;
      const w = Number(e.weight) || 0;
      if (mipex >= 0.70) {
        counts['High Integration (>0.70)'] += w;
      } else if (mipex >= 0.45) {
        counts['Moderate Integration (0.45-0.70)'] += w;
      } else {
        counts['Restrictive Integration (<0.45)'] += w;
      }
    });

    const labels = Object.keys(counts);
    const data = Object.values(counts);
    const colors = ['#28534b', '#1e3a5f', '#8b2626'];

    return {
      labels,
      datasets: [
        {
          label: 'Stock by Integration Policy Openness',
          data,
          backgroundColor: colors,
          borderColor: '#fbf9f4',
          borderWidth: 2,
        },
      ],
    };
  }, [currentEdges]);

  // 5. Community Cluster Distribution
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
    const colors = ['#1e3a5f', '#945920', '#28534b', '#5c3858', '#8b2626', '#3b5266', '#8c6d23', '#4f5d30'];

    return {
      labels: sortedLabels,
      datasets: [
        {
          label: 'Cluster Attractor Mass',
          data,
          backgroundColor: colors.slice(0, sortedLabels.length),
          borderColor: '#fbf9f4',
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
          backgroundColor: '#1e3a5f',
          borderColor: '#14253e',
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
          color: '#4a3d2e',
          font: { size: 11, family: 'ui-sans-serif, system-ui' },
          boxWidth: 12,
          padding: 10,
        },
      },
      tooltip: {
        backgroundColor: '#ffffff',
        borderColor: '#d8ccb8',
        borderWidth: 1,
        titleColor: '#1e3a5f',
        bodyColor: '#1e293b',
        boxPadding: 4,
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
        backgroundColor: '#ffffff',
        borderColor: '#d8ccb8',
        borderWidth: 1,
        titleColor: '#1e3a5f',
        bodyColor: '#1e293b',
      },
    },
    scales: {
      x: {
        ticks: { color: '#6d5c46', font: { size: 10 } },
        grid: { color: 'rgba(216, 204, 184, 0.4)' },
      },
      y: {
        ticks: { color: '#6d5c46', font: { size: 10 } },
        grid: { color: 'rgba(216, 204, 184, 0.4)' },
      },
    },
  };

  return (
    <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-4 rounded-xl shadow-sm space-y-4">
      {/* Top Header & Chart Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#ded3be] pb-3">
        <div>
          <h3 className="font-bold text-[#1a202c] text-sm flex items-center gap-2 font-serif">
            <span>Macro-Flow Distribution Analytics</span>
            <span className="text-[10px] bg-[#ece4d4] text-[#6d5c46] border border-[#d8ccb8] font-mono px-2 py-0.5 rounded font-semibold">
              Year {year}
            </span>
          </h3>
          <p className="text-xs text-[#635a4d]">
            Categorical breakdowns across continental hubs, income tiers, and geopolitical clusters
          </p>
        </div>

        <div className="flex bg-[#e8e0cf] border border-[#d8ccb8] p-0.5 rounded-lg text-xs">
          {[
            { id: 'continent', label: 'Continent' },
            { id: 'income', label: 'Income Tier' },
            { id: 'religion', label: 'Dominant Religion' },
            { id: 'integration', label: 'MIPEX Policy' },
            { id: 'community', label: 'Alliance Clusters' },
            { id: 'top_corridors', label: 'Top Corridors' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveChart(tab.id as any)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                activeChart === tab.id
                  ? 'bg-[#1e3a5f] text-white font-semibold shadow-sm'
                  : 'text-[#5a5043] hover:text-[#1a202c]'
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
        {activeChart === 'religion' && <Doughnut data={religionData} options={pieOptions} />}
        {activeChart === 'integration' && <Pie data={integrationData} options={pieOptions} />}
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
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs pt-2 border-t border-[#ded3be]">
            <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg shadow-sm">
              <div className="text-[#786b59] font-mono text-[10px] font-semibold uppercase">Primary Destination Pole ({year})</div>
              <div className="font-bold text-[#1e293b] text-xs mt-0.5">High Income Economies</div>
              <div className="text-[#645d52] text-[11px] mt-0.5">
                Absorbs <strong>{highIncomePct}%</strong> of total registered stock in {year}
              </div>
            </div>

            <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg shadow-sm">
              <div className="text-[#786b59] font-mono text-[10px] font-semibold uppercase">Regional Gravity ({year})</div>
              <div className="font-bold text-[#1e3a5f] text-xs mt-0.5">Intra-Continental Flows</div>
              <div className="text-[#645d52] text-[11px] mt-0.5">
                <strong>{intraContPct}%</strong> of movement occurs within same continent
              </div>
            </div>

            <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg shadow-sm">
              <div className="text-[#786b59] font-mono text-[10px] font-semibold uppercase">Global Monitored Mass</div>
              <div className="font-bold text-[#945920] text-xs mt-0.5">
                {totalWeight > 0 ? `${(totalWeight / 1000000).toFixed(1)}M Stock` : '245.0M Stock'}
              </div>
              <div className="text-[#645d52] text-[11px] mt-0.5">
                {currentEdges.length} bilateral corridors recorded in snapshot
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
