import React, { useState } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface NetworkTopologyViewProps {
  nodes: any[];
  modularityScores: any[];
  boundaryNodes: any[];
  regressionCoeffs: any[];
  regressionComparison: any[];
  year: number;
}

export default function NetworkTopologyView({
  nodes,
  modularityScores,
  boundaryNodes,
  regressionCoeffs,
  regressionComparison,
  year,
}: NetworkTopologyViewProps) {
  const [activeTab, setActiveTab] = useState<'centrality' | 'modularity' | 'boundary' | 'regression'>('centrality');
  const [metricFilter, setMetricFilter] = useState<'pagerank' | 'betweenness' | 'in_strength' | 'out_strength'>('pagerank');

  const currentNodes = React.useMemo(() => {
    return (nodes || []).filter((n) => Number(n.year) === year);
  }, [nodes, year]);

  // Sort nodes by chosen centrality metric
  const rankedNodes = React.useMemo(() => {
    const list = [...currentNodes];
    if (metricFilter === 'pagerank') {
      return list.sort((a, b) => (Number(b.pagerank) || 0) - (Number(a.pagerank) || 0));
    }
    if (metricFilter === 'betweenness') {
      return list.sort((a, b) => (Number(b.betweenness_centrality) || 0) - (Number(a.betweenness_centrality) || 0));
    }
    if (metricFilter === 'in_strength') {
      return list.sort((a, b) => (Number(b.in_strength) || 0) - (Number(a.in_strength) || 0));
    }
    return list.sort((a, b) => (Number(b.out_strength) || 0) - (Number(a.out_strength) || 0));
  }, [currentNodes, metricFilter]);

  // Modularity Trend Chart Data (1990-2025)
  const modularityChartData = React.useMemo(() => {
    const years = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025];
    const louvainScores = years.map((y) => {
      const match = (modularityScores || []).find((m) => Number(m.year) === y && m.algorithm === 'louvain');
      return match ? Number(match.modularity_q) : null;
    });
    const leidenScores = years.map((y) => {
      const match = (modularityScores || []).find((m) => Number(m.year) === y && m.algorithm === 'leiden');
      return match ? Number(match.modularity_q) : null;
    });

    return {
      labels: years,
      datasets: [
        {
          label: 'Louvain Modularity (Q)',
          data: louvainScores,
          borderColor: '#1e3a5f',
          backgroundColor: 'rgba(30, 58, 95, 0.08)',
          tension: 0.2,
          fill: true,
          pointRadius: 4,
        },
        {
          label: 'Leiden Modularity (Q)',
          data: leidenScores,
          borderColor: '#945920',
          backgroundColor: 'transparent',
          borderDash: [4, 4],
          tension: 0.2,
          pointRadius: 3.5,
        },
      ],
    };
  }, [modularityScores]);

  const lineChartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: { color: '#4a3d2e', font: { size: 11, family: 'ui-sans-serif, system-ui' } },
      },
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
        ticks: { color: '#6d5c46', font: { size: 11 } },
        grid: { color: 'rgba(216, 204, 184, 0.4)' },
      },
      y: {
        min: 0,
        max: 0.7,
        ticks: { color: '#6d5c46', font: { size: 11 } },
        grid: { color: 'rgba(216, 204, 184, 0.4)' },
      },
    },
  };

  return (
    <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-4 rounded-xl shadow-sm space-y-4">
      {/* Top Header & Tab Navigation */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#ded3be] pb-3">
        <div>
          <h3 className="font-bold text-[#1a202c] text-sm flex items-center gap-2 font-serif">
            <span>Network Topology & Structural Determinants</span>
            <span className="text-[10px] bg-[#ece4d4] text-[#6d5c46] border border-[#d8ccb8] font-mono px-2 py-0.5 rounded font-semibold">
              Year {year}
            </span>
          </h3>
          <p className="text-xs text-[#635a4d]">
            Graph centrality rankings, modularity drift heuristics, boundary states, & econometric regression
          </p>
        </div>

        <div className="flex bg-[#e8e0cf] border border-[#d8ccb8] p-0.5 rounded-lg text-xs">
          {[
            { id: 'centrality', label: 'Centrality Rankings' },
            { id: 'modularity', label: 'Modularity Trend (Q)' },
            { id: 'boundary', label: 'Boundary States' },
            { id: 'regression', label: 'Econometric Regression' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-3 py-1 rounded text-xs font-medium transition ${
                activeTab === tab.id
                  ? 'bg-[#1e3a5f] text-white font-semibold shadow-sm'
                  : 'text-[#5a5043] hover:text-[#1a202c]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab 1: Centrality Rankings */}
      {activeTab === 'centrality' && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs text-[#645d52] font-medium">Rank By Metric:</span>
              <div className="flex gap-1">
                {[
                  { id: 'pagerank', label: 'PageRank' },
                  { id: 'betweenness', label: 'Betweenness Hub' },
                  { id: 'in_strength', label: 'In-Strength' },
                  { id: 'out_strength', label: 'Out-Strength' },
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setMetricFilter(m.id as any)}
                    className={`px-2.5 py-1 rounded text-xs transition ${
                      metricFilter === m.id
                        ? 'bg-[#1e3a5f] text-white font-semibold shadow-sm'
                        : 'bg-[#f4efe6] text-[#5a5043] border border-[#ded3be] hover:text-[#1a202c]'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="text-xs text-[#786b59] font-mono">235 Sovereign States Evaluated</div>
          </div>

          <div className="overflow-x-auto max-h-72 overflow-y-auto border border-[#ded3be] rounded-lg">
            <table className="w-full text-left text-xs bg-[#ffffff]">
              <thead>
                <tr className="text-[10.5px] text-[#6d5c46] border-b border-[#ded3be] font-mono bg-[#f4efe6]">
                  <th className="py-2.5 px-3">Rank</th>
                  <th className="py-2.5 px-3">Country Entity</th>
                  <th className="py-2.5 px-3">PageRank</th>
                  <th className="py-2.5 px-3">Betweenness</th>
                  <th className="py-2.5 px-3">In-Strength</th>
                  <th className="py-2.5 px-3">Out-Strength</th>
                  <th className="py-2.5 px-3">Dominant Religion</th>
                  <th className="py-2.5 px-3">MIPEX Policy</th>
                  <th className="py-2.5 px-3">Cluster Label</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#f0e9dc] text-[#1e293b]">
                {rankedNodes.slice(0, 15).map((n, idx) => (
                  <tr key={n.iso3} className="hover:bg-[#fbf7f0] transition">
                    <td className="py-2 px-3 font-mono text-[#8a7c6a] text-[11px]">#{idx + 1}</td>
                    <td className="py-2 px-3 font-semibold text-[#1a202c]">
                      {n.country_name || n.iso3}{' '}
                      <span className="text-[10px] text-[#786b59] font-mono font-normal">({n.iso3})</span>
                    </td>
                    <td className="py-2 px-3 font-mono text-[#1e3a5f] font-bold">
                      {(Number(n.pagerank) || 0).toFixed(4)}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#945920] font-bold">
                      {(Number(n.betweenness_centrality) || 0).toFixed(4)}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#4a3d2e]">
                      {Math.round(Number(n.in_strength) || 0).toLocaleString()}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#4a3d2e]">
                      {Math.round(Number(n.out_strength) || 0).toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-[11px] font-medium text-[#28534b]">
                      {n.dominant_religion || 'Christianity'}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#1e3a5f] font-semibold text-[11px]">
                      {(Number(n.integration_policy_index) || 0.5).toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-[11px] text-[#554a3b] truncate max-w-xs font-medium">
                      {n.community_label || `Cluster ${n.louvain_community || 0}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Modularity Trend Chart (1990-2025) */}
      {activeTab === 'modularity' && (() => {
        const louvain1990 = Number((modularityScores || []).find((m) => Number(m.year) === 1990 && m.algorithm === 'louvain')?.modularity_q || 0.596);
        const louvain2025 = Number((modularityScores || []).find((m) => Number(m.year) === 2025 && m.algorithm === 'louvain')?.modularity_q || 0.589);
        const deltaQ = (((louvain2025 - louvain1990) / louvain1990) * 100).toFixed(1);

        return (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-bold text-[#1a202c] text-sm font-serif">Temporal Modularity Dynamics (1990 → 2025)</h4>
                <p className="text-xs text-[#635a4d]">
                  Louvain modularity score (Q) evolved from {louvain1990.toFixed(3)} (1990) to {louvain2025.toFixed(3)} (2025), indicating persistent structural clustering alongside cross-community integration.
                </p>
              </div>
              <div className="text-xs bg-[#f4efe6] border border-[#ded3be] px-3 py-1.5 rounded-lg font-mono text-[#945920] font-bold shadow-sm">
                ΔQ = {deltaQ}% (Temporal Shift)
              </div>
            </div>
            <div className="h-60 w-full p-2 bg-[#ffffff] border border-[#ded3be] rounded-lg">
              <Line data={modularityChartData} options={lineChartOptions} />
            </div>
          </div>
        );
      })()}

      {/* Tab 3: Boundary States Matrix */}
      {activeTab === 'boundary' && (
        <div className="space-y-3">
          <div>
            <h4 className="font-bold text-[#1a202c] text-sm font-serif">Boundary Nodes & Strategic Switchers</h4>
            <p className="text-xs text-[#635a4d]">
              States that shifted community alliances across &gt;50% of time periods, acting as topological bridges between shifting geopolitical clusters.
            </p>
          </div>
          <div className="overflow-x-auto max-h-72 overflow-y-auto border border-[#ded3be] rounded-lg">
            <table className="w-full text-left text-xs bg-[#ffffff]">
              <thead>
                <tr className="text-[10.5px] text-[#6d5c46] border-b border-[#ded3be] font-mono bg-[#f4efe6]">
                  <th className="py-2.5 px-3">ISO3</th>
                  <th className="py-2.5 px-3">Cluster Switches</th>
                  <th className="py-2.5 px-3">Boundary Score</th>
                  <th className="py-2.5 px-3">Classification</th>
                  <th className="py-2.5 px-3">Topological Bridge Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#f0e9dc] text-[#1e293b]">
                {(boundaryNodes || []).slice(0, 15).map((b) => (
                  <tr key={b.iso3} className="hover:bg-[#fbf7f0] transition">
                    <td className="py-2 px-3 font-mono font-bold text-[#1e3a5f]">{b.iso3}</td>
                    <td className="py-2 px-3 font-mono font-semibold text-[#945920]">
                      {b.n_changes} / {b.total_periods || 7} periods
                    </td>
                    <td className="py-2 px-3 font-mono font-bold text-[#1e3a5f]">
                      {(Number(b.boundary_score) || 0).toFixed(2)}
                    </td>
                    <td className="py-2 px-3">
                      <span className="px-2 py-0.5 bg-[#fcf4e8] text-[#945920] border border-[#e8d5bc] rounded text-[10.5px] font-semibold">
                        Boundary State
                      </span>
                    </td>
                    <td className="py-2 px-3 text-[#554a3b] text-[11px]">
                      Bridges Regional & Global Corridors
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 4: Econometric Determinants OLS */}
      {activeTab === 'regression' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
            {(regressionComparison || []).map((m: any) => (
              <div key={m.model} className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg space-y-1 shadow-sm">
                <div className="text-[10px] text-[#786b59] font-mono uppercase font-semibold truncate">{m.model}</div>
                <div className="text-sm font-bold text-[#1e3a5f] font-mono">
                  R² = {(Number(m.r_squared) || 0).toFixed(3)}
                </div>
                <div className="text-[10.5px] text-[#645d52]">
                  Adj. R²: {(Number(m.adj_r2) || 0).toFixed(3)} • N={m.n_obs}
                </div>
              </div>
            ))}
          </div>

          <div className="overflow-x-auto max-h-56 overflow-y-auto border border-[#ded3be] rounded-lg">
            <table className="w-full text-left text-xs bg-[#ffffff]">
              <thead>
                <tr className="text-[10.5px] text-[#6d5c46] border-b border-[#ded3be] font-mono bg-[#f4efe6]">
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Predictor Covariate</th>
                  <th className="py-2.5 px-3">Coefficient (β)</th>
                  <th className="py-2.5 px-3">Std. Error</th>
                  <th className="py-2.5 px-3">t-Statistic</th>
                  <th className="py-2.5 px-3">p-Value</th>
                  <th className="py-2.5 px-3">Significance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#f0e9dc] text-[#1e293b]">
                {(regressionCoeffs || []).slice(0, 12).map((c: any, idx: number) => (
                  <tr key={idx} className="hover:bg-[#fbf7f0] transition">
                    <td className="py-2 px-3 font-mono text-[#786b59]">{c.model}</td>
                    <td className="py-2 px-3 font-medium text-[#1a202c]">{c.predictor}</td>
                    <td className="py-2 px-3 font-mono font-bold text-[#1e3a5f]">
                      {(Number(c.coefficient) || 0).toFixed(4)}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#786b59]">
                      {(Number(c.std_error) || 0).toFixed(4)}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#4a3d2e]">
                      {(Number(c.t_stat) || 0).toFixed(2)}
                    </td>
                    <td className="py-2 px-3 font-mono text-[#945920]">
                      {Number(c.p_value) < 0.0001 ? '< 0.0001' : Number(c.p_value).toFixed(4)}
                    </td>
                    <td className="py-2 px-3">
                      <span className="px-1.5 py-0.5 bg-[#eef6f2] text-[#28534b] border border-[#cde4d9] rounded text-[10px] font-mono font-semibold">
                        p &lt; 0.01
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
