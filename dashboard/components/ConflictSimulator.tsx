import React, { useState, useEffect } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut, Pie } from 'react-chartjs-2';
import { COUNTRY_GEO_MAP, CountryGeo, getCountryGeo } from '../lib/countryGeo';

ChartJS.register(ArcElement, Tooltip, Legend);

interface ConflictSimulatorProps {
  onSimulateResults?: (results: any) => void;
  onSelectCorridors?: (corridors: any[]) => void;
  onSetEpicenter?: (iso3: string) => void;
  selectedEpicenter?: string | null;
}

const PRESET_SCENARIOS = [
  {
    id: 'ukraine_2022',
    title: '🇺🇦 Ukraine (2022)',
    epicenter: 'UKR',
    scenario: 'conflict',
    intensity: 95,
    displaced: 6500000,
    policy: 'universal',
  },
  {
    id: 'syria_2015',
    title: '🇸🇾 Syria (2015)',
    epicenter: 'SYR',
    scenario: 'conflict',
    intensity: 90,
    displaced: 5200000,
    policy: 'kinship_first',
  },
  {
    id: 'sudan_2023',
    title: '🇸🇩 Sudan (2023)',
    epicenter: 'SDN',
    scenario: 'conflict',
    intensity: 85,
    displaced: 3400000,
    policy: 'universal',
  },
  {
    id: 'afghanistan_2021',
    title: '🇦🇫 Afghanistan (2021)',
    epicenter: 'AFG',
    scenario: 'conflict',
    intensity: 85,
    displaced: 2800000,
    policy: 'kinship_first',
  },
  {
    id: 'climate_bangladesh',
    title: '🇧🇩 Bangladesh Climate',
    epicenter: 'BGD',
    scenario: 'climate',
    intensity: 75,
    displaced: 2500000,
    policy: 'universal',
  },
  {
    id: 'venezuela_economic',
    title: '🇻🇪 Venezuela Econ',
    epicenter: 'VEN',
    scenario: 'economic',
    intensity: 80,
    displaced: 4200000,
    policy: 'universal',
  },
];

export default function ConflictSimulator({
  onSimulateResults,
  onSelectCorridors,
  onSetEpicenter,
  selectedEpicenter,
}: ConflictSimulatorProps) {
  const [epicenter, setEpicenter] = useState(selectedEpicenter || 'UKR');
  const [scenarioType, setScenarioType] = useState<'conflict' | 'climate' | 'visa' | 'economic'>('conflict');
  const [intensity, setIntensity] = useState(85);
  const [displacedScale, setDisplacedScale] = useState(2500000);
  const [borderPolicy, setBorderPolicy] = useState<'universal' | 'kinship_first' | 'strict_border'>('universal');
  const [loading, setLoading] = useState(false);
  const [simulationData, setSimulationData] = useState<any>(null);
  const [activeChartTab, setActiveChartTab] = useState<'continent' | 'region' | 'income'>('continent');
  
  // Track last executed parameters to detect staged changes
  const [lastExecuted, setLastExecuted] = useState<{
    epicenter: string;
    scenario: string;
    intensity: number;
    displaced: number;
    policy: string;
  } | null>(null);

  const isParametersChanged =
    !lastExecuted ||
    epicenter !== lastExecuted.epicenter ||
    scenarioType !== lastExecuted.scenario ||
    intensity !== lastExecuted.intensity ||
    displacedScale !== lastExecuted.displaced ||
    borderPolicy !== lastExecuted.policy;

  // Trigger Simulation API call
  const runSimulation = async (
    targetEpicenter = epicenter,
    targetScenario = scenarioType,
    targetIntensity = intensity,
    targetDisplaced = displacedScale,
    targetPolicy = borderPolicy
  ) => {
    setLoading(true);
    if (onSetEpicenter) onSetEpicenter(targetEpicenter);

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario: targetScenario,
          epicenter_iso3: targetEpicenter,
          intensity: targetIntensity,
          displaced_scale: targetDisplaced,
          border_policy: targetPolicy,
        }),
      });

      const data = await res.json();
      if (data.success) {
        setSimulationData(data);
        setLastExecuted({
          epicenter: targetEpicenter,
          scenario: targetScenario,
          intensity: targetIntensity,
          displaced: targetDisplaced,
          policy: targetPolicy,
        });
        if (onSimulateResults) onSimulateResults(data);
        if (onSelectCorridors) onSelectCorridors(data.top_corridors || []);
      }
    } catch (e) {
      console.error('Simulation execution failed:', e);
    } finally {
      setLoading(false);
    }
  };

  // Sync with externally selected epicenter (e.g. from map click)
  useEffect(() => {
    if (selectedEpicenter && selectedEpicenter !== epicenter) {
      setEpicenter(selectedEpicenter);
    }
  }, [selectedEpicenter]);

  // Run initial simulation on mount once
  useEffect(() => {
    runSimulation('UKR', 'conflict', 90, 3500000, 'universal');
  }, []);

  const loadPreset = (p: (typeof PRESET_SCENARIOS)[0]) => {
    setEpicenter(p.epicenter);
    setScenarioType(p.scenario as any);
    setIntensity(p.intensity);
    setDisplacedScale(p.displaced);
    setBorderPolicy(p.policy as any);
    runSimulation(p.epicenter, p.scenario as any, p.intensity, p.displaced, p.policy as any);
  };

  // Chart datasets
  const continentChartData = React.useMemo(() => {
    if (!simulationData?.continent_breakdown) return null;
    const labels = Object.keys(simulationData.continent_breakdown);
    const data = Object.values(simulationData.continent_breakdown) as number[];
    const colors = ['#2563eb', '#d97706', '#059669', '#dc2626', '#7c3aed', '#64748b'];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors.slice(0, labels.length),
          borderColor: '#080c14',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const regionChartData = React.useMemo(() => {
    if (!simulationData?.region_breakdown) return null;
    const entries = Object.entries(simulationData.region_breakdown).sort(
      (a: any, b: any) => b[1] - a[1]
    );
    const topEntries = entries.slice(0, 5);
    const labels = topEntries.map((e) => e[0]);
    const data = topEntries.map((e) => e[1] as number);
    const colors = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors.slice(0, labels.length),
          borderColor: '#080c14',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const incomeChartData = React.useMemo(() => {
    if (!simulationData?.income_breakdown) return null;
    const labels = Object.keys(simulationData.income_breakdown);
    const data = Object.values(simulationData.income_breakdown) as number[];
    const colors = ['#059669', '#2563eb', '#d97706', '#dc2626'];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors.slice(0, labels.length),
          borderColor: '#080c14',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const chartOptions: any = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          color: '#cbd5e1',
          font: { size: 10 },
          boxWidth: 8,
          padding: 6,
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
            const pct = ((val / total) * 100).toFixed(1);
            return ` ${val.toLocaleString()} (${pct}%)`;
          },
        },
      },
    },
    cutout: '60%',
  };

  const epicGeo = getCountryGeo(epicenter);

  return (
    <div className="space-y-3">
      {/* 1. Preset Crisis Scenarios (Compact 6-grid) */}
      <div className="bg-[#0b101b] border border-slate-800 p-2.5 rounded-lg shadow-sm">
        <div className="text-[10.5px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center justify-between">
          <span>Crisis Shock Presets</span>
          <span className="text-[10px] text-slate-500 font-mono">Select to simulate</span>
        </div>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-1.5">
          {PRESET_SCENARIOS.map((p) => {
            const isSelected = epicenter === p.epicenter && scenarioType === p.scenario;
            return (
              <button
                key={p.id}
                onClick={() => loadPreset(p)}
                className={`text-left p-1.5 rounded border text-[11px] transition ${
                  isSelected
                    ? 'bg-[#101827] border-blue-500 text-slate-100 font-semibold'
                    : 'bg-[#080c14] border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <div className="truncate">{p.title}</div>
                <div className="text-[9.5px] text-slate-500 font-mono">
                  {(p.displaced / 1000000).toFixed(1)}M
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Simulation Parameter Inputs Deck */}
      <div className={`bg-[#0b101b] border ${isParametersChanged ? 'border-blue-500/60 ring-1 ring-blue-500/30' : 'border-slate-800'} p-3 rounded-lg shadow-sm space-y-2.5 transition-colors`}>
        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-100">Displacement Parameter Cockpit</span>
            {isParametersChanged && (
              <span className="text-[9.5px] bg-blue-950 text-blue-300 border border-blue-800/80 px-1.5 py-0.2 rounded font-mono animate-pulse">
                ● Staged Changes
              </span>
            )}
          </div>
          <span className="text-[9.5px] bg-red-950 text-red-400 border border-red-800/80 px-1.5 py-0.2 rounded font-mono font-medium">
            Gravity Engine
          </span>
        </div>

        {/* Epicenter & Scenario Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <div>
            <label className="block text-[10.5px] text-slate-400 mb-0.5">Epicenter (Origin):</label>
            <select
              value={epicenter}
              onChange={(e) => setEpicenter(e.target.value)}
              className="w-full bg-[#080c14] border border-slate-700 text-slate-200 px-2 py-1 rounded text-xs focus:outline-none focus:border-blue-500"
            >
              {Object.values(COUNTRY_GEO_MAP).map((c) => (
                <option key={c.iso3} value={c.iso3}>
                  {c.flag} {c.name} ({c.iso3})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[10.5px] text-slate-400 mb-0.5">Crisis Catalyst:</label>
            <select
              value={scenarioType}
              onChange={(e) => setScenarioType(e.target.value as any)}
              className="w-full bg-[#080c14] border border-slate-700 text-slate-200 px-2 py-1 rounded text-xs focus:outline-none focus:border-blue-500"
            >
              <option value="conflict">Armed Conflict / War</option>
              <option value="climate">Climate / Sea-Level Surge</option>
              <option value="visa">Border / Visa Policy Shift</option>
              <option value="economic">Economic / Hyperinflation</option>
            </select>
          </div>
        </div>

        {/* Sliders: Displaced Population & Intensity */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
          <div className="space-y-0.5">
            <div className="flex items-center justify-between text-[10.5px]">
              <span className="text-slate-400">Displaced Mass:</span>
              <span className="font-mono text-blue-400 font-semibold">
                {displacedScale >= 1000000
                  ? `${(displacedScale / 1000000).toFixed(2)}M`
                  : `${(displacedScale / 1000).toFixed(0)}k`}
              </span>
            </div>
            <input
              type="range"
              min={100000}
              max={8000000}
              step={100000}
              value={displacedScale}
              onChange={(e) => setDisplacedScale(Number(e.target.value))}
              className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-blue-500"
            />
          </div>

          <div className="space-y-0.5">
            <div className="flex items-center justify-between text-[10.5px]">
              <span className="text-slate-400">Shock Severity:</span>
              <span className="font-mono text-amber-400 font-semibold">{intensity}%</span>
            </div>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={intensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-amber-500"
            />
          </div>
        </div>

        {/* Border Absorption Policy & Execute Button */}
        <div className="flex items-center gap-2 pt-0.5">
          <div className="flex flex-1 bg-[#080c14] border border-slate-800 p-0.5 rounded text-[10.5px]">
            {[
              { id: 'universal', label: 'Universal' },
              { id: 'kinship_first', label: 'Kinship' },
              { id: 'strict_border', label: 'Strict' },
            ].map((pol) => (
              <button
                key={pol.id}
                onClick={() => setBorderPolicy(pol.id as any)}
                className={`flex-1 py-1 rounded text-center transition ${
                  borderPolicy === pol.id
                    ? 'bg-slate-200 text-slate-950 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {pol.label}
              </button>
            ))}
          </div>

          <button
            onClick={() => runSimulation(epicenter, scenarioType, intensity, displacedScale, borderPolicy)}
            disabled={loading}
            className={`px-4 py-1.5 font-medium rounded text-xs transition flex items-center gap-1.5 shadow-sm ${
              isParametersChanged
                ? 'bg-blue-600 hover:bg-blue-500 text-white font-semibold ring-2 ring-blue-400/50 shadow-blue-500/20'
                : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
            }`}
          >
            {loading ? (
              <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
            ) : (
              <span className="text-sm leading-none">⚡</span>
            )}
            <span>{loading ? 'Computing...' : isParametersChanged ? 'Compute Simulation' : 'Recompute'}</span>
          </button>
        </div>
      </div>

      {/* 3. Absorption Forecast Distribution Chart */}
      <div className="bg-[#0b101b] border border-slate-800 p-2.5 rounded-lg shadow-sm space-y-1.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1">
          <span className="text-[11px] font-semibold text-slate-200">
            Absorption Distribution ({epicGeo.name})
          </span>
          <div className="flex bg-[#080c14] border border-slate-800 p-0.5 rounded text-[10px]">
            {(['continent', 'region', 'income'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveChartTab(tab)}
                className={`px-1.5 py-0.2 rounded capitalize transition ${
                  activeChartTab === tab
                    ? 'bg-slate-200 text-slate-950 font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        <div className="h-32 w-full flex items-center justify-center">
          {activeChartTab === 'continent' && continentChartData && (
            <Doughnut data={continentChartData} options={chartOptions} />
          )}
          {activeChartTab === 'region' && regionChartData && (
            <Doughnut data={regionChartData} options={chartOptions} />
          )}
          {activeChartTab === 'income' && incomeChartData && (
            <Pie data={incomeChartData} options={chartOptions} />
          )}
        </div>
      </div>

      {/* 4. Top Recipient States Table */}
      <div className="bg-[#0b101b] border border-slate-800 p-2.5 rounded-lg shadow-sm space-y-1.5">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1">
          <span className="text-[11px] font-semibold text-slate-200">
            Top Projected Recipient States
          </span>
          <span className="text-[9.5px] text-slate-500 font-mono">
            {displacedScale.toLocaleString()} Displaced
          </span>
        </div>

        <div className="overflow-x-auto max-h-48 overflow-y-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-[10px] text-slate-400 border-b border-slate-800 font-mono">
                <th className="py-1 px-1.5">#</th>
                <th className="py-1 px-1.5">State</th>
                <th className="py-1 px-1.5">Projected Influx</th>
                <th className="py-1 px-1.5">Distance</th>
                <th className="py-1 px-1.5">Strain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {simulationData?.top_recipients?.slice(0, 8).map((r: any, idx: number) => (
                <tr key={r.iso3} className="hover:bg-slate-800/40 transition">
                  <td className="py-1 px-1.5 text-slate-500 font-mono text-[10px]">{idx + 1}</td>
                  <td className="py-1 px-1.5 font-medium text-slate-100 flex items-center gap-1 text-[11px]">
                    <span>{r.flag}</span>
                    <span className="truncate max-w-[90px]">{r.country_name}</span>
                  </td>
                  <td className="py-1 px-1.5 font-mono font-semibold text-blue-400 text-[11px]">
                    {r.predicted_influx.toLocaleString()}
                  </td>
                  <td className="py-1 px-1.5 text-slate-400 font-mono text-[10px]">
                    {r.dist_km.toLocaleString()} km
                  </td>
                  <td className="py-1 px-1.5">
                    <span
                      className="px-1.5 py-0.2 rounded text-[9.5px] font-semibold"
                      style={{
                        backgroundColor: `${r.strain_color}20`,
                        color: r.strain_color,
                        border: `1px solid ${r.strain_color}40`,
                      }}
                    >
                      {r.strain_category}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
