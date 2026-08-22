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
    policy: 'cultural_kinship',
  },
  {
    id: 'myanmar_2017',
    title: '🇲🇲 Myanmar (2017)',
    epicenter: 'MMR',
    scenario: 'cultural_kinship',
    intensity: 90,
    displaced: 1200000,
    policy: 'cultural_kinship',
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
    id: 'climate_bangladesh',
    title: '🇧🇩 Bangladesh Climate',
    epicenter: 'BGD',
    scenario: 'climate',
    intensity: 75,
    displaced: 2500000,
    policy: 'integration_havens',
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
  const [scenarioType, setScenarioType] = useState<'conflict' | 'climate' | 'visa' | 'economic' | 'cultural_kinship'>('conflict');
  const [intensity, setIntensity] = useState(85);
  const [displacedScale, setDisplacedScale] = useState(2500000);
  const [borderPolicy, setBorderPolicy] = useState<'universal' | 'kinship_first' | 'cultural_kinship' | 'integration_havens' | 'strict_border'>('universal');
  const [loading, setLoading] = useState(false);
  const [simulationData, setSimulationData] = useState<any>(null);
  const [activeChartTab, setActiveChartTab] = useState<'continent' | 'region' | 'income' | 'religion' | 'integration'>('continent');
  
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
    const heritageColors = ['#1e3a5f', '#945920', '#28534b', '#8b2626', '#5c3858', '#4f5d30'];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: heritageColors.slice(0, labels.length),
          borderColor: '#fbf9f4',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const regionChartData = React.useMemo(() => {
    if (!simulationData?.region_breakdown) return null;
    const labels = Object.keys(simulationData.region_breakdown).slice(0, 6);
    const data = labels.map((k) => simulationData.region_breakdown[k]);
    const heritageColors = ['#1e3a5f', '#28534b', '#945920', '#5c3858', '#8b2626', '#3b5266'];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: heritageColors.slice(0, labels.length),
          borderColor: '#fbf9f4',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const incomeChartData = React.useMemo(() => {
    if (!simulationData?.income_breakdown) return null;
    const labels = Object.keys(simulationData.income_breakdown);
    const data = Object.values(simulationData.income_breakdown) as number[];
    const colorMap: Record<string, string> = {
      'High income': '#28534b',
      'Upper middle income': '#1e3a5f',
      'Lower middle income': '#945920',
      'Low income': '#8b2626',
    };
    const colors = labels.map((l) => colorMap[l] || '#786b59');

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors,
          borderColor: '#fbf9f4',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const religionChartData = React.useMemo(() => {
    if (!simulationData?.religion_breakdown) return null;
    const labels = Object.keys(simulationData.religion_breakdown);
    const data = Object.values(simulationData.religion_breakdown) as number[];
    const colorMap: Record<string, string> = {
      'Christianity': '#1e3a5f',
      'Islam': '#28534b',
      'Hinduism': '#945920',
      'Buddhism': '#8c6d23',
      'Judaism': '#5c3858',
      'Unaffiliated': '#556270',
    };
    const colors = labels.map((l) => colorMap[l] || '#786b59');

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors,
          borderColor: '#fbf9f4',
          borderWidth: 1.5,
        },
      ],
    };
  }, [simulationData]);

  const integrationChartData = React.useMemo(() => {
    if (!simulationData?.integration_breakdown) return null;
    const labels = Object.keys(simulationData.integration_breakdown);
    const data = Object.values(simulationData.integration_breakdown) as number[];
    const colors = ['#28534b', '#1e3a5f', '#8b2626'];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: colors,
          borderColor: '#fbf9f4',
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
          color: '#4a3d2e',
          font: { size: 10, family: 'ui-sans-serif, system-ui' },
          boxWidth: 8,
          padding: 6,
        },
      },
      tooltip: {
        backgroundColor: '#ffffff',
        borderColor: '#d8ccb8',
        borderWidth: 1,
        titleColor: '#1e3a5f',
        bodyColor: '#1e293b',
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
    cutout: '60%',
  };

  const epicGeo = getCountryGeo(epicenter);

  return (
    <div className="space-y-3">
      {/* 1. Preset Crisis Scenarios (Compact 6-grid) */}
      <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-3 rounded-lg shadow-sm">
        <div className="text-[10.5px] font-semibold text-[#786b59] uppercase tracking-wider mb-2 flex items-center justify-between font-mono">
          <span>Crisis Shock Presets</span>
          <span className="text-[10px] text-[#948064]">Select to simulate</span>
        </div>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-1.5">
          {PRESET_SCENARIOS.map((p) => {
            const isSelected = epicenter === p.epicenter && scenarioType === p.scenario;
            return (
              <button
                key={p.id}
                onClick={() => loadPreset(p)}
                className={`text-left p-2 rounded-lg border text-[11px] transition shadow-xs ${
                  isSelected
                    ? 'bg-[#1e3a5f] border-[#14253e] text-white font-semibold'
                    : 'bg-[#f4efe6] border-[#ded3be] text-[#5a5043] hover:border-[#b8a589] hover:text-[#1a202c]'
                }`}
              >
                <div className="truncate font-medium">{p.title}</div>
                <div className={`text-[9.5px] font-mono mt-0.5 ${isSelected ? 'text-blue-100' : 'text-[#786b59]'}`}>
                  {(p.displaced / 1000000).toFixed(1)}M
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Simulation Parameter Inputs Deck */}
      <div className={`bg-[#fbf9f4] border ${isParametersChanged ? 'border-[#945920] ring-1 ring-[#945920]/30' : 'border-[#d8ccb8]'} p-3.5 rounded-lg shadow-sm space-y-3 transition-colors`}>
        <div className="flex items-center justify-between border-b border-[#ded3be] pb-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-[#1a202c] font-serif">Displacement Parameter Cockpit</span>
            {isParametersChanged && (
              <span className="text-[9.5px] bg-[#fcf4e8] text-[#945920] border border-[#e8d5bc] px-1.5 py-0.2 rounded font-mono font-semibold animate-pulse">
                ● Staged Changes
              </span>
            )}
          </div>
          <span className="text-[9.5px] bg-[#fcf4e8] text-[#945920] border border-[#e8d5bc] px-1.5 py-0.2 rounded font-mono font-bold">
            Gravity Engine
          </span>
        </div>

        {/* Epicenter & Scenario Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          <div>
            <label className="block text-[10.5px] text-[#645d52] font-medium mb-1">Epicenter (Origin):</label>
            <select
              value={epicenter}
              onChange={(e) => setEpicenter(e.target.value)}
              className="w-full bg-[#ffffff] border border-[#ded3be] text-[#1e293b] px-2.5 py-1.5 rounded-lg text-xs focus:outline-none focus:border-[#1e3a5f]"
            >
              {Object.values(COUNTRY_GEO_MAP).map((c) => (
                <option key={c.iso3} value={c.iso3}>
                  {c.flag} {c.name} ({c.iso3})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[10.5px] text-[#645d52] font-medium mb-1">Crisis Catalyst:</label>
            <select
              value={scenarioType}
              onChange={(e) => setScenarioType(e.target.value as any)}
              className="w-full bg-[#ffffff] border border-[#ded3be] text-[#1e293b] px-2.5 py-1.5 rounded-lg text-xs focus:outline-none focus:border-[#1e3a5f]"
            >
              <option value="conflict">Armed Conflict / War</option>
              <option value="cultural_kinship">Cultural & Religious Kinship</option>
              <option value="climate">Climate / Sea-Level Surge</option>
              <option value="visa">Border / Visa Policy Shift</option>
              <option value="economic">Economic / Hyperinflation</option>
            </select>
          </div>
        </div>

        {/* Sliders: Displaced Population & Intensity */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs">
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10.5px]">
              <span className="text-[#645d52] font-medium">Displaced Mass:</span>
              <span className="font-mono text-[#1e3a5f] font-bold">
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
              className="w-full h-1.5 bg-[#e0d6c4] rounded appearance-none cursor-pointer accent-[#1e3a5f]"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10.5px]">
              <span className="text-[#645d52] font-medium">Shock Severity:</span>
              <span className="font-mono text-[#945920] font-bold">{intensity}%</span>
            </div>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={intensity}
              onChange={(e) => setIntensity(Number(e.target.value))}
              className="w-full h-1.5 bg-[#e0d6c4] rounded appearance-none cursor-pointer accent-[#945920]"
            />
          </div>
        </div>

        {/* Border Absorption Policy & Execute Button */}
        <div className="flex flex-wrap items-center gap-2.5 pt-1">
          <div className="flex flex-1 flex-wrap bg-[#e8e0cf] border border-[#d8ccb8] p-0.5 rounded-lg text-[10px]">
            {[
              { id: 'universal', label: 'Universal' },
              { id: 'kinship_first', label: 'Diaspora' },
              { id: 'cultural_kinship', label: 'Religion Kin' },
              { id: 'integration_havens', label: 'MIPEX Havens' },
              { id: 'strict_border', label: 'Strict' },
            ].map((pol) => (
              <button
                key={pol.id}
                onClick={() => setBorderPolicy(pol.id as any)}
                className={`flex-1 min-w-[50px] py-1 px-1.5 rounded text-center transition ${
                  borderPolicy === pol.id
                    ? 'bg-[#1e3a5f] text-white font-semibold shadow-xs'
                    : 'text-[#5a5043] hover:text-[#1a202c]'
                }`}
              >
                {pol.label}
              </button>
            ))}
          </div>

          <button
            onClick={() => runSimulation(epicenter, scenarioType, intensity, displacedScale, borderPolicy)}
            disabled={loading}
            className={`px-4 py-2 font-medium rounded-lg text-xs transition flex items-center gap-1.5 shadow-sm ${
              isParametersChanged
                ? 'bg-[#945920] hover:bg-[#7e4d2a] text-white font-bold ring-2 ring-[#945920]/40'
                : 'bg-[#1e3a5f] hover:bg-[#162c46] text-white'
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
      <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-3 rounded-lg shadow-sm space-y-2">
        <div className="flex items-center justify-between border-b border-[#ded3be] pb-1.5">
          <span className="text-xs font-bold text-[#1a202c] font-serif">
            Absorption Distribution ({epicGeo.name})
          </span>
          <div className="flex bg-[#e8e0cf] border border-[#d8ccb8] p-0.5 rounded text-[10px]">
            {(['continent', 'region', 'income', 'religion', 'integration'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveChartTab(tab)}
                className={`px-1.5 py-0.5 rounded capitalize transition ${
                  activeChartTab === tab
                    ? 'bg-[#1e3a5f] text-white font-bold shadow-xs'
                    : 'text-[#5a5043] hover:text-[#1a202c]'
                }`}
              >
                {tab === 'integration' ? 'MIPEX' : tab}
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
          {activeChartTab === 'religion' && religionChartData && (
            <Doughnut data={religionChartData} options={chartOptions} />
          )}
          {activeChartTab === 'integration' && integrationChartData && (
            <Pie data={integrationChartData} options={chartOptions} />
          )}
        </div>
      </div>

      {/* 4. Top Recipient States Table */}
      <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-3 rounded-lg shadow-sm space-y-2">
        <div className="flex items-center justify-between border-b border-[#ded3be] pb-1.5">
          <span className="text-xs font-bold text-[#1a202c] font-serif">
            Top Projected Recipient States
          </span>
          <span className="text-[10px] text-[#786b59] font-mono font-semibold">
            {displacedScale.toLocaleString()} Displaced
          </span>
        </div>

        <div className="overflow-x-auto max-h-48 overflow-y-auto border border-[#ded3be] rounded-lg">
          <table className="w-full text-left text-xs bg-[#ffffff]">
            <thead>
              <tr className="text-[10px] text-[#6d5c46] border-b border-[#ded3be] font-mono bg-[#f4efe6]">
                <th className="py-1.5 px-2">#</th>
                <th className="py-1.5 px-2">State</th>
                <th className="py-1.5 px-2">Projected Influx</th>
                <th className="py-1.5 px-2">Distance</th>
                <th className="py-1.5 px-2">Strain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#f0e9dc] text-[#1e293b]">
              {simulationData?.top_recipients?.slice(0, 8).map((r: any, idx: number) => (
                <tr key={r.iso3} className="hover:bg-[#fbf7f0] transition">
                  <td className="py-1.5 px-2 text-[#8a7c6a] font-mono text-[10px]">#{idx + 1}</td>
                  <td className="py-1.5 px-2 font-semibold text-[#1a202c] flex items-center gap-1.5 text-[11px]">
                    <span>{r.flag}</span>
                    <span className="truncate max-w-[90px]">{r.country_name}</span>
                  </td>
                  <td className="py-1.5 px-2 font-mono font-bold text-[#1e3a5f] text-[11px]">
                    {r.predicted_influx.toLocaleString()}
                  </td>
                  <td className="py-1.5 px-2 text-[#645d52] font-mono text-[10px]">
                    {r.dist_km.toLocaleString()} km
                  </td>
                  <td className="py-1.5 px-2">
                    <span
                      className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                      style={{
                        backgroundColor: `${r.strain_color}18`,
                        color: r.strain_color === '#ef4444' ? '#8b2626' : r.strain_color,
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
