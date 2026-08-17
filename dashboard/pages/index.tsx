import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Header from '../components/Header';
import WorldMapInteractive from '../components/WorldMapInteractive';
import ConflictSimulator from '../components/ConflictSimulator';
import InteractivePieCharts from '../components/InteractivePieCharts';
import NetworkTopologyView from '../components/NetworkTopologyView';
import CorridorComparator from '../components/CorridorComparator';

export default function Home() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [year, setYear] = useState<number>(2025);
  const [activeView, setActiveView] = useState<'map' | 'simulator' | 'charts' | 'topology' | 'comparator'>('map');
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [mapMode, setMapMode] = useState<'corridors' | 'centrality' | 'communities' | 'simulation'>('corridors');

  // Crisis Simulator State
  const [simulationEpicenter, setSimulationEpicenter] = useState<string | null>('UKR');
  const [simulationCorridors, setSimulationCorridors] = useState<any[]>([]);
  const [simulationResults, setSimulationResults] = useState<any>(null);

  // Load unified API dataset
  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await fetch('/api/nodes_and_nations');
        const json = await res.json();
        if (json.success) {
          setData(json);
        }
      } catch (err) {
        console.error('Failed to load nodes and nations dataset:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Map nodes lookup table for fast access
  const nodesMap = React.useMemo(() => {
    if (!data?.nodes_master) return {};
    const map: Record<string, any> = {};
    data.nodes_master
      .filter((n: any) => Number(n.year) === year)
      .forEach((n: any) => {
        map[n.iso3] = n;
      });
    return map;
  }, [data, year]);

  // Active corridors to pass to the interactive world map
  const activeMapCorridors = React.useMemo(() => {
    if (activeView === 'simulator' || mapMode === 'simulation') {
      return simulationCorridors.length > 0 ? simulationCorridors : (data?.edges || []);
    }
    return (data?.edges || []).filter((e: any) => Number(e.year) === year);
  }, [activeView, mapMode, simulationCorridors, data, year]);

  return (
    <div className="min-h-screen bg-[#06090e] text-slate-100 flex flex-col font-sans">
      <Head>
        <title>Nodes & Nations // Global Migration Topology & Simulation Platform</title>
        <meta
          name="description"
          content="A complex network analysis and real-world crisis displacement simulation platform for global migration flows (1990-2025)."
        />
      </Head>

      {/* Global Header HUD with Time-Travel Scrubber */}
      <Header
        currentYear={year}
        onYearChange={setYear}
        activeView={activeView}
        onViewChange={(v) => {
          setActiveView(v);
          if (v === 'simulator') setMapMode('simulation');
          else if (mapMode === 'simulation') setMapMode('corridors');
        }}
        summaryStats={data?.summary_stats || []}
        networkSummary={data?.network_summary || []}
        modularityScores={data?.modularity_scores || []}
      />

      <main className="max-w-7xl w-full mx-auto px-4 lg:px-8 py-5 space-y-5 flex-1">
        {loading ? (
          <div className="h-[520px] bg-[#0b101b] border border-slate-800 rounded-xl flex flex-col items-center justify-center space-y-3">
            <div className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <div className="text-xs font-mono text-slate-400">Loading Global Migration Topology Engine...</div>
          </div>
        ) : (
          <>
            {activeView === 'simulator' ? (
              /* Stock Market / Financial Terminal Layout for Crisis Simulator */
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
                {/* Left Column (7 cols): Map + Crisis Vectors Telemetry Bar */}
                <div className="lg:col-span-7 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse"></span>
                      <h2 className="font-semibold text-slate-200 text-xs uppercase tracking-wider">
                        Crisis Displacement Surface
                      </h2>
                      <span className="text-xs text-slate-400 font-mono">
                        (Shock Epicenter: <strong className="text-red-400">{simulationEpicenter || 'UKR'}</strong>)
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[10px] bg-red-950 text-red-300 border border-red-800/80 px-2 py-0.5 rounded font-mono font-medium">
                        Gravity Displacement Routing
                      </span>
                    </div>
                  </div>

                  {/* Interactive World Map in Simulator Mode */}
                  <WorldMapInteractive
                    corridors={activeMapCorridors}
                    selectedCountry={selectedCountry}
                    onSelectCountry={(iso) => {
                      setSelectedCountry(iso === selectedCountry ? null : iso);
                      if (iso) {
                        setSimulationEpicenter(iso);
                      }
                    }}
                    mode="simulation"
                    nodesData={nodesMap}
                    simulationEpicenter={simulationEpicenter}
                    year={year}
                    heightClass="h-[460px] xl:h-[490px]"
                  />

                  {/* Live Shock Telemetry Ticker Card (Stock Market Terminal Style) */}
                  <div className="bg-[#0b101b] border border-slate-800 p-3 rounded-xl shadow-md space-y-2">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[10.5px] font-semibold text-blue-400 uppercase tracking-wide">
                          Live Displacement Telemetry
                        </span>
                        <span className="text-slate-600">|</span>
                        <span className="text-[11px] text-slate-300 font-medium">
                          {simulationResults?.epicenter_name || 'Ukraine'} ({simulationEpicenter})
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-400 font-mono">
                        {simulationResults?.scenario?.toUpperCase() || 'CONFLICT'} SHOCK
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      <div className="p-2 bg-[#080c14] border border-slate-800/80 rounded-lg">
                        <div className="text-[10px] text-slate-500 uppercase font-mono">Displaced Mass</div>
                        <div className="text-sm font-semibold font-mono text-blue-400 mt-0.5">
                          {simulationResults?.displaced_total ? (simulationResults.displaced_total / 1000000).toFixed(2) + 'M' : '3.50M'}
                        </div>
                        <div className="text-[9.5px] text-slate-400 mt-0.5">
                          {simulationResults?.intensity || 90}% Severity
                        </div>
                      </div>

                      <div className="p-2 bg-[#080c14] border border-slate-800/80 rounded-lg">
                        <div className="text-[10px] text-slate-500 uppercase font-mono">Primary Sink</div>
                        <div className="text-xs font-semibold font-mono text-slate-200 mt-0.5 truncate">
                          {simulationResults?.top_recipients?.[0]?.country_name || 'Poland'}
                        </div>
                        <div className="text-[9.5px] text-red-400 font-mono mt-0.5">
                          {simulationResults?.top_recipients?.[0]?.predicted_influx?.toLocaleString() || '1,820,000'}
                        </div>
                      </div>

                      <div className="p-2 bg-[#080c14] border border-slate-800/80 rounded-lg">
                        <div className="text-[10px] text-slate-500 uppercase font-mono">Mean Distance</div>
                        <div className="text-xs font-semibold font-mono text-amber-400 mt-0.5">
                          {simulationResults?.top_recipients?.[0]?.dist_km ? Math.round(simulationResults.top_recipients.reduce((acc: number, r: any) => acc + r.dist_km, 0) / simulationResults.top_recipients.length).toLocaleString() + ' km' : '1,420 km'}
                        </div>
                        <div className="text-[9.5px] text-slate-400 mt-0.5">
                          Decay Gradient
                        </div>
                      </div>

                      <div className="p-2 bg-[#080c14] border border-slate-800/80 rounded-lg">
                        <div className="text-[10px] text-slate-500 uppercase font-mono">Policy Filter</div>
                        <div className="text-xs font-semibold font-mono text-emerald-400 mt-0.5 capitalize truncate">
                          {simulationResults?.policy?.replace('_', ' ') || 'Universal'}
                        </div>
                        <div className="text-[9.5px] text-slate-400 mt-0.5">
                          Border Regimes
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Column (5 cols): Conflict Simulator Cockpit & Distribution Deck */}
                <div className="lg:col-span-5">
                  <ConflictSimulator
                    onSetEpicenter={setSimulationEpicenter}
                    onSelectCorridors={setSimulationCorridors}
                    onSimulateResults={setSimulationResults}
                    selectedEpicenter={simulationEpicenter}
                  />
                </div>
              </div>
            ) : (
              /* Standard Full-Width Layout for Map, Topology, Charts, Comparator */
              <>
                {/* World Map Section with Clean Layer Toggle */}
                <section className="space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <h2 className="font-semibold text-slate-200 text-xs uppercase tracking-wider">
                        Planetary Vector Surface ({year})
                      </h2>
                      <span className="text-xs text-slate-500 font-mono">
                        ({data?.country_list?.length || 235} Sovereign States • {activeMapCorridors.length} Monitored Flow Vectors)
                      </span>
                    </div>

                    {/* Clean Map Layer Toggle */}
                    <div className="flex items-center gap-1 bg-[#0b101b] border border-slate-800 p-0.5 rounded-lg text-xs">
                      <button
                        onClick={() => setMapMode('corridors')}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                          mapMode === 'corridors'
                            ? 'bg-slate-200 text-slate-950 font-semibold shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Bilateral Corridors
                      </button>
                      <button
                        onClick={() => setMapMode('communities')}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                          mapMode === 'communities'
                            ? 'bg-slate-200 text-slate-950 font-semibold shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Alliance Clusters
                      </button>
                    </div>
                  </div>

                  {/* Interactive World Map Component */}
                  <WorldMapInteractive
                    corridors={activeMapCorridors}
                    selectedCountry={selectedCountry}
                    onSelectCountry={(iso) => {
                      setSelectedCountry(iso === selectedCountry ? null : iso);
                    }}
                    mode={mapMode}
                    nodesData={nodesMap}
                    simulationEpicenter={null}
                    year={year}
                  />
                </section>

                {/* Dynamic Analytics Deck based on Active Tab */}
                <section className="transition-all duration-200">
                  {activeView === 'charts' && (
                    <InteractivePieCharts
                      edges={data?.edges || []}
                      nodes={data?.nodes_master || []}
                      year={year}
                    />
                  )}

                  {activeView === 'topology' && (
                    <NetworkTopologyView
                      nodes={data?.nodes_master || []}
                      modularityScores={data?.modularity_scores || []}
                      boundaryNodes={data?.boundary_nodes || []}
                      regressionCoeffs={data?.regression_coefficients || []}
                      regressionComparison={data?.regression_comparison || []}
                      year={year}
                    />
                  )}

                  {activeView === 'comparator' && (
                    <CorridorComparator
                      nodes={data?.nodes_master || []}
                      edges={data?.edges || []}
                      year={year}
                    />
                  )}

                  {activeView === 'map' && (
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                      {/* Left: Quick Corridor Table */}
                  <div className="lg:col-span-8 bg-[#0b101b] border border-slate-800 p-4 rounded-xl shadow-md space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                      <div>
                        <h3 className="font-semibold text-slate-100 text-sm">
                          Top Bilateral Migration Corridors ({year})
                        </h3>
                        <p className="text-xs text-slate-400">
                          Ranked by total registered bilateral migrant stock volume
                        </p>
                      </div>
                      <span className="text-[11px] bg-blue-950 text-blue-300 border border-blue-800/80 px-2 py-0.5 rounded font-mono">
                        {(data?.edges || []).filter((e: any) => Number(e.year) === year).length} Monitored Corridors
                      </span>
                    </div>

                    <div className="overflow-x-auto max-h-72 overflow-y-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="text-[10.5px] text-slate-400 border-b border-slate-800 font-mono">
                            <th className="py-2 px-2.5">#</th>
                            <th className="py-2 px-2.5">Origin State</th>
                            <th className="py-2 px-2.5">Destination State</th>
                            <th className="py-2 px-2.5">Migrant Stock</th>
                            <th className="py-2 px-2.5">Geography Rel</th>
                            <th className="py-2 px-2.5">Income Tier</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 text-slate-300">
                          {(data?.edges || [])
                            .filter((e: any) => Number(e.year) === year)
                            .slice(0, 15)
                            .map((e: any, idx: number) => (
                              <tr
                                key={idx}
                                onClick={() => setSelectedCountry(e.origin_iso3)}
                                className="hover:bg-slate-800/40 cursor-pointer transition"
                              >
                                <td className="py-2 px-2.5 font-mono text-slate-500 text-[11px]">#{idx + 1}</td>
                                <td className="py-2 px-2.5 font-medium text-slate-100 flex items-center gap-1.5">
                                  <span>{e.origin_flag}</span>
                                  <span>{e.origin_country_name || e.origin_iso3}</span>
                                  <span className="text-[10px] text-slate-500 font-mono">({e.origin_iso3})</span>
                                </td>
                                <td className="py-2 px-2.5 font-medium text-slate-100">
                                  <span className="inline-flex items-center gap-1.5">
                                    <span>{e.dest_flag}</span>
                                    <span>{e.dest_country_name || e.dest_iso3}</span>
                                    <span className="text-[10px] text-slate-500 font-mono">({e.dest_iso3})</span>
                                  </span>
                                </td>
                                <td className="py-2 px-2.5 font-mono font-semibold text-blue-400">
                                  {Math.round(Number(e.weight) || 0).toLocaleString()}
                                </td>
                                <td className="py-2 px-2.5">
                                  <span
                                    className={`px-2 py-0.5 rounded text-[10px] ${
                                      e.same_continent === 1
                                        ? 'bg-blue-950 text-blue-300 border border-blue-800'
                                        : 'bg-slate-800 text-slate-400'
                                    }`}
                                  >
                                    {e.same_continent === 1 ? 'Intra-Continental' : 'Inter-Continental'}
                                  </span>
                                </td>
                                <td className="py-2 px-2.5">
                                  <span className="text-[10.5px] text-slate-400 font-mono">
                                    {e.dest_income_group || 'High income'}
                                  </span>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Right: Quick Action Hub & Mini-Summary */}
                  <div className="lg:col-span-4 space-y-4">
                    {/* Simulator Callout Card */}
                    <div className="p-4 bg-[#0d1422] border border-slate-700/80 rounded-xl shadow-md space-y-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-semibold text-red-400 uppercase tracking-wide">
                          Simulation Module
                        </span>
                        <span className="w-2 h-2 rounded-full bg-red-500"></span>
                      </div>
                      <h4 className="font-semibold text-slate-100 text-sm">
                        Crisis Displacement Simulator
                      </h4>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        Evaluate gravity-routing models for acute shocks across conflict, climate, border shifts, and economic asymmetry with localized capacity strain assessments.
                      </p>
                      <button
                        onClick={() => {
                          setActiveView('simulator');
                          setMapMode('simulation');
                        }}
                        className="w-full py-2 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow-sm"
                      >
                        <span>Launch Crisis Simulator</span>
                      </button>
                    </div>

                    {/* Quick Macro Indicators */}
                    <div className="p-4 bg-[#0b101b] border border-slate-800 rounded-xl shadow-md space-y-2.5">
                      <h4 className="font-semibold text-slate-200 text-xs uppercase tracking-wide">Systemic Insights</h4>
                      <div className="space-y-2 text-xs">
                        <div className="p-2.5 bg-[#080c14] border border-slate-800 rounded-lg">
                          <div className="text-slate-400 font-medium text-xs">Integration Trend:</div>
                          <div className="text-slate-300 text-[11px] mt-0.5">
                            Network modularity declined by <strong>20.5%</strong> from 1990 to 2025, demonstrating an increasingly interconnected global migration architecture.
                          </div>
                        </div>
                        <div className="p-2.5 bg-[#080c14] border border-slate-800 rounded-lg">
                          <div className="text-slate-400 font-medium text-xs">Economic Pull:</div>
                          <div className="text-slate-300 text-[11px] mt-0.5">
                            GDP per capita disparity explains over <strong>60% of variance</strong> in corridor choice, compounded by established historical diaspora kinship.
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </section>
          </>
        )}
      </>
    )}
  </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-[#06090e] py-3 text-center text-xs text-slate-500 font-mono">
        Nodes and Nations // Topological Study of Global Human Movement (1990–2025)
      </footer>
    </div>
  );
}
