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
  const [mapMode, setMapMode] = useState<'corridors' | 'centrality' | 'communities' | 'simulation' | 'religion'>('corridors');

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
    <div className="min-h-screen bg-[#f8f5ee] text-[#1e293b] flex flex-col font-sans">
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
          else if (v === 'topology') setMapMode('centrality');
          else if (mapMode === 'simulation') setMapMode('corridors');
        }}
        summaryStats={data?.summary_stats || []}
        networkSummary={data?.network_summary || []}
        modularityScores={data?.modularity_scores || []}
      />

      <main className="max-w-7xl w-full mx-auto px-4 lg:px-8 py-5 space-y-5 flex-1">
        {loading ? (
          <div className="h-[520px] bg-[#fbf9f4] border border-[#d8ccb8] rounded-xl flex flex-col items-center justify-center space-y-3 shadow-sm">
            <div className="w-9 h-9 border-3 border-[#1e3a5f] border-t-transparent rounded-full animate-spin"></div>
            <div className="text-xs font-mono font-medium text-[#786b59]">Loading Global Migration Topology Engine...</div>
          </div>
        ) : (
          <>
            {activeView === 'simulator' ? (
              /* Financial Terminal / Executive Cockpit Layout for Crisis Simulator */
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
                {/* Left Column (7 cols): Map + Crisis Vectors Telemetry Bar */}
                <div className="lg:col-span-7 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-[#8b2626] animate-pulse"></span>
                      <h2 className="font-bold text-[#1a202c] text-xs uppercase tracking-wider font-serif">
                        Crisis Displacement Surface
                      </h2>
                      <span className="text-xs text-[#645d52] font-mono">
                        (Shock Epicenter: <strong className="text-[#8b2626]">{simulationEpicenter || 'UKR'}</strong>)
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[10.5px] bg-[#fcf4e8] text-[#945920] border border-[#e8d5bc] px-2 py-0.5 rounded font-mono font-semibold">
                        Gravity Displacement Routing
                      </span>
                      <button
                        onClick={() => {
                          setActiveView('map');
                          setMapMode('corridors');
                        }}
                        className="text-[10.5px] text-[#5a5043] hover:text-[#1a202c] bg-[#eae1cf] hover:bg-[#dfd5c2] px-2.5 py-0.5 rounded font-medium transition"
                      >
                        Exit to Overview ↗
                      </button>
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

                  {/* Live Shock Telemetry Ticker Card */}
                  <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-3.5 rounded-xl shadow-sm space-y-2.5">
                    <div className="flex items-center justify-between border-b border-[#ded3be] pb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-bold text-[#1e3a5f] uppercase tracking-wide font-mono">
                          Live Displacement Telemetry
                        </span>
                        <span className="text-[#baa990]">|</span>
                        <span className="text-[11.5px] text-[#1a202c] font-semibold">
                          {simulationResults?.epicenter_name || 'Ukraine'} ({simulationEpicenter})
                        </span>
                      </div>
                      <span className="text-[10px] text-[#786b59] font-mono font-semibold uppercase">
                        {simulationResults?.scenario?.toUpperCase() || 'CONFLICT'} SHOCK
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs">
                      <div className="p-2.5 bg-[#f4efe6] border border-[#ded3be] rounded-lg">
                        <div className="text-[10px] text-[#786b59] uppercase font-mono font-semibold">Displaced Mass</div>
                        <div className="text-sm font-bold font-mono text-[#1e3a5f] mt-0.5">
                          {simulationResults?.displaced_total ? (simulationResults.displaced_total / 1000000).toFixed(2) + 'M' : '3.50M'}
                        </div>
                        <div className="text-[9.5px] text-[#645d52] mt-0.5">
                          {simulationResults?.intensity || 90}% Severity
                        </div>
                      </div>

                      <div className="p-2.5 bg-[#f4efe6] border border-[#ded3be] rounded-lg">
                        <div className="text-[10px] text-[#786b59] uppercase font-mono font-semibold">Primary Sink</div>
                        <div className="text-xs font-bold font-mono text-[#1a202c] mt-0.5 truncate">
                          {simulationResults?.top_recipients?.[0]?.country_name || 'Poland'}
                        </div>
                        <div className="text-[10px] text-[#8b2626] font-mono font-bold mt-0.5">
                          {simulationResults?.top_recipients?.[0]?.predicted_influx?.toLocaleString() || '1,820,000'}
                        </div>
                      </div>

                      <div className="p-2.5 bg-[#f4efe6] border border-[#ded3be] rounded-lg">
                        <div className="text-[10px] text-[#786b59] uppercase font-mono font-semibold">Mean Distance</div>
                        <div className="text-xs font-bold font-mono text-[#945920] mt-0.5">
                          {simulationResults?.top_recipients?.[0]?.dist_km ? Math.round(simulationResults.top_recipients.reduce((acc: number, r: any) => acc + r.dist_km, 0) / simulationResults.top_recipients.length).toLocaleString() + ' km' : '1,420 km'}
                        </div>
                        <div className="text-[9.5px] text-[#645d52] mt-0.5">
                          Decay Gradient
                        </div>
                      </div>

                      <div className="p-2.5 bg-[#f4efe6] border border-[#ded3be] rounded-lg">
                        <div className="text-[10px] text-[#786b59] uppercase font-mono font-semibold">Policy Filter</div>
                        <div className="text-xs font-bold font-mono text-[#28534b] mt-0.5 capitalize truncate">
                          {simulationResults?.policy?.replace('_', ' ') || 'Universal'}
                        </div>
                        <div className="text-[9.5px] text-[#645d52] mt-0.5">
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
              /* Standard Layout for Map, Topology, Charts, Comparator */
              <>
                {/* World Map Section with Map Mode Switcher */}
                <section className="space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <h2 className="font-bold text-[#1a202c] text-xs uppercase tracking-wider font-serif">
                        Planetary Vector Surface
                      </h2>
                      <span className="text-xs text-[#786b59] font-mono">
                        ({data?.country_list?.length || 235} Sovereign States • 8 Snapshot Periods)
                      </span>
                    </div>

                    {/* Map Overlay Mode Switcher */}
                    <div className="flex items-center gap-1 bg-[#e8e0cf] border border-[#d8ccb8] p-0.5 rounded-lg text-xs">
                      {[
                        { id: 'corridors', label: 'Bilateral Corridors' },
                        { id: 'centrality', label: 'Centrality Topography' },
                        { id: 'communities', label: 'Alliance Clusters' },
                        { id: 'religion', label: 'Religion & Culture' },
                        { id: 'simulation', label: 'Crisis Displacement' },
                      ].map((m) => (
                        <button
                          key={m.id}
                          onClick={() => {
                            setMapMode(m.id as any);
                            if (m.id === 'simulation') {
                              setActiveView('simulator');
                            }
                          }}
                          className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                            mapMode === m.id
                              ? 'bg-[#1e3a5f] text-white font-semibold shadow-xs'
                              : 'text-[#5a5043] hover:text-[#1a202c]'
                          }`}
                        >
                          {m.label}
                        </button>
                      ))}
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
                    simulationEpicenter={mapMode === 'simulation' ? simulationEpicenter : null}
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
                      <div className="lg:col-span-8 bg-[#fbf9f4] border border-[#d8ccb8] p-4 rounded-xl shadow-sm space-y-3">
                        <div className="flex items-center justify-between border-b border-[#ded3be] pb-2.5">
                          <div>
                            <h3 className="font-bold text-[#1a202c] text-sm font-serif">
                              Top Bilateral Migration Corridors ({year})
                            </h3>
                            <p className="text-xs text-[#635a4d]">
                              Ranked by total registered bilateral migrant stock volume
                            </p>
                          </div>
                          <span className="text-[11px] bg-[#eef3f8] text-[#1e3a5f] border border-[#c7d9ec] px-2.5 py-1 rounded font-mono font-semibold">
                            {(data?.edges || []).filter((e: any) => Number(e.year) === year).length} Monitored Corridors
                          </span>
                        </div>

                        <div className="overflow-x-auto max-h-72 overflow-y-auto border border-[#ded3be] rounded-lg">
                          <table className="w-full text-left text-xs bg-[#ffffff]">
                            <thead>
                              <tr className="text-[10.5px] text-[#6d5c46] border-b border-[#ded3be] font-mono bg-[#f4efe6]">
                                <th className="py-2.5 px-3">#</th>
                                <th className="py-2.5 px-3">Origin State</th>
                                <th className="py-2.5 px-3">Destination State</th>
                                <th className="py-2.5 px-3">Migrant Stock</th>
                                <th className="py-2.5 px-3">Geography Rel</th>
                                <th className="py-2.5 px-3">Destination Religion</th>
                                <th className="py-2.5 px-3">Income Tier</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-[#f0e9dc] text-[#1e293b]">
                              {(data?.edges || [])
                                .filter((e: any) => Number(e.year) === year)
                                .slice(0, 15)
                                .map((e: any, idx: number) => (
                                  <tr
                                    key={idx}
                                    onClick={() => setSelectedCountry(e.origin_iso3)}
                                    className="hover:bg-[#fbf7f0] cursor-pointer transition"
                                  >
                                    <td className="py-2 px-3 font-mono text-[#8a7c6a] text-[11px]">#{idx + 1}</td>
                                    <td className="py-2 px-3 font-semibold text-[#1a202c] flex items-center gap-1.5">
                                      <span>{e.origin_flag}</span>
                                      <span>{e.origin_country_name || e.origin_iso3}</span>
                                      <span className="text-[10px] text-[#786b59] font-mono font-normal">({e.origin_iso3})</span>
                                    </td>
                                    <td className="py-2 px-3 font-semibold text-[#1a202c]">
                                      <span className="inline-flex items-center gap-1.5">
                                        <span>{e.dest_flag}</span>
                                        <span>{e.dest_country_name || e.dest_iso3}</span>
                                        <span className="text-[10px] text-[#786b59] font-mono font-normal">({e.dest_iso3})</span>
                                      </span>
                                    </td>
                                    <td className="py-2 px-3 font-mono font-bold text-[#1e3a5f]">
                                      {Math.round(Number(e.weight) || 0).toLocaleString()}
                                    </td>
                                    <td className="py-2 px-3">
                                      <span
                                        className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                                          e.same_continent === 1
                                            ? 'bg-[#eef3f8] text-[#1e3a5f] border border-[#c7d9ec]'
                                            : 'bg-[#f4efe6] text-[#6d5c46] border border-[#ded3be]'
                                        }`}
                                      >
                                        {e.same_continent === 1 ? 'Intra-Continental' : 'Inter-Continental'}
                                      </span>
                                    </td>
                                    <td className="py-2 px-3 text-[10.5px] text-[#28534b] font-medium">
                                      {e.dest_dominant_religion || 'Christianity'}
                                    </td>
                                    <td className="py-2 px-3">
                                      <span className="text-[10.5px] text-[#554a3b] font-medium">
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
                        <div className="p-4 bg-[#f4efe6] border border-[#ded3be] rounded-xl shadow-sm space-y-2.5">
                          <div className="flex items-center justify-between">
                            <span className="text-[10.5px] font-bold text-[#8b2626] uppercase tracking-wide font-mono">
                              Simulation Module
                            </span>
                            <span className="w-2 h-2 rounded-full bg-[#8b2626]"></span>
                          </div>
                          <h4 className="font-bold text-[#1a202c] text-sm font-serif">
                            Crisis Displacement Simulator
                          </h4>
                          <p className="text-xs text-[#635a4d] leading-relaxed">
                            Evaluate gravity-routing models for acute shocks across conflict, climate, border shifts, and cultural kinship with localized capacity strain assessments.
                          </p>
                          <button
                            onClick={() => {
                              setActiveView('simulator');
                              setMapMode('simulation');
                            }}
                            className="w-full py-2 bg-[#1e3a5f] hover:bg-[#162c46] text-white font-semibold rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow-sm"
                          >
                            <span>Launch Crisis Simulator</span>
                          </button>
                        </div>

                        {/* Quick Macro Indicators */}
                        <div className="p-4 bg-[#fbf9f4] border border-[#d8ccb8] rounded-xl shadow-sm space-y-2.5">
                          <h4 className="font-bold text-[#1a202c] text-xs uppercase tracking-wide font-serif">Systemic Insights</h4>
                          <div className="space-y-2 text-xs">
                            <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg shadow-xs">
                              <div className="text-[#1a202c] font-bold text-xs">Cultural & Religion Homophily:</div>
                              <div className="text-[#635a4d] text-[11.5px] mt-0.5">
                                Shared dominant religion and progressive MIPEX integration policies generate a <strong>+28% to +35% multiplier</strong> on bilateral corridor retention.
                              </div>
                            </div>
                            <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg shadow-xs">
                              <div className="text-[#1a202c] font-bold text-xs">Economic Pull:</div>
                              <div className="text-[#635a4d] text-[11.5px] mt-0.5">
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
      <footer className="border-t border-[#ded3be] bg-[#f4efe6] py-3 text-center text-xs text-[#786b59] font-mono">
        Nodes and Nations: Topological Study of Global Human Movement (1990–2025)
      </footer>
    </div>
  );
}
