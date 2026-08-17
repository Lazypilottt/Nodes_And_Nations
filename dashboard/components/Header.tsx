import React, { useState, useEffect } from 'react';

interface HeaderProps {
  currentYear: number;
  onYearChange: (year: number) => void;
  activeView: 'map' | 'simulator' | 'charts' | 'topology' | 'comparator';
  onViewChange: (view: 'map' | 'simulator' | 'charts' | 'topology' | 'comparator') => void;
  summaryStats?: any[];
  networkSummary?: any[];
  modularityScores?: any[];
}

const SNAPSHOT_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025];

export default function Header({
  currentYear,
  onYearChange,
  activeView,
  onViewChange,
  summaryStats = [],
  networkSummary = [],
  modularityScores = [],
}: HeaderProps) {
  const [isPlaying, setIsPlaying] = useState(false);

  // Dynamic Pipeline Metrics for Current Year
  const currentSummary = React.useMemo(() => {
    return (networkSummary || []).find((s: any) => Number(s.year) === currentYear);
  }, [networkSummary, currentYear]);

  const summary1990 = React.useMemo(() => {
    return (networkSummary || []).find((s: any) => Number(s.year) === 1990);
  }, [networkSummary]);

  const currentModularity = React.useMemo(() => {
    return (modularityScores || []).find(
      (m: any) => Number(m.year) === currentYear && m.algorithm === 'louvain'
    );
  }, [modularityScores, currentYear]);

  const totalStockStr = currentSummary?.total_migrant_stock
    ? (Number(currentSummary.total_migrant_stock) / 1000000).toFixed(1) + 'M'
    : '245.0M';

  const stockGrowthPct =
    currentSummary?.total_migrant_stock && summary1990?.total_migrant_stock
      ? Math.round(
          ((Number(currentSummary.total_migrant_stock) - Number(summary1990.total_migrant_stock)) /
            Number(summary1990.total_migrant_stock)) *
            100
        )
      : 0;

  const activeCorridorsStr = currentSummary?.n_edges
    ? Number(currentSummary.n_edges).toLocaleString() + ' Pairs'
    : '8,548 Pairs';

  const modularityQ = currentModularity?.modularity_q
    ? Number(currentModularity.modularity_q).toFixed(3)
    : '0.589';

  const modularityDesc =
    Number(modularityQ) >= 0.58
      ? 'Segmented'
      : Number(modularityQ) >= 0.54
      ? 'Transition'
      : 'Integrated';

  // Time-Machine Auto-Player Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        onYearChange(
          SNAPSHOT_YEARS[
            (SNAPSHOT_YEARS.indexOf(currentYear) + 1) % SNAPSHOT_YEARS.length
          ]
        );
      }, 2600);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentYear, onYearChange]);

  const stepYear = (direction: 'next' | 'prev') => {
    const idx = SNAPSHOT_YEARS.indexOf(currentYear);
    if (direction === 'next') {
      onYearChange(SNAPSHOT_YEARS[(idx + 1) % SNAPSHOT_YEARS.length]);
    } else {
      onYearChange(SNAPSHOT_YEARS[(idx - 1 + SNAPSHOT_YEARS.length) % SNAPSHOT_YEARS.length]);
    }
  };

  return (
    <header className="bg-[#080c14] border-b border-slate-800 sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-4 lg:px-8 py-3 space-y-3">
        {/* Top Branding & KPI Row */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 font-bold text-sm">
              NN
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-slate-100 text-base tracking-tight">
                  NODES & NATIONS
                </h1>
                <span className="text-[10px] bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded font-mono font-medium uppercase">
                  Global Topology Platform
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                Topological Analysis and Crisis Displacement Simulator (1990–2025)
              </p>
            </div>
          </div>

          {/* Dynamic Key Metric Chips for Current Year */}
          <div className="flex items-center gap-2 text-xs">
            <div className="bg-[#0f1624] border border-slate-800 px-3 py-1.5 rounded-lg">
              <div className="text-[10px] text-slate-500 font-mono uppercase">Global Migrant Stock ({currentYear})</div>
              <div className="font-semibold text-slate-200 font-mono text-xs">
                {totalStockStr}{' '}
                <span className="text-[10px] text-emerald-400 font-normal">
                  ({stockGrowthPct >= 0 ? `+${stockGrowthPct}%` : `${stockGrowthPct}%`})
                </span>
              </div>
            </div>

            <div className="bg-[#0f1624] border border-slate-800 px-3 py-1.5 rounded-lg">
              <div className="text-[10px] text-slate-500 font-mono uppercase">Active Corridors</div>
              <div className="font-semibold text-blue-400 font-mono text-xs">{activeCorridorsStr}</div>
            </div>

            <div className="bg-[#0f1624] border border-slate-800 px-3 py-1.5 rounded-lg">
              <div className="text-[10px] text-slate-500 font-mono uppercase">Network Modularity Q</div>
              <div className="font-semibold text-amber-400 font-mono text-xs">
                {modularityQ}{' '}
                <span className="text-[10px] text-slate-400 font-normal">({modularityDesc})</span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs & Time-Travel Scrubber */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800/80">
          {/* Main Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-[#0b101b] border border-slate-800 p-1 rounded-lg text-xs">
            {[
              { id: 'map', label: 'Overview & Corridors' },
              { id: 'simulator', label: 'Crisis Simulator', badge: 'Active' },
              { id: 'charts', label: 'Distribution Analytics' },
              { id: 'topology', label: 'Centralities & Topology' },
              { id: 'comparator', label: 'Bilateral Comparator' },
            ].map((tab) => {
              const isActive = activeView === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => onViewChange(tab.id as any)}
                  className={`px-3 py-1.5 rounded-md font-medium text-xs transition flex items-center gap-1.5 ${
                    isActive
                      ? 'bg-blue-600 text-white font-semibold shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <span>{tab.label}</span>
                  {tab.badge && (
                    <span
                      className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                        isActive ? 'bg-blue-800 text-blue-100' : 'bg-red-950 text-red-400 border border-red-800/60'
                      }`}
                    >
                      {tab.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Timeline Player (1990 - 2025) */}
          <div className="flex items-center gap-2 bg-[#0b101b] border border-slate-800 px-3 py-1 rounded-lg text-xs">
            <span className="text-[10px] text-slate-500 font-mono uppercase mr-1">Period:</span>

            <button
              onClick={() => stepYear('prev')}
              className="px-1.5 py-0.5 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 transition"
              title="Previous Snapshot"
            >
              ◀
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`px-2.5 py-1 rounded font-medium text-xs transition ${
                isPlaying
                  ? 'bg-amber-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {isPlaying ? 'Pause' : 'Play Timeline'}
            </button>

            <button
              onClick={() => stepYear('next')}
              className="px-1.5 py-0.5 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 transition"
              title="Next Snapshot"
            >
              ▶
            </button>

            {/* Year Selector Buttons */}
            <div className="flex gap-1 ml-1.5">
              {SNAPSHOT_YEARS.map((yr) => (
                <button
                  key={yr}
                  onClick={() => {
                    setIsPlaying(false);
                    onYearChange(yr);
                  }}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono transition ${
                    currentYear === yr
                      ? 'bg-slate-200 text-slate-950 font-bold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  {yr}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
