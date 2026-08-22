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
    <header className="bg-[#f4efe6] border-b border-[#ded3be] relative shadow-sm">
      <div className="max-w-7xl mx-auto px-4 lg:px-8 py-3.5 space-y-3.5">
        {/* Top Branding & KPI Row */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[#1e3a5f] border border-[#14253e] flex items-center justify-center text-[#fbf9f4] font-serif font-bold text-base shadow-sm">
              NN
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-[#1a202c] text-base tracking-tight font-serif">
                  NODES & NATIONS
                </h1>
                <span className="text-[10px] bg-[#ece4d4] text-[#6d5c46] border border-[#d8ccb8] px-2 py-0.5 rounded font-mono font-semibold uppercase tracking-wider">
                  Global Topology Platform
                </span>
              </div>
              <p className="text-[11.5px] text-[#635a4d]">
                Topological Analysis and Crisis Displacement Simulator (1990–2025)
              </p>
            </div>
          </div>

          {/* Dynamic Key Metric Chips for Current Year */}
          <div className="flex items-center gap-2 text-xs">
            <div className="bg-[#ece5d6] border border-[#ded3be] px-3 py-1.5 rounded-lg shadow-sm">
              <div className="text-[10px] text-[#786b59] font-mono uppercase font-semibold">Global Migrant Stock ({currentYear})</div>
              <div className="font-bold text-[#1e293b] font-mono text-xs mt-0.5">
                {totalStockStr}{' '}
                <span className="text-[10px] text-[#28534b] font-semibold">
                  ({stockGrowthPct >= 0 ? `+${stockGrowthPct}%` : `${stockGrowthPct}%`})
                </span>
              </div>
            </div>

            <div className="bg-[#ece5d6] border border-[#ded3be] px-3 py-1.5 rounded-lg shadow-sm">
              <div className="text-[10px] text-[#786b59] font-mono uppercase font-semibold">Active Corridors</div>
              <div className="font-bold text-[#1e3a5f] font-mono text-xs mt-0.5">{activeCorridorsStr}</div>
            </div>

            <div className="bg-[#ece5d6] border border-[#ded3be] px-3 py-1.5 rounded-lg shadow-sm">
              <div className="text-[10px] text-[#786b59] font-mono uppercase font-semibold">Network Modularity Q</div>
              <div className="font-bold text-[#945920] font-mono text-xs mt-0.5">
                {modularityQ}{' '}
                <span className="text-[10px] text-[#786b59] font-normal">({modularityDesc})</span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs & Time-Travel Scrubber */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2.5 border-t border-[#ded3be]">
          {/* Main Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-[#e8e0cf] border border-[#d8ccb8] p-1 rounded-lg text-xs">
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
                      ? 'bg-[#1e3a5f] text-white font-semibold shadow-sm'
                      : 'text-[#5a5043] hover:text-[#1a202c] hover:bg-[#dfd5c2]'
                  }`}
                >
                  <span>{tab.label}</span>
                  {tab.badge && (
                    <span
                      className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                        isActive ? 'bg-[#2d5282] text-white' : 'bg-[#f4dede] text-[#8b2626] border border-[#e6b8b8]'
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
          <div className="flex items-center gap-2 bg-[#e8e0cf] border border-[#d8ccb8] px-3 py-1 rounded-lg text-xs">
            <span className="text-[10px] text-[#786b59] font-mono uppercase font-semibold mr-1">Period:</span>

            <button
              onClick={() => stepYear('prev')}
              className="px-1.5 py-0.5 hover:bg-[#dfd5c2] rounded text-[#5a5043] hover:text-[#1a202c] transition"
              title="Previous Snapshot"
            >
              ◀
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`px-2.5 py-1 rounded font-medium text-xs transition shadow-sm ${
                isPlaying
                  ? 'bg-[#945920] text-white'
                  : 'bg-[#f4efe6] text-[#4a3d2e] border border-[#d8ccb8] hover:bg-[#eae1cf]'
              }`}
            >
              {isPlaying ? 'Pause' : 'Play Timeline'}
            </button>

            <button
              onClick={() => stepYear('next')}
              className="px-1.5 py-0.5 hover:bg-[#dfd5c2] rounded text-[#5a5043] hover:text-[#1a202c] transition"
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
                      ? 'bg-[#1e3a5f] text-white font-bold shadow-sm'
                      : 'text-[#5a5043] hover:text-[#1a202c] hover:bg-[#dfd5c2]'
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
