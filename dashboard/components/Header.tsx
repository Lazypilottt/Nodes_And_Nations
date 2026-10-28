import React, { useEffect, useState } from 'react';
import { fmtCompact, fmtInt, fmtNum } from '../lib/theme';
import { SNAPSHOT_YEARS, type View, type Year } from '../lib/types';

const TABS: { id: View; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'simulator', label: 'Crisis simulator' },
  { id: 'distributions', label: 'Distributions' },
  { id: 'topology', label: 'Network' },
  { id: 'comparator', label: 'Compare' },
];

interface HeaderProps {
  year: Year;
  onYearChange: (y: Year) => void;
  view: View;
  onViewChange: (v: View) => void;
  networkSummary: Record<string, any>[];
  modularityScores: Record<string, any>[];
}

export default function Header({ year, onYearChange, view, onViewChange, networkSummary, modularityScores }: HeaderProps) {
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = setTimeout(() => {
      const i = SNAPSHOT_YEARS.indexOf(year);
      if (i === SNAPSHOT_YEARS.length - 1) setPlaying(false);
      else onYearChange(SNAPSHOT_YEARS[i + 1]);
    }, 1600);
    return () => clearTimeout(id);
  }, [playing, year, onYearChange]);

  const togglePlay = () => {
    if (!playing && year === SNAPSHOT_YEARS[SNAPSHOT_YEARS.length - 1]) onYearChange(SNAPSHOT_YEARS[0]);
    setPlaying((p) => !p);
  };

  const current = networkSummary.find((s) => Number(s.year) === year);
  const base = networkSummary.find((s) => Number(s.year) === SNAPSHOT_YEARS[0]);
  const q = modularityScores.find((m) => Number(m.year) === year && m.algorithm === 'louvain');
  const growth =
    current && base && base.total_migrant_stock
      ? (current.total_migrant_stock - base.total_migrant_stock) / base.total_migrant_stock
      : null;

  return (
    <header className="sticky top-0 z-30 border-b border-zinc-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 pt-3 sm:px-6 lg:px-8">
        <div className="flex items-center gap-2.5">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden className="text-blue-600">
            <circle cx="5" cy="6" r="2.5" fill="currentColor" />
            <circle cx="19" cy="8" r="2.5" fill="currentColor" />
            <circle cx="11" cy="18" r="2.5" fill="currentColor" />
            <path d="M5 6 19 8 11 18 5 6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeOpacity=".4" />
          </svg>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-zinc-900">Nodes &amp; Nations</div>
            <div className="text-[11px] text-zinc-500">Global migration network, 1990–2025</div>
          </div>
        </div>

        <dl className="flex items-center gap-5 text-xs sm:gap-7">
          <div>
            <dt className="text-zinc-500">Migrant stock</dt>
            <dd className="font-semibold tabular-nums text-zinc-900">
              {fmtCompact(current?.total_migrant_stock)}
              {growth !== null && year !== SNAPSHOT_YEARS[0] && (
                <span className="ml-1 font-normal text-emerald-600">+{Math.round(growth * 100)}%</span>
              )}
            </dd>
          </div>
          <div>
            <dt className="text-zinc-500">Corridors</dt>
            <dd className="font-semibold tabular-nums text-zinc-900">{fmtInt(current?.n_edges)}</dd>
          </div>
          <div>
            <dt className="text-zinc-500">Modularity Q</dt>
            <dd className="font-semibold tabular-nums text-zinc-900">{fmtNum(q?.modularity_q, 3)}</dd>
          </div>
        </dl>
      </div>

      <div className="mx-auto flex max-w-7xl flex-wrap items-end justify-between gap-x-6 gap-y-2 px-4 sm:px-6 lg:px-8">
        <nav className="-mb-px flex gap-5 overflow-x-auto" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onViewChange(t.id)}
              aria-current={view === t.id ? 'page' : undefined}
              className={`whitespace-nowrap border-b-2 pb-2.5 pt-3 text-sm transition-colors ${
                view === t.id
                  ? 'border-blue-600 font-medium text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2 pb-2">
          <button
            onClick={togglePlay}
            aria-label={playing ? 'Pause timeline' : 'Play timeline'}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-zinc-200 text-zinc-700 hover:bg-zinc-50"
          >
            {playing ? (
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><rect x="1" y="1" width="3" height="8" fill="currentColor" /><rect x="6" y="1" width="3" height="8" fill="currentColor" /></svg>
            ) : (
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M2 1 9 5 2 9Z" fill="currentColor" /></svg>
            )}
          </button>
          <div className="flex overflow-x-auto rounded-lg bg-zinc-100 p-0.5" role="tablist" aria-label="Snapshot year">
            {SNAPSHOT_YEARS.map((y) => (
              <button
                key={y}
                role="tab"
                aria-selected={y === year}
                onClick={() => {
                  setPlaying(false);
                  onYearChange(y);
                }}
                className={`rounded-md px-2 py-0.5 text-xs tabular-nums transition-colors ${
                  y === year ? 'bg-white font-semibold text-zinc-900 shadow-sm' : 'text-zinc-500 hover:text-zinc-800'
                }`}
              >
                {y}
              </button>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}
