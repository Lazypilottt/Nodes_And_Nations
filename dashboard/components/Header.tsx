import { motion } from 'framer-motion';
import React, { useEffect, useState } from 'react';
import { fmtCompact, fmtInt } from '../lib/theme';
import { type ThemeMode, useTheme } from '../lib/themeContext';
import { SNAPSHOT_YEARS, type View, type Year } from '../lib/types';
import { CountUp } from './ui';

export const TABS: { id: View; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'planner', label: 'Displacement planner' },
  { id: 'brief', label: 'Country brief' },
  { id: 'flows', label: 'Flows & gaps' },
  { id: 'drivers', label: 'Drivers' },
  { id: 'topology', label: 'Network' },
  { id: 'distributions', label: 'Distributions' },
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

const THEME_ICONS: Record<ThemeMode, React.ReactNode> = {
  light: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  ),
  dark: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
    </svg>
  ),
  system: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <rect x="3" y="4" width="18" height="12" rx="2" />
      <path d="M8 20h8M12 16v4" />
    </svg>
  ),
};

function ThemeSwitch() {
  const { mode, setMode } = useTheme();
  return (
    <div role="radiogroup" aria-label="Colour theme" className="flex rounded-full border border-zinc-200 p-0.5">
      {(['light', 'system', 'dark'] as ThemeMode[]).map((m) => (
        <button
          key={m}
          role="radio"
          aria-checked={mode === m}
          aria-label={`${m[0].toUpperCase()}${m.slice(1)} theme`}
          title={`${m[0].toUpperCase()}${m.slice(1)} theme`}
          onClick={() => setMode(m)}
          className={`relative flex h-6 w-6 items-center justify-center rounded-full transition-colors ${
            mode === m ? 'text-zinc-900' : 'text-zinc-400 hover:text-zinc-700'
          }`}
        >
          {mode === m && (
            <motion.span
              layoutId="theme-pill"
              className="absolute inset-0 rounded-full bg-zinc-100"
              transition={{ type: 'spring', stiffness: 500, damping: 35 }}
            />
          )}
          <span className="relative">{THEME_ICONS[m]}</span>
        </button>
      ))}
    </div>
  );
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
    <header className="no-print sticky top-0 z-30 border-b border-zinc-200 bg-white/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 pt-3 sm:px-6 lg:px-8">
        <div className="flex items-center gap-2.5">
          <motion.svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            aria-hidden
            className="text-blue-600 dark:text-blue-400"
            initial={{ rotate: -20, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            <motion.path
              d="M5 6 19 8 11 18 5 6"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeOpacity=".45"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.1, ease: 'easeInOut', delay: 0.2 }}
            />
            <circle cx="5" cy="6" r="2.5" fill="currentColor" />
            <circle cx="19" cy="8" r="2.5" fill="currentColor" />
            <circle cx="11" cy="18" r="2.5" fill="currentColor" />
          </motion.svg>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-zinc-900">Nodes &amp; Nations</div>
            <div className="text-[11px] text-zinc-500">Migration intelligence, 1990–2025</div>
          </div>
        </div>

        <div className="flex items-center gap-5 sm:gap-7">
          <dl className="hidden items-center gap-5 text-xs sm:flex sm:gap-7">
            <div>
              <dt className="text-zinc-500">Migrant stock</dt>
              <dd className="font-semibold tabular-nums text-zinc-900">
                <CountUp value={current?.total_migrant_stock} format={(n) => fmtCompact(n)} />
                {growth !== null && year !== SNAPSHOT_YEARS[0] && (
                  <span className="ml-1 font-normal text-emerald-600 dark:text-emerald-400">+{Math.round(growth * 100)}%</span>
                )}
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500">Corridors</dt>
              <dd className="font-semibold tabular-nums text-zinc-900">
                <CountUp value={current?.n_edges} format={(n) => fmtInt(n)} />
              </dd>
            </div>
            <div>
              <dt className="text-zinc-500">Modularity Q</dt>
              <dd className="font-semibold tabular-nums text-zinc-900">
                <CountUp value={q?.modularity_q} format={(n) => n.toFixed(3)} />
              </dd>
            </div>
          </dl>
          <ThemeSwitch />
        </div>
      </div>

      <div className="mx-auto flex max-w-7xl flex-wrap items-end justify-between gap-x-6 gap-y-2 px-4 sm:px-6 lg:px-8">
        <nav className="-mb-px flex max-w-full gap-5 overflow-x-auto" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onViewChange(t.id)}
              aria-current={view === t.id ? 'page' : undefined}
              className={`relative whitespace-nowrap pb-2.5 pt-3 text-sm transition-colors ${
                view === t.id ? 'font-medium text-zinc-900' : 'text-zinc-500 hover:text-zinc-800'
              }`}
            >
              {t.label}
              {view === t.id && (
                <motion.span
                  layoutId="tab-underline"
                  className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-blue-600 dark:bg-blue-400"
                  transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                />
              )}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-2 pb-2">
          <button
            onClick={togglePlay}
            aria-label={playing ? 'Pause timeline' : 'Play timeline'}
            className="relative flex h-7 w-7 items-center justify-center rounded-full border border-zinc-200 text-zinc-700 transition hover:bg-zinc-100"
          >
            {playing && <span className="absolute inset-0 animate-ping rounded-full border border-blue-500/40" />}
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
                className={`relative rounded-md px-2 py-0.5 text-xs tabular-nums transition-colors ${
                  y === year ? 'font-semibold text-zinc-900' : 'text-zinc-500 hover:text-zinc-800'
                }`}
              >
                {y === year && (
                  <motion.span
                    layoutId="year-pill"
                    className="absolute inset-0 rounded-md bg-white shadow-sm dark:bg-zinc-300/30"
                    transition={{ type: 'spring', stiffness: 500, damping: 38 }}
                  />
                )}
                <span className="relative">{y}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}
