import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { applyChartTheme } from './charts';

export type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeState {
  mode: ThemeMode;
  dark: boolean;
  setMode: (m: ThemeMode) => void;
}

const STORAGE_KEY = 'nn-theme';
const ThemeContext = createContext<ThemeState>({ mode: 'system', dark: false, setMode: () => {} });

function readStored(): ThemeMode {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    return v === 'light' || v === 'dark' ? v : 'system';
  } catch {
    return 'system';
  }
}

/**
 * Light / dark / system theme. The class on <html> is set before first paint by
 * the inline script in pages/_document.tsx; this provider keeps it in sync,
 * remembers the choice, and retunes Chart.js defaults.
 */
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>('system');
  const [systemDark, setSystemDark] = useState(false);

  useEffect(() => {
    setModeState(readStored());
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    setSystemDark(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const dark = mode === 'dark' || (mode === 'system' && systemDark);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    applyChartTheme(dark);
  }, [dark]);

  const setMode = useCallback((m: ThemeMode) => {
    setModeState(m);
    try {
      if (m === 'system') window.localStorage.removeItem(STORAGE_KEY);
      else window.localStorage.setItem(STORAGE_KEY, m);
    } catch {
      /* storage unavailable: the choice lasts for this visit only */
    }
  }, []);

  const value = useMemo(() => ({ mode, dark, setMode }), [mode, dark, setMode]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

/** Colours for SVG and canvas drawing that cannot use CSS variables directly. */
export function usePalette() {
  const { dark } = useTheme();
  return useMemo(
    () =>
      dark
        ? { land: '#3a3a42', landStroke: '#18181c', ink: '#f4f4f6', muted: '#9c9ca6', grid: '#2e2e35', surface: '#18181c', accent: '#60a5fa' }
        : { land: '#e4e4e7', landStroke: '#ffffff', ink: '#18181b', muted: '#71717a', grid: '#f4f4f5', surface: '#ffffff', accent: '#2563eb' },
    [dark]
  );
}

/** Key for chart components so they redraw with the new defaults on a theme change. */
export function useChartKey() {
  return useTheme().dark ? 'dark' : 'light';
}
