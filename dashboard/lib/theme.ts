// Single source of truth for colours used by charts, the map and the UI chrome.

export const INK = '#18181b';
export const MUTED = '#71717a';
export const BORDER = '#e4e4e7';
export const ACCENT = '#2563eb';
export const LAND = '#e4e4e7';
export const LAND_HOVER = '#d4d4d8';

// Categorical palette: muted, distinguishable, colour-blind friendly ordering.
export const CATEGORICAL = [
  '#2563eb', // blue
  '#0d9488', // teal
  '#d97706', // amber
  '#db2777', // pink
  '#7c3aed', // violet
  '#65a30d', // lime
  '#0891b2', // cyan
  '#dc2626', // red
  '#64748b', // slate
  '#a16207', // ochre
];

export const RELIGION_COLORS: Record<string, string> = {
  Christianity: '#2563eb',
  Islam: '#0d9488',
  Hinduism: '#d97706',
  Buddhism: '#db2777',
  Judaism: '#7c3aed',
  Unaffiliated: '#64748b',
};

export const INCOME_ORDER = ['High income', 'Upper middle income', 'Lower middle income', 'Low income'];
export const INCOME_COLORS: Record<string, string> = {
  'High income': '#1e40af',
  'Upper middle income': '#3b82f6',
  'Lower middle income': '#93c5fd',
  'Low income': '#cbd5e1',
};

export type StrainLevel = 'Critical' | 'High' | 'Moderate' | 'Low';
export const STRAIN_COLORS: Record<StrainLevel, string> = {
  Critical: '#dc2626',
  High: '#ea580c',
  Moderate: '#d97706',
  Low: '#16a34a',
};

export const MIPEX_TIERS = ['High (≥ 0.70)', 'Moderate (0.45–0.70)', 'Restrictive (< 0.45)'];
export const MIPEX_COLORS = ['#0d9488', '#93c5fd', '#cbd5e1'];

export function mipexTier(v: number | null | undefined): string | null {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  if (v >= 0.7) return MIPEX_TIERS[0];
  if (v >= 0.45) return MIPEX_TIERS[1];
  return MIPEX_TIERS[2];
}

export function categorical(i: number): string {
  return CATEGORICAL[((i % CATEGORICAL.length) + CATEGORICAL.length) % CATEGORICAL.length];
}

/** Blue sequential ramp, t in [0, 1]. */
export function sequential(t: number): string {
  const stops = [
    [239, 246, 255],
    [191, 219, 254],
    [96, 165, 250],
    [37, 99, 235],
    [30, 58, 138],
  ];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const c = stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function fmtCompact(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${(n / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(digits)}k`;
  return `${Math.round(n)}`;
}

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

export function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

export function fmtPct(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(digits)}%`;
}
