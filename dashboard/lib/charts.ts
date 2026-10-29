import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  LogarithmicScale,
  PointElement,
  Tooltip,
} from 'chart.js';
import { BORDER, INK, MUTED } from './theme';

ChartJS.register(
  ArcElement, BarElement, CategoryScale, LinearScale, LogarithmicScale, PointElement, LineElement, Filler, Tooltip, Legend,
);

ChartJS.defaults.font.family = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, Roboto, sans-serif';
ChartJS.defaults.font.size = 11;
ChartJS.defaults.color = MUTED;
ChartJS.defaults.borderColor = BORDER;
ChartJS.defaults.plugins.legend.labels.boxWidth = 8;
ChartJS.defaults.plugins.legend.labels.boxHeight = 8;
ChartJS.defaults.plugins.legend.labels.usePointStyle = true;
ChartJS.defaults.plugins.tooltip.backgroundColor = INK;
ChartJS.defaults.plugins.tooltip.padding = 8;
ChartJS.defaults.plugins.tooltip.cornerRadius = 6;
ChartJS.defaults.plugins.tooltip.displayColors = true;
ChartJS.defaults.plugins.tooltip.boxPadding = 4;

let gridColor = '#f4f4f5';

/** Grid lines read the current theme each time a chart draws. */
export const lineGrid = { color: () => gridColor };

/** Retune Chart.js defaults for light or dark mode (called by ThemeProvider). */
export function applyChartTheme(dark: boolean) {
  gridColor = dark ? '#2e2e35' : '#f4f4f5';
  ChartJS.defaults.color = dark ? '#9c9ca6' : MUTED;
  ChartJS.defaults.borderColor = dark ? '#2e2e35' : BORDER;
  ChartJS.defaults.plugins.tooltip.backgroundColor = dark ? '#f4f4f6' : INK;
  ChartJS.defaults.plugins.tooltip.titleColor = dark ? '#18181b' : '#ffffff';
  ChartJS.defaults.plugins.tooltip.bodyColor = dark ? '#18181b' : '#ffffff';
  // Mounted charts pick these up because chart components are keyed by theme
  // (see useChartKey in lib/themeContext.tsx).
}

// Smooth, short default animation for every chart
ChartJS.defaults.animation = { duration: 700, easing: 'easeOutQuart' } as any;
