import {
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

ChartJS.register(CategoryScale, LinearScale, LogarithmicScale, PointElement, LineElement, Filler, Tooltip, Legend);

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

export const lineGrid = { color: '#f4f4f5' };
