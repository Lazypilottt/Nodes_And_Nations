/** @type {import('tailwindcss').Config} */

// Neutral colours resolve to CSS variables (styles/globals.css) so the whole UI
// switches theme by toggling the `dark` class on <html>, without a `dark:`
// variant on every element.
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;
const zinc = Object.fromEntries(
  [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950].map((k) => [k, v(`zinc-${k}`)])
);

module.exports = {
  darkMode: 'class',
  content: ['./pages/**/*.{js,ts,jsx,tsx}', './components/**/*.{js,ts,jsx,tsx}', './lib/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        white: v('white'),
        zinc,
      },
      keyframes: {
        'flow-dash': { to: { strokeDashoffset: '-24' } },
        'pulse-ring': {
          '0%': { transform: 'scale(0.6)', opacity: '0.6' },
          '100%': { transform: 'scale(2.4)', opacity: '0' },
        },
        shimmer: { '100%': { transform: 'translateX(100%)' } },
      },
      animation: {
        'flow-dash': 'flow-dash 1.2s linear infinite',
        'pulse-ring': 'pulse-ring 2s cubic-bezier(0.2, 0.6, 0.4, 1) infinite',
        shimmer: 'shimmer 1.6s infinite',
      },
    },
  },
  plugins: [],
};
