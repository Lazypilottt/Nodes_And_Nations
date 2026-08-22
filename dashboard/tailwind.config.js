/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        cream: {
          50: '#fdfcf9',
          100: '#fbf9f4',
          200: '#f4efe6',
          300: '#eae2d3',
          400: '#ddd2be',
          500: '#cfc1a8',
          600: '#b8a589',
          700: '#948064',
          800: '#6d5c46',
          900: '#4a3d2e',
        },
        oxford: {
          50: '#f0f4f9',
          100: '#dce5f1',
          200: '#bccfe3',
          300: '#93b1d1',
          400: '#648ebd',
          500: '#4270a6',
          600: '#30588c',
          700: '#264670',
          800: '#223c5e',
          900: '#1e3a5f',
          950: '#14253e',
        },
        cognac: {
          50: '#fbf7f2',
          100: '#f6ede1',
          200: '#eed8c0',
          300: '#e1bd97',
          400: '#d29c6d',
          500: '#b87e45',
          600: '#9a6433',
          700: '#7e4d2a',
          800: '#673f27',
          900: '#563523',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ping-slow': 'ping 2.5s cubic-bezier(0, 0, 0.2, 1) infinite',
        'radar': 'radar 4s linear infinite',
      },
      keyframes: {
        radar: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        }
      }
    },
  },
  plugins: [],
}
