import { MotionConfig } from 'framer-motion';
import type { AppProps } from 'next/app';
import '../lib/charts';
import { ThemeProvider } from '../lib/themeContext';
import '../styles/globals.css';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <ThemeProvider>
      {/* Respect the operating system's reduced-motion setting */}
      <MotionConfig reducedMotion="user">
        <Component {...pageProps} />
      </MotionConfig>
    </ThemeProvider>
  );
}
