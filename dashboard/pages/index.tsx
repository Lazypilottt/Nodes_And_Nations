import { AnimatePresence, motion } from 'framer-motion';
import Head from 'next/head';
import { useRouter } from 'next/router';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import useSWR from 'swr';
import CorridorComparator from '../components/CorridorComparator';
import CountryBrief from '../components/CountryBrief';
import DisplacementPlanner from '../components/DisplacementPlanner';
import Distributions from '../components/Distributions';
import Drivers from '../components/Drivers';
import Flows from '../components/Flows';
import Header from '../components/Header';
import NetworkView from '../components/NetworkView';
import Overview from '../components/Overview';
import { Skeleton } from '../components/ui';
import { COUNTRY_GEO_MAP } from '../lib/countryGeo';
import { fetcher } from '../lib/fetcher';
import { SNAPSHOT_YEARS, VIEWS, type Dataset, type NodeRow, type View, type Year } from '../lib/types';

export default function Home() {
  const router = useRouter();
  const { data, error, isLoading } = useSWR<Dataset>('/api/dataset', fetcher, { revalidateOnFocus: false });
  const [year, setYear] = useState<Year>(2025);
  const [view, setView] = useState<View>('overview');
  const [briefIso, setBriefIso] = useState('IND');
  const [hydrated, setHydrated] = useState(false);

  // Read view, year and country from the URL once, so links are shareable
  useEffect(() => {
    if (!router.isReady || hydrated) return;
    const q = router.query;
    if (typeof q.view === 'string' && (VIEWS as string[]).includes(q.view)) setView(q.view as View);
    const y = Number(q.year);
    if ((SNAPSHOT_YEARS as number[]).includes(y)) setYear(y as Year);
    if (typeof q.country === 'string' && COUNTRY_GEO_MAP[q.country.toUpperCase()]) setBriefIso(q.country.toUpperCase());
    setHydrated(true);
  }, [router.isReady, router.query, hydrated]);

  // ...and keep the URL in sync afterwards
  useEffect(() => {
    if (!hydrated) return;
    const query: Record<string, string> = { view, year: String(year) };
    if (view === 'brief') query.country = briefIso;
    router.replace({ pathname: '/', query }, undefined, { shallow: true, scroll: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, year, briefIso, hydrated]);

  const openBrief = useCallback((iso3: string) => {
    setBriefIso(iso3);
    setView('brief');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const yearNodes = useMemo(() => (data?.nodes ?? []).filter((n) => n.year === year), [data, year]);
  const nodesByIso = useMemo(() => {
    const map: Record<string, NodeRow> = {};
    for (const n of yearNodes) map[n.iso3] = n;
    return map;
  }, [yearNodes]);

  return (
    <div className="flex min-h-screen flex-col">
      <Head>
        <title>Nodes & Nations — Migration intelligence</title>
        <meta
          name="description"
          content="Forecast where people displaced by a crisis will go, find gaps in migration corridors, and explore the global migration network, 1990–2025."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <Header
        year={year}
        onYearChange={setYear}
        view={view}
        onViewChange={setView}
        networkSummary={data?.network_summary ?? []}
        modularityScores={data?.modularity_scores ?? []}
      />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8">
        {error ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-6 text-sm text-red-700 dark:text-red-300">
            <div className="font-medium">Could not load the dataset</div>
            <div className="mt-1 opacity-80">{error.message}</div>
          </div>
        ) : isLoading || !data ? (
          <div className="space-y-6">
            <Skeleton className="h-[520px]" />
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <Skeleton className="h-64 lg:col-span-2" />
              <Skeleton className="h-64" />
            </div>
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={view}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            >
              {view === 'overview' && <Overview data={data} year={year} nodesByIso={nodesByIso} onOpenBrief={openBrief} />}
              {view === 'planner' && <DisplacementPlanner data={data} nodesByIso={nodesByIso} />}
              {view === 'brief' && <CountryBrief data={data} year={year} iso3={briefIso} onChange={setBriefIso} />}
              {view === 'flows' && <Flows data={data} year={year} onOpenBrief={openBrief} />}
              {view === 'drivers' && <Drivers data={data} year={year} />}
              {view === 'topology' && <NetworkView data={data} year={year} nodes={yearNodes} />}
              {view === 'distributions' && <Distributions data={data} year={year} />}
              {view === 'comparator' && <CorridorComparator nodesByIso={nodesByIso} year={year} />}
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      <footer className="no-print border-t border-zinc-200 py-6 text-xs text-zinc-500">
        <div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-4 sm:px-6 lg:px-8">
          <span>
            Migrant stock: UN DESA International Migrant Stock 2024 (2025 extrapolated). Displacement: UNHCR. Geography: CEPII GeoDist.
          </span>
          <span>Religion and integration indices are modelled approximations — see README.</span>
        </div>
      </footer>
    </div>
  );
}
