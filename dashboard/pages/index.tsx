import Head from 'next/head';
import React, { useMemo, useState } from 'react';
import useSWR from 'swr';
import CorridorComparator from '../components/CorridorComparator';
import CrisisSimulator from '../components/CrisisSimulator';
import Distributions from '../components/Distributions';
import Header from '../components/Header';
import NetworkView from '../components/NetworkView';
import Overview from '../components/Overview';
import { fetcher } from '../lib/fetcher';
import type { Dataset, NodeRow, View, Year } from '../lib/types';

export default function Home() {
  const { data, error, isLoading } = useSWR<Dataset>('/api/dataset', fetcher, { revalidateOnFocus: false });
  const [year, setYear] = useState<Year>(2025);
  const [view, setView] = useState<View>('overview');

  const yearNodes = useMemo(() => (data?.nodes ?? []).filter((n) => n.year === year), [data, year]);
  const nodesByIso = useMemo(() => {
    const map: Record<string, NodeRow> = {};
    for (const n of yearNodes) map[n.iso3] = n;
    return map;
  }, [yearNodes]);

  return (
    <div className="flex min-h-screen flex-col">
      <Head>
        <title>Nodes & Nations — Global Migration Network</title>
        <meta
          name="description"
          content="Network analysis and displacement simulation of global bilateral migration, 1990–2025."
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
          <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700">
            <div className="font-medium">Could not load the dataset</div>
            <div className="mt-1 text-red-600">{error.message}</div>
          </div>
        ) : isLoading || !data ? (
          <div className="space-y-6">
            <div className="h-[520px] animate-pulse rounded-xl bg-zinc-100" />
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
              <div className="h-64 animate-pulse rounded-xl bg-zinc-100 lg:col-span-2" />
              <div className="h-64 animate-pulse rounded-xl bg-zinc-100" />
            </div>
          </div>
        ) : (
          <>
            {view === 'overview' && <Overview data={data} year={year} nodesByIso={nodesByIso} />}
            {view === 'simulator' && <CrisisSimulator nodesByIso={nodesByIso} />}
            {view === 'distributions' && <Distributions data={data} year={year} />}
            {view === 'topology' && <NetworkView data={data} year={year} nodes={yearNodes} />}
            {view === 'comparator' && <CorridorComparator nodesByIso={nodesByIso} year={year} />}
          </>
        )}
      </main>

      <footer className="border-t border-zinc-200 py-6 text-xs text-zinc-500">
        <div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-4 sm:px-6 lg:px-8">
          <span>Bilateral migrant stock: UN DESA International Migrant Stock 2024. 2025 values are extrapolated.</span>
          <span>Religion and integration indices are modelled approximations — see README.</span>
        </div>
      </footer>
    </div>
  );
}
