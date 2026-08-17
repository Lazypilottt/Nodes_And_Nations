import React, { useState } from 'react';
import { COUNTRY_GEO_MAP, CountryGeo, getCountryGeo } from '../lib/countryGeo';

interface CorridorComparatorProps {
  nodes: any[];
  edges: any[];
  year: number;
}

export default function CorridorComparator({ nodes, edges, year }: CorridorComparatorProps) {
  const [countryA, setCountryA] = useState('IND');
  const [countryB, setCountryB] = useState('USA');

  const geoA = getCountryGeo(countryA);
  const geoB = getCountryGeo(countryB);

  const nodeA = (nodes || []).find((n) => n.iso3 === countryA && Number(n.year) === year) || {};
  const nodeB = (nodes || []).find((n) => n.iso3 === countryB && Number(n.year) === year) || {};

  // Historical flows A -> B and B -> A
  const flowAtoB = (edges || []).find(
    (e) => e.origin_iso3 === countryA && e.dest_iso3 === countryB && Number(e.year) === year
  );
  const flowBtoA = (edges || []).find(
    (e) => e.origin_iso3 === countryB && e.dest_iso3 === countryA && Number(e.year) === year
  );

  const stockAtoB = Number(flowAtoB?.weight || 0);
  const stockBtoA = Number(flowBtoA?.weight || 0);
  const netFlow = stockAtoB - stockBtoA;

  return (
    <div className="bg-[#0b101b] border border-slate-800 p-4 rounded-xl shadow-md space-y-4">
      {/* Header & Country Selectors */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
            <span>Bilateral Corridor Comparator & Dyadic Analysis</span>
            <span className="text-[10px] bg-slate-800 text-slate-400 font-mono px-2 py-0.5 rounded">
              Year {year}
            </span>
          </h3>
          <p className="text-xs text-slate-400">
            Compare socioeconomic push/pull determinants and bilateral migration trajectories side-by-side
          </p>
        </div>

        {/* Country Selectors */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400">State A:</span>
            <select
              value={countryA}
              onChange={(e) => setCountryA(e.target.value)}
              className="bg-[#080c14] border border-slate-700 text-slate-200 px-2.5 py-1.5 rounded-lg text-xs focus:outline-none focus:border-blue-500"
            >
              {Object.values(COUNTRY_GEO_MAP).map((c) => (
                <option key={c.iso3} value={c.iso3}>
                  {c.flag} {c.name} ({c.iso3})
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => {
              const temp = countryA;
              setCountryA(countryB);
              setCountryB(temp);
            }}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs transition"
            title="Swap Origin and Destination"
          >
            Swap
          </button>

          <div className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400">State B:</span>
            <select
              value={countryB}
              onChange={(e) => setCountryB(e.target.value)}
              className="bg-[#080c14] border border-slate-700 text-slate-200 px-2.5 py-1.5 rounded-lg text-xs focus:outline-none focus:border-blue-500"
            >
              {Object.values(COUNTRY_GEO_MAP).map((c) => (
                <option key={c.iso3} value={c.iso3}>
                  {c.flag} {c.name} ({c.iso3})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Side-by-Side Socioeconomic Factor Grid */}
      <div className="grid grid-cols-1 md:grid-cols-11 gap-3 items-center">
        {/* Country A Card (5 cols) */}
        <div className="md:col-span-5 bg-[#080c14] border border-slate-800/80 p-3.5 rounded-lg space-y-2.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
            <div className="flex items-center gap-2 font-medium text-slate-100">
              <span className="text-lg">{geoA.flag}</span>
              <div>
                <div className="text-xs font-semibold">{geoA.name}</div>
                <div className="text-[10px] text-slate-500 font-mono">{geoA.iso3} • {geoA.un_region}</div>
              </div>
            </div>
            <span className="text-[10px] bg-blue-950 text-blue-300 border border-blue-800 px-2 py-0.5 rounded font-mono">
              {nodeA.income_group || geoA.income_group}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">GDP per Capita</div>
              <div className="font-semibold text-slate-200 font-mono text-xs">
                ${Math.round(Number(nodeA.gdp_per_capita) || 0).toLocaleString()}
              </div>
            </div>
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">Population</div>
              <div className="font-semibold text-slate-200 font-mono text-xs">
                {(Number(nodeA.population) || 0) >= 1000000
                  ? `${((Number(nodeA.population) || 0) / 1000000).toFixed(1)}M`
                  : `${Math.round((Number(nodeA.population) || 0) / 1000)}k`}
              </div>
            </div>
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">Visa Openness Score</div>
              <div className="font-semibold text-amber-400 font-mono text-xs">
                {(Number(nodeA.visa_openness_index) || 0).toFixed(1)}
              </div>
            </div>
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">Climate Vulnerability</div>
              <div className="font-semibold text-slate-300 font-mono text-xs">
                {(Number(nodeA.climate_vulnerability) || 0).toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        {/* Central Vector (1 col) */}
        <div className="md:col-span-1 flex flex-col items-center justify-center space-y-1 text-center py-1">
          <div className="text-[11px] font-mono font-semibold text-blue-400">VS</div>
          <div className="w-px h-6 bg-slate-800"></div>
          <div className="text-[9.5px] text-slate-500 font-mono uppercase">Bilateral</div>
        </div>

        {/* Country B Card (5 cols) */}
        <div className="md:col-span-5 bg-[#080c14] border border-slate-800/80 p-3.5 rounded-lg space-y-2.5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
            <div className="flex items-center gap-2 font-medium text-slate-100">
              <span className="text-lg">{geoB.flag}</span>
              <div>
                <div className="text-xs font-semibold">{geoB.name}</div>
                <div className="text-[10px] text-slate-500 font-mono">{geoB.iso3} • {geoB.un_region}</div>
              </div>
            </div>
            <span className="text-[10px] bg-amber-950 text-amber-300 border border-amber-800 px-2 py-0.5 rounded font-mono">
              {nodeB.income_group || geoB.income_group}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">GDP per Capita</div>
              <div className="font-semibold text-slate-200 font-mono text-xs">
                ${Math.round(Number(nodeB.gdp_per_capita) || 0).toLocaleString()}
              </div>
            </div>
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">Population</div>
              <div className="font-semibold text-slate-200 font-mono text-xs">
                {(Number(nodeB.population) || 0) >= 1000000
                  ? `${((Number(nodeB.population) || 0) / 1000000).toFixed(1)}M`
                  : `${Math.round((Number(nodeB.population) || 0) / 1000)}k`}
              </div>
            </div>
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">Visa Openness Score</div>
              <div className="font-semibold text-amber-400 font-mono text-xs">
                {(Number(nodeB.visa_openness_index) || 0).toFixed(1)}
              </div>
            </div>
            <div className="p-2 bg-[#0b101b] rounded">
              <div className="text-[10px] text-slate-500">Climate Vulnerability</div>
              <div className="font-semibold text-slate-300 font-mono text-xs">
                {(Number(nodeB.climate_vulnerability) || 0).toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bilateral Net Balance */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 border-t border-slate-800">
        <div className="p-3 bg-[#080c14] border border-slate-800 rounded-lg space-y-0.5">
          <div className="text-slate-500 text-[10px] uppercase font-mono">
            {geoA.name} → {geoB.name} Stock
          </div>
          <div className="text-lg font-semibold text-blue-400 font-mono">
            {stockAtoB.toLocaleString()}
          </div>
          <div className="text-[10.5px] text-slate-400">
            {stockAtoB > 0 ? 'Active bilateral stock' : 'No recorded stock'}
          </div>
        </div>

        <div className="p-3 bg-[#080c14] border border-slate-800 rounded-lg space-y-0.5">
          <div className="text-slate-500 text-[10px] uppercase font-mono">
            {geoB.name} → {geoA.name} Stock
          </div>
          <div className="text-lg font-semibold text-amber-400 font-mono">
            {stockBtoA.toLocaleString()}
          </div>
          <div className="text-[10.5px] text-slate-400">Reverse migration stock</div>
        </div>

        <div className="p-3 bg-[#080c14] border border-slate-800 rounded-lg space-y-0.5">
          <div className="text-slate-500 text-[10px] uppercase font-mono">Net Asymmetry Balance</div>
          <div className="text-lg font-semibold text-emerald-400 font-mono">
            {netFlow >= 0 ? `+${netFlow.toLocaleString()}` : netFlow.toLocaleString()}
          </div>
          <div className="text-[10.5px] text-slate-400">
            Directional gradient towards {netFlow >= 0 ? geoB.name : geoA.name}
          </div>
        </div>
      </div>
    </div>
  );
}
