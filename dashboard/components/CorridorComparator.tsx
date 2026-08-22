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
    <div className="bg-[#fbf9f4] border border-[#d8ccb8] p-4 rounded-xl shadow-sm space-y-4">
      {/* Header & Country Selectors */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#ded3be] pb-3">
        <div>
          <h3 className="font-bold text-[#1a202c] text-sm flex items-center gap-2 font-serif">
            <span>Bilateral Corridor Comparator & Dyadic Analysis</span>
            <span className="text-[10px] bg-[#ece4d4] text-[#6d5c46] border border-[#d8ccb8] font-mono px-2 py-0.5 rounded font-semibold">
              Year {year}
            </span>
          </h3>
          <p className="text-xs text-[#635a4d]">
            Compare socioeconomic push/pull determinants and bilateral migration trajectories side-by-side
          </p>
        </div>

        {/* Country Selectors */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-[#645d52] font-medium">State A:</span>
            <select
              value={countryA}
              onChange={(e) => setCountryA(e.target.value)}
              className="bg-[#ffffff] border border-[#ded3be] text-[#1e293b] px-2.5 py-1.5 rounded-lg text-xs focus:outline-none focus:border-[#1e3a5f]"
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
            className="px-2.5 py-1.5 bg-[#1e3a5f] hover:bg-[#162c46] text-white rounded-lg text-xs font-semibold transition shadow-xs"
            title="Swap Origin and Destination"
          >
            ⇄ Swap
          </button>

          <div className="flex items-center gap-1.5">
            <span className="text-xs text-[#645d52] font-medium">State B:</span>
            <select
              value={countryB}
              onChange={(e) => setCountryB(e.target.value)}
              className="bg-[#ffffff] border border-[#ded3be] text-[#1e293b] px-2.5 py-1.5 rounded-lg text-xs focus:outline-none focus:border-[#1e3a5f]"
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
        <div className="md:col-span-5 bg-[#f4efe6] border border-[#ded3be] p-3.5 rounded-lg space-y-2.5 shadow-sm">
          <div className="flex items-center justify-between border-b border-[#ded3be] pb-2">
            <div className="flex items-center gap-2 font-medium text-[#1a202c]">
              <span className="text-xl">{geoA.flag}</span>
              <div>
                <div className="text-xs font-bold text-[#1a202c] font-serif">{geoA.name}</div>
                <div className="text-[10px] text-[#786b59] font-mono">{geoA.iso3} • {geoA.un_region}</div>
              </div>
            </div>
            <span className="text-[10px] bg-[#eef3f8] text-[#1e3a5f] border border-[#c7d9ec] px-2 py-0.5 rounded font-mono font-semibold">
              {nodeA.income_group || geoA.income_group}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">GDP per Capita</div>
              <div className="font-bold text-[#1e3a5f] font-mono text-xs mt-0.5">
                ${Math.round(Number(nodeA.gdp_per_capita) || 0).toLocaleString()}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Population</div>
              <div className="font-bold text-[#1e293b] font-mono text-xs mt-0.5">
                {(Number(nodeA.population) || 0) >= 1000000
                  ? `${((Number(nodeA.population) || 0) / 1000000).toFixed(1)}M`
                  : `${Math.round((Number(nodeA.population) || 0) / 1000)}k`}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Dominant Religion</div>
              <div className="font-bold text-[#28534b] font-mono text-xs mt-0.5 truncate">
                {nodeA.dominant_religion || 'Christianity'}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Integration (MIPEX)</div>
              <div className="font-bold text-[#1e3a5f] font-mono text-xs mt-0.5">
                {(Number(nodeA.integration_policy_index) || 0.5).toFixed(2)} / 1.0
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Cultural Openness</div>
              <div className="font-bold text-[#945920] font-mono text-xs mt-0.5">
                {(Number(nodeA.cultural_openness_index) || 0.5).toFixed(2)}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Visa Openness</div>
              <div className="font-bold text-[#4a3d2e] font-mono text-xs mt-0.5">
                {(Number(nodeA.visa_openness_index) || 0).toFixed(1)}
              </div>
            </div>
          </div>
        </div>

        {/* Central Vector (1 col) */}
        <div className="md:col-span-1 flex flex-col items-center justify-center space-y-1 text-center py-1">
          <div className="text-[11px] font-mono font-bold text-[#1e3a5f] bg-[#e8e0cf] px-2 py-0.5 rounded">VS</div>
          <div className="w-px h-6 bg-[#ded3be]"></div>
          <div className="text-[9.5px] text-[#786b59] font-mono uppercase font-semibold">Bilateral</div>
        </div>

        {/* Country B Card (5 cols) */}
        <div className="md:col-span-5 bg-[#f4efe6] border border-[#ded3be] p-3.5 rounded-lg space-y-2.5 shadow-sm">
          <div className="flex items-center justify-between border-b border-[#ded3be] pb-2">
            <div className="flex items-center gap-2 font-medium text-[#1a202c]">
              <span className="text-xl">{geoB.flag}</span>
              <div>
                <div className="text-xs font-bold text-[#1a202c] font-serif">{geoB.name}</div>
                <div className="text-[10px] text-[#786b59] font-mono">{geoB.iso3} • {geoB.un_region}</div>
              </div>
            </div>
            <span className="text-[10px] bg-[#fcf4e8] text-[#945920] border border-[#e8d5bc] px-2 py-0.5 rounded font-mono font-semibold">
              {nodeB.income_group || geoB.income_group}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">GDP per Capita</div>
              <div className="font-bold text-[#1e3a5f] font-mono text-xs mt-0.5">
                ${Math.round(Number(nodeB.gdp_per_capita) || 0).toLocaleString()}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Population</div>
              <div className="font-bold text-[#1e293b] font-mono text-xs mt-0.5">
                {(Number(nodeB.population) || 0) >= 1000000
                  ? `${((Number(nodeB.population) || 0) / 1000000).toFixed(1)}M`
                  : `${Math.round((Number(nodeB.population) || 0) / 1000)}k`}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Dominant Religion</div>
              <div className="font-bold text-[#28534b] font-mono text-xs mt-0.5 truncate">
                {nodeB.dominant_religion || 'Christianity'}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Integration (MIPEX)</div>
              <div className="font-bold text-[#1e3a5f] font-mono text-xs mt-0.5">
                {(Number(nodeB.integration_policy_index) || 0.5).toFixed(2)} / 1.0
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Cultural Openness</div>
              <div className="font-bold text-[#945920] font-mono text-xs mt-0.5">
                {(Number(nodeB.cultural_openness_index) || 0.5).toFixed(2)}
              </div>
            </div>
            <div className="p-2 bg-[#ffffff] border border-[#e4dbcb] rounded-lg">
              <div className="text-[10px] text-[#786b59] font-medium">Visa Openness</div>
              <div className="font-bold text-[#4a3d2e] font-mono text-xs mt-0.5">
                {(Number(nodeB.visa_openness_index) || 0).toFixed(1)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bilateral Net Balance & Cultural Affinity Deck */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-2 border-t border-[#ded3be]">
        <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg space-y-0.5 shadow-sm">
          <div className="text-[#786b59] text-[10px] uppercase font-mono font-semibold">
            {geoA.name} → {geoB.name} Stock
          </div>
          <div className="text-lg font-bold text-[#1e3a5f] font-mono">
            {stockAtoB.toLocaleString()}
          </div>
          <div className="text-[10.5px] text-[#645d52]">
            {stockAtoB > 0 ? 'Active corridor stock' : 'No recorded stock'}
          </div>
        </div>

        <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg space-y-0.5 shadow-sm">
          <div className="text-[#786b59] text-[10px] uppercase font-mono font-semibold">
            {geoB.name} → {geoA.name} Stock
          </div>
          <div className="text-lg font-bold text-[#945920] font-mono">
            {stockBtoA.toLocaleString()}
          </div>
          <div className="text-[10.5px] text-[#645d52]">Reverse corridor stock</div>
        </div>

        <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg space-y-0.5 shadow-sm">
          <div className="text-[#786b59] text-[10px] uppercase font-mono font-semibold">Net Asymmetry Balance</div>
          <div className="text-lg font-bold text-[#28534b] font-mono">
            {netFlow >= 0 ? `+${netFlow.toLocaleString()}` : netFlow.toLocaleString()}
          </div>
          <div className="text-[10.5px] text-[#645d52]">
            Gradient towards {netFlow >= 0 ? geoB.name : geoA.name}
          </div>
        </div>

        <div className="p-3 bg-[#f4efe6] border border-[#ded3be] rounded-lg space-y-0.5 shadow-sm">
          <div className="text-[#786b59] text-[10px] uppercase font-mono font-semibold">Cultural & Religious Affinity</div>
          <div className="text-xs font-bold text-[#1a202c] font-mono flex items-center gap-1.5 mt-0.5">
            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
              nodeA.dominant_religion === nodeB.dominant_religion
                ? 'bg-[#eaf3ed] text-[#28534b] border border-[#b2d8be]'
                : 'bg-[#f8ede8] text-[#8b2626] border border-[#e8c0b5]'
            }`}>
              {nodeA.dominant_religion === nodeB.dominant_religion ? 'Shared Religion' : 'Cross-Religious'}
            </span>
          </div>
          <div className="text-[10px] text-[#645d52] font-mono mt-1">
            Distance: {Math.abs(Number(nodeA.cultural_openness_index || 0.5) - Number(nodeB.cultural_openness_index || 0.5)).toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
