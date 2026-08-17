import React, { useState, useRef } from 'react';
import { CountryGeo, COUNTRY_GEO_MAP, getCountryGeo, projectLatLng } from '../lib/countryGeo';
import { WORLD_COUNTRIES_SVG } from '../lib/worldGeoSvg';

export interface EdgeCorridor {
  origin_iso3: string;
  origin_name?: string;
  origin_lat?: number;
  origin_lng?: number;
  origin_flag?: string;
  dest_iso3: string;
  dest_name?: string;
  dest_lat?: number;
  dest_lng?: number;
  dest_flag?: string;
  weight?: number;
  predicted_influx?: number;
  strain_category?: string;
  strain_color?: string;
  year?: number;
}

interface WorldMapProps {
  corridors: EdgeCorridor[];
  selectedCountry: string | null;
  onSelectCountry: (iso3: string) => void;
  mode: 'corridors' | 'centrality' | 'communities' | 'simulation';
  nodesData?: Record<string, any>;
  simulationEpicenter?: string | null;
  year?: number;
  heightClass?: string;
}

export default function WorldMapInteractive({
  corridors,
  selectedCountry,
  onSelectCountry,
  mode,
  nodesData = {},
  simulationEpicenter,
  year = 2025,
  heightClass = 'h-[500px]',
}: WorldMapProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredCountry, setHoveredCountry] = useState<CountryGeo | null>(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [activeCorridorsLimit, setActiveCorridorsLimit] = useState(mode === 'simulation' ? 12 : 20);
  const [showScrollHint, setShowScrollHint] = useState(false);
  const scrollHintTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  // Filter corridors to display
  const displayCorridors = React.useMemo(() => {
    let list = corridors || [];
    if (selectedCountry && mode !== 'simulation') {
      list = list.filter(
        (c) => c.origin_iso3 === selectedCountry || c.dest_iso3 === selectedCountry
      );
    }
    return list.slice(0, activeCorridorsLimit);
  }, [corridors, selectedCountry, mode, activeCorridorsLimit]);

  // Pan & Zoom handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    // Only drag with primary left button
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  // Zoom anchored directly at specific client coordinates (cursor)
  const zoomAtPoint = (factor: number, clientX: number, clientY: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mouseX = clientX - rect.left;
    const mouseY = clientY - rect.top;

    setZoom((prevZoom) => {
      const newZoom = Math.max(0.75, Math.min(4.5, prevZoom * factor));
      const scaleRatio = newZoom / prevZoom;
      setPan((prevPan) => ({
        x: mouseX - (mouseX - prevPan.x) * scaleRatio,
        y: mouseY - (mouseY - prevPan.y) * scaleRatio,
      }));
      return newZoom;
    });
  };

  // Zoom centered on the viewport (for toolbar buttons)
  const zoomAtCenter = (factor: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    const cx = rect ? rect.width / 2 : 500;
    const cy = rect ? rect.height / 2 : 250;
    zoomAtPoint(factor, (rect?.left || 0) + cx, (rect?.top || 0) + cy);
  };

  // Wheel zoom anchored at cursor
  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.18 : 0.85;
      zoomAtPoint(zoomFactor, e.clientX, e.clientY);
    } else {
      // Show polite helper badge without blocking page scroll
      setShowScrollHint(true);
      if (scrollHintTimeoutRef.current) clearTimeout(scrollHintTimeoutRef.current);
      scrollHintTimeoutRef.current = setTimeout(() => setShowScrollHint(false), 1500);
    }
  };

  // Double click zooms towards cursor
  const handleDoubleClick = (e: React.MouseEvent) => {
    zoomAtPoint(1.35, e.clientX, e.clientY);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className={`relative w-full ${heightClass} bg-[#070b12] rounded-xl overflow-hidden border border-slate-800 select-none shadow-md`}>
      {/* Floating HUD Controls */}
      <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2">
        <div className="flex items-center bg-[#0d131f]/95 border border-slate-700/80 px-2.5 py-1 rounded-md text-xs space-x-2 shadow-sm">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
          <span className="font-semibold text-slate-200 uppercase tracking-wide text-[10.5px]">
            {mode === 'simulation'
              ? 'Crisis Displacement Vectors'
              : mode === 'centrality'
              ? 'Node Centrality Topography'
              : mode === 'communities'
              ? 'Alliance Clusters'
              : `Bilateral Corridors (${year})`}
          </span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400 font-mono text-[10.5px]">{displayCorridors.length} vectors</span>
        </div>

        {selectedCountry && (
          <button
            onClick={() => onSelectCountry('')}
            className="flex items-center gap-1 bg-blue-950/90 border border-blue-600/60 text-blue-300 hover:bg-blue-900/60 px-2 py-1 rounded-md text-[11px] transition"
          >
            <span>Focus: {getCountryGeo(selectedCountry).name}</span>
            <span className="text-[10px] ml-0.5 bg-blue-800/80 px-1 rounded">✕</span>
          </button>
        )}
      </div>

      {/* Map Control Tools (Zoom, Reset, Limits) */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
        <div className="flex bg-[#0d131f]/95 border border-slate-700/80 rounded-md p-0.5 text-xs text-slate-300 shadow-sm">
          <button
            onClick={() => zoomAtCenter(1.25)}
            className="px-2 py-0.5 hover:bg-slate-800 rounded font-medium text-slate-200 text-xs transition"
            title="Zoom In (or ⌘/Ctrl + scroll)"
          >
            +
          </button>
          <button
            onClick={() => zoomAtCenter(0.8)}
            className="px-2 py-0.5 hover:bg-slate-800 rounded font-medium text-slate-200 text-xs transition"
            title="Zoom Out (or ⌘/Ctrl + scroll)"
          >
            −
          </button>
          <button
            onClick={resetView}
            className="px-2 py-0.5 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 text-[10.5px] transition"
            title="Reset Map Pan/Zoom"
          >
            Reset
          </button>
        </div>

        <div className="bg-[#0d131f]/95 border border-slate-700/80 px-2 py-0.5 rounded-md text-xs flex items-center gap-1.5">
          <span className="text-slate-400 text-[10.5px]">Vectors:</span>
          {[10, 20, 40].map((num) => (
            <button
              key={num}
              onClick={() => setActiveCorridorsLimit(num)}
              className={`px-1.5 py-0.5 rounded text-[10.5px] font-mono transition ${
                activeCorridorsLimit === num
                  ? 'bg-blue-600 text-white font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {num}
            </button>
          ))}
        </div>
      </div>

      {/* Map Legend Overlay */}
      <div className="absolute bottom-3 left-3 z-20 bg-[#0d131f]/95 border border-slate-700/80 px-2.5 py-1.5 rounded-md text-xs text-slate-300 space-y-0.5 shadow-sm">
        <div className="font-medium text-slate-200 text-[10px] flex items-center justify-between gap-3">
          <span>{mode === 'simulation' ? 'Capacity Strain' : 'Topology Key'}</span>
          <span className="text-[9.5px] text-slate-500 font-mono">{year}</span>
        </div>
        {mode === 'simulation' ? (
          <div className="grid grid-cols-2 gap-x-2.5 gap-y-0.5 text-[9.5px] text-slate-400">
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500"></span>
              <span>Critical (&gt;12/1k)</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
              <span>High (4-12/1k)</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
              <span>Moderate (1-4/1k)</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
              <span>Manageable (&lt;1/1k)</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2.5 text-[9.5px] text-slate-400">
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
              <span>Attractor Hub</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
              <span>Transit Bridge</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
              <span>Node</span>
            </div>
          </div>
        )}
      </div>

      {/* Scroll Helper Toast (Google Maps style) */}
      {showScrollHint && (
        <div className="absolute inset-0 z-30 pointer-events-none flex items-center justify-center bg-black/40 backdrop-blur-[1px] transition-opacity duration-300">
          <div className="bg-[#0b101b]/95 border border-slate-700 text-slate-200 px-4 py-2 rounded-lg text-xs font-mono shadow-xl flex items-center gap-2">
            <span className="text-blue-400">ℹ</span>
            <span>Use <strong>⌘ + scroll</strong> (or <strong>Ctrl + scroll</strong>) to zoom map</span>
          </div>
        </div>
      )}

      {/* SVG Map Canvas Surface */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
        style={{ touchAction: 'pan-y' }}
        className={`w-full h-full cursor-${isDragging ? 'grabbing' : 'grab'}`}
      >
        <svg
          viewBox="0 0 1000 500"
          className="w-full h-full"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: '0 0',
            transition: isDragging ? 'none' : 'transform 0.12s ease-out',
          }}
        >
          {/* Subtle Grid Lat/Long Reference Lines */}
          <g stroke="rgba(30, 41, 59, 0.4)" strokeWidth="0.5" fill="none">
            <line x1="0" y1="125" x2="1000" y2="125" strokeDasharray="2 3" />
            <line x1="0" y1="250" x2="1000" y2="250" />
            <line x1="0" y1="375" x2="1000" y2="375" strokeDasharray="2 3" />
            <line x1="250" y1="0" x2="250" y2="500" strokeDasharray="2 3" />
            <line x1="500" y1="0" x2="500" y2="500" />
            <line x1="750" y1="0" x2="750" y2="500" strokeDasharray="2 3" />
          </g>

          {/* High-Precision Official Natural Earth Country Polygons */}
          <g>
            {WORLD_COUNTRIES_SVG.map((c) => {
              const isSelected = selectedCountry === c.iso3;
              const isEpicenter = simulationEpicenter === c.iso3;
              const nodeInfo = nodesData[c.iso3] || {};

              let countryFill = '#131b2a';
              let countryStroke = '#223147';
              let strokeWidth = 0.5;

              if (isEpicenter) {
                countryFill = '#450a0a';
                countryStroke = '#ef4444';
                strokeWidth = 1.0;
              } else if (isSelected) {
                countryFill = '#1e3a8a';
                countryStroke = '#60a5fa';
                strokeWidth = 1.0;
              } else if (mode === 'communities' && nodeInfo.louvain_community !== undefined) {
                const comm = Number(nodeInfo.louvain_community || 0);
                const commColors = [
                  '#1e293b',
                  '#1e3a8a',
                  '#065f46',
                  '#78350f',
                  '#581c87',
                  '#831843',
                  '#134e4a',
                  '#312e81',
                ];
                countryFill = commColors[comm % commColors.length] || '#131b2a';
                countryStroke = '#334155';
              }

              return (
                <path
                  key={c.iso3 + c.name}
                  d={c.d}
                  fill={countryFill}
                  stroke={countryStroke}
                  strokeWidth={strokeWidth}
                  className="hover:fill-[#1b263b] hover:stroke-slate-400 transition-colors cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (c.iso3 && c.iso3 !== 'UNK') {
                      onSelectCountry(c.iso3);
                    }
                  }}
                  onMouseEnter={(e) => {
                    const rect = containerRef.current?.getBoundingClientRect();
                    if (rect) {
                      setHoverPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
                    }
                    if (c.iso3 && COUNTRY_GEO_MAP[c.iso3]) {
                      setHoveredCountry(COUNTRY_GEO_MAP[c.iso3]);
                    } else {
                      setHoveredCountry({
                        iso3: c.iso3,
                        name: c.name,
                        lat: 0,
                        lng: 0,
                        continent: 'World',
                        un_region: 'Global',
                        income_group: 'Sovereign',
                        flag: '🌐',
                      });
                    }
                  }}
                  onMouseLeave={() => setHoveredCountry(null)}
                />
              );
            })}
          </g>

          {/* Static Dotted / Dashed Geodesic Flight Arcs */}
          <g fill="none">
            {displayCorridors.map((c, idx) => {
              const oGeo = getCountryGeo(c.origin_iso3);
              const dGeo = getCountryGeo(c.dest_iso3);
              const p1 = projectLatLng(oGeo.lat, oGeo.lng, 1000, 500);
              const p2 = projectLatLng(dGeo.lat, dGeo.lng, 1000, 500);

              const midX = (p1.x + p2.x) / 2;
              const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
              const midY = Math.min(p1.y, p2.y) - Math.min(90, dist * 0.25);

              const strokeColor =
                c.strain_color ||
                (mode === 'simulation'
                  ? 'rgba(239, 68, 68, 0.75)'
                  : selectedCountry === c.origin_iso3
                  ? 'rgba(96, 165, 250, 0.85)'
                  : 'rgba(148, 163, 184, 0.45)');

              const strokeW = Math.max(1, Math.min(2.5, (c.predicted_influx || c.weight || 10000) / 400000));

              return (
                <g key={`arc-${c.origin_iso3}-${c.dest_iso3}-${idx}`}>
                  {/* Static Dotted Arc */}
                  <path
                    d={`M ${p1.x} ${p1.y} Q ${midX} ${midY} ${p2.x} ${p2.y}`}
                    stroke={strokeColor}
                    strokeWidth={strokeW}
                    strokeDasharray="3 4"
                    opacity={0.85}
                  />
                  {/* Destination Circle */}
                  <circle
                    cx={p2.x}
                    cy={p2.y}
                    r={2.5}
                    fill={strokeColor}
                    opacity={0.9}
                  />
                </g>
              );
            })}
          </g>

          {/* Simulation Epicenter Static Indicator */}
          {simulationEpicenter && (
            <g>
              {(() => {
                const epicGeo = getCountryGeo(simulationEpicenter);
                const pt = projectLatLng(epicGeo.lat, epicGeo.lng, 1000, 500);
                return (
                  <>
                    <circle cx={pt.x} cy={pt.y} r="14" fill="none" stroke="rgba(239, 68, 68, 0.35)" strokeWidth="1.5" strokeDasharray="3 3" />
                    <circle cx={pt.x} cy={pt.y} r="6" fill="rgba(239, 68, 68, 0.25)" stroke="#ef4444" strokeWidth="1.5" />
                    <circle cx={pt.x} cy={pt.y} r="2.5" fill="#ef4444" />
                  </>
                );
              })()}
            </g>
          )}

          {/* Country Centroid Dots (when selected or hovered) */}
          <g>
            {Object.values(COUNTRY_GEO_MAP).map((c) => {
              const pt = projectLatLng(c.lat, c.lng, 1000, 500);
              const isSelected = selectedCountry === c.iso3;
              const isEpicenter = simulationEpicenter === c.iso3;

              if (!isSelected && !isEpicenter) return null;

              return (
                <g key={c.iso3} className="pointer-events-none">
                  <circle
                    cx={pt.x}
                    cy={pt.y}
                    r={isSelected ? 4.5 : 5}
                    fill={isEpicenter ? '#ef4444' : '#3b82f6'}
                    stroke="#ffffff"
                    strokeWidth={1.2}
                  />
                  <text
                    x={pt.x}
                    y={pt.y - 8}
                    textAnchor="middle"
                    fill="#93c5fd"
                    fontSize="9.5"
                    fontWeight="600"
                  >
                    {c.flag} {c.name}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Tooltip Card */}
      {hoveredCountry && (
        <div
          className="absolute z-30 pointer-events-none bg-[#0b101b]/95 border border-slate-700 px-2.5 py-1.5 rounded-md shadow-lg text-xs space-y-0.5 transform -translate-x-1/2 -translate-y-full"
          style={{
            left: Math.max(70, Math.min(hoverPos.x, 920)),
            top: Math.max(50, hoverPos.y - 10),
          }}
        >
          <div className="flex items-center gap-1.5 font-semibold text-slate-100 text-xs">
            <span>{hoveredCountry.flag}</span>
            <span>{hoveredCountry.name}</span>
            <span className="text-[9.5px] bg-slate-800 text-slate-400 px-1 py-0.2 rounded font-mono">
              {hoveredCountry.iso3}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-x-2 text-[9.5px] text-slate-400">
            <div>Region: <span className="text-slate-200">{hoveredCountry.un_region}</span></div>
            <div>Income: <span className="text-slate-200">{hoveredCountry.income_group}</span></div>
          </div>
          {nodesData[hoveredCountry.iso3] && (
            <div className="pt-0.5 border-t border-slate-800 grid grid-cols-2 gap-x-2 text-[9px] text-slate-300 font-mono">
              <div>PageRank: {(nodesData[hoveredCountry.iso3].pagerank || 0).toFixed(4)}</div>
              <div>Cluster: {nodesData[hoveredCountry.iso3].community_label || 'Cluster 0'}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
