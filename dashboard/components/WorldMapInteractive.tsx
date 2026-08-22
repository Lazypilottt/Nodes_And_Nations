import React, { useState, useRef, useEffect } from 'react';
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
  mode: 'corridors' | 'centrality' | 'communities' | 'simulation' | 'religion';
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

  // Pan & Drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
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

  // Smooth Zoom anchored directly at specific client coordinates (cursor)
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

  // Wheel zoom anchored at cursor with gentle, non-aggressive sensitivity
  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      // Gentle proportional damping to avoid over-sensitivity
      const delta = Math.max(-100, Math.min(100, e.deltaY));
      const zoomFactor = Math.pow(0.9985, delta);
      zoomAtPoint(zoomFactor, e.clientX, e.clientY);
    } else {
      setShowScrollHint(true);
      if (scrollHintTimeoutRef.current) clearTimeout(scrollHintTimeoutRef.current);
      scrollHintTimeoutRef.current = setTimeout(() => setShowScrollHint(false), 1500);
    }
  };

  // Double click zooms towards cursor
  const handleDoubleClick = (e: React.MouseEvent) => {
    zoomAtPoint(1.3, e.clientX, e.clientY);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Auto-zoom into clicked country
  useEffect(() => {
    if (selectedCountry && selectedCountry !== 'UNK') {
      const geo = getCountryGeo(selectedCountry);
      if (geo && (geo.lat !== 0 || geo.lng !== 0)) {
        const pt = projectLatLng(geo.lat, geo.lng, 1000, 500);
        const targetZoom = 2.4;
        const rect = containerRef.current?.getBoundingClientRect();
        const W = rect ? rect.width : 1000;
        const H = rect ? rect.height : 500;

        const screenX = (pt.x / 1000) * W;
        const screenY = (pt.y / 500) * H;

        const targetPanX = W / 2 - screenX * targetZoom;
        const targetPanY = H / 2 - screenY * targetZoom;

        setZoom(targetZoom);
        setPan({ x: targetPanX, y: targetPanY });
      }
    }
  }, [selectedCountry]);

  return (
    <div className={`relative w-full ${heightClass} bg-[#ece5d6] rounded-xl overflow-hidden border border-[#d8ccb8] select-none shadow-sm`}>
      {/* Floating HUD Controls */}
      <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2">
        <div className="flex items-center bg-[#fbf9f4]/95 border border-[#d8ccb8] px-3 py-1.5 rounded-lg text-xs space-x-2 shadow-sm">
          <span className="w-2 h-2 rounded-full bg-[#1e3a5f]"></span>
          <span className="font-semibold text-[#1e293b] uppercase tracking-wide text-[10.5px]">
            {mode === 'simulation'
              ? 'Crisis Displacement Vectors'
              : mode === 'centrality'
              ? 'Node Centrality Topography'
              : mode === 'communities'
              ? 'Alliance Clusters'
              : mode === 'religion'
              ? 'Dominant Religion Demographics'
              : `Bilateral Corridors (${year})`}
          </span>
          <span className="text-[#baa990]">|</span>
          <span className="text-[#786b59] font-mono text-[10.5px]">{displayCorridors.length} vectors</span>
        </div>

        {selectedCountry && (
          <button
            onClick={() => {
              onSelectCountry('');
              resetView();
            }}
            className="flex items-center gap-1.5 bg-[#fbf9f4] border border-[#1e3a5f] text-[#1e3a5f] hover:bg-[#eae1cf] px-2.5 py-1.5 rounded-lg text-[11px] font-semibold transition shadow-sm"
          >
            <span>Focus: {getCountryGeo(selectedCountry).name}</span>
            <span className="text-[10px] bg-[#1e3a5f] text-white px-1.5 py-0.2 rounded">✕</span>
          </button>
        )}
      </div>

      {/* Map Control Tools (Zoom, Reset, Limits) */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
        <div className="flex bg-[#fbf9f4]/95 border border-[#d8ccb8] rounded-lg p-0.5 text-xs text-[#4a3d2e] shadow-sm">
          <button
            onClick={() => zoomAtCenter(1.25)}
            className="px-2.5 py-1 hover:bg-[#ede4d4] rounded font-bold text-[#1e293b] text-xs transition"
            title="Zoom In (or ⌘/Ctrl + scroll)"
          >
            +
          </button>
          <button
            onClick={() => zoomAtCenter(0.8)}
            className="px-2.5 py-1 hover:bg-[#ede4d4] rounded font-bold text-[#1e293b] text-xs transition"
            title="Zoom Out (or ⌘/Ctrl + scroll)"
          >
            −
          </button>
          <button
            onClick={resetView}
            className="px-2.5 py-1 hover:bg-[#ede4d4] rounded text-[#6d5c46] hover:text-[#1a202c] text-[11px] font-medium transition"
            title="Reset Map Pan/Zoom"
          >
            Reset
          </button>
        </div>

        <div className="bg-[#fbf9f4]/95 border border-[#d8ccb8] px-2.5 py-1 rounded-lg text-xs flex items-center gap-1.5 shadow-sm">
          <span className="text-[#786b59] text-[10.5px] font-medium">Vectors:</span>
          {[10, 20, 40].map((num) => (
            <button
              key={num}
              onClick={() => setActiveCorridorsLimit(num)}
              className={`px-1.5 py-0.5 rounded text-[10.5px] font-mono transition ${
                activeCorridorsLimit === num
                  ? 'bg-[#1e3a5f] text-white font-bold'
                  : 'text-[#6d5c46] hover:text-[#1e293b]'
              }`}
            >
              {num}
            </button>
          ))}
        </div>
      </div>

      {/* Map Legend Overlay */}
      <div className="absolute bottom-3 left-3 z-20 bg-[#fbf9f4]/95 border border-[#d8ccb8] px-3 py-2 rounded-lg text-xs text-[#334155] space-y-1 shadow-sm">
        <div className="font-semibold text-[#1e293b] text-[10.5px] flex items-center justify-between gap-3 border-b border-[#e8dfcf] pb-1">
          <span>{mode === 'simulation' ? 'Capacity Strain' : mode === 'religion' ? 'Dominant Religion' : 'Topology Key'}</span>
          <span className="text-[10px] text-[#786b59] font-mono font-bold">{year}</span>
        </div>
        {mode === 'simulation' ? (
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] text-[#554a3b]">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#8b2626]"></span>
              <span>Critical (&gt;12/1k)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#945920]"></span>
              <span>High (4-12/1k)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#1e3a5f]"></span>
              <span>Moderate (1-4/1k)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#28534b]"></span>
              <span>Manageable (&lt;1/1k)</span>
            </div>
          </div>
        ) : mode === 'religion' ? (
          <div className="grid grid-cols-3 gap-x-3 gap-y-1 text-[10px] text-[#554a3b]">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#1e3a5f]"></span>
              <span>Christianity</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#28534b]"></span>
              <span>Islam</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#945920]"></span>
              <span>Hinduism</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#8c6d23]"></span>
              <span>Buddhism</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#5c3858]"></span>
              <span>Judaism</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#556270]"></span>
              <span>Unaffiliated</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-[10px] text-[#554a3b]">
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-[#1e3a5f]"></span>
              <span>Attractor Hub</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-[#945920]"></span>
              <span>Transit Bridge</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-[#8a7c6a]"></span>
              <span>Sovereign Node</span>
            </div>
          </div>
        )}
      </div>

      {/* Scroll Helper Toast */}
      {showScrollHint && (
        <div className="absolute inset-0 z-30 pointer-events-none flex items-center justify-center bg-black/20 backdrop-blur-[1px] transition-opacity duration-300">
          <div className="bg-[#fbf9f4] border border-[#d8ccb8] text-[#1e293b] px-4 py-2 rounded-lg text-xs font-mono shadow-xl flex items-center gap-2">
            <span className="text-[#1e3a5f] font-bold">ℹ</span>
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
            transition: isDragging ? 'none' : 'transform 0.16s ease-out',
          }}
        >
          {/* Subtle Grid Lat/Long Reference Lines */}
          <g stroke="rgba(180, 165, 140, 0.45)" strokeWidth="0.5" fill="none">
            <line x1="0" y1="125" x2="1000" y2="125" strokeDasharray="2 3" />
            <line x1="0" y1="250" x2="1000" y2="250" />
            <line x1="0" y1="375" x2="1000" y2="375" strokeDasharray="2 3" />
            <line x1="250" y1="0" x2="250" y2="500" strokeDasharray="2 3" />
            <line x1="500" y1="0" x2="500" y2="500" />
            <line x1="750" y1="0" x2="750" y2="500" strokeDasharray="2 3" />
          </g>

          {/* High-Precision Natural Earth Country Polygons (Warm Linen Palette) */}
          <g>
            {WORLD_COUNTRIES_SVG.map((c) => {
              const isSelected = selectedCountry === c.iso3;
              const isEpicenter = simulationEpicenter === c.iso3;
              const nodeInfo = nodesData[c.iso3] || {};

              let countryFill = '#dfd7c6';
              let countryStroke = '#c5b8a5';
              let strokeWidth = 0.5;

              if (isEpicenter) {
                countryFill = '#8b2626';
                countryStroke = '#cf4848';
                strokeWidth = 1.2;
              } else if (isSelected) {
                countryFill = '#1e3a5f';
                countryStroke = '#945920';
                strokeWidth = 1.2;
              } else if (mode === 'religion') {
                const rel = nodeInfo.dominant_religion || 'Unknown';
                const religionColors: Record<string, string> = {
                  'Christianity': '#1e3a5f', // Navy
                  'Islam': '#28534b',        // Cypress
                  'Hinduism': '#945920',     // Cognac
                  'Buddhism': '#8c6d23',     // Ochre Gold
                  'Judaism': '#5c3858',      // Plum
                  'Unaffiliated': '#556270', // Slate
                };
                countryFill = religionColors[rel] || '#786b59';
                countryStroke = '#ded3be';
              } else if (mode === 'communities' && nodeInfo.louvain_community !== undefined) {
                const comm = Number(nodeInfo.louvain_community || 0);
                const heritageColors = [
                  '#244b7a', // Navy
                  '#28534b', // Cypress
                  '#945920', // Cognac
                  '#5c3858', // Plum
                  '#3b5266', // Slate Spruce
                  '#8c6d23', // Ochre Gold
                  '#7a2b38', // Burgundy
                  '#4f5d30', // Moss
                ];
                countryFill = heritageColors[comm % heritageColors.length] || '#dfd7c6';
                countryStroke = '#a89880';
              }

              return (
                <path
                  key={c.iso3 + c.name}
                  d={c.d}
                  fill={countryFill}
                  stroke={countryStroke}
                  strokeWidth={strokeWidth}
                  className="hover:fill-[#cebeaa] hover:stroke-[#8f7e68] transition-colors cursor-pointer"
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
                  ? 'rgba(139, 38, 38, 0.85)'
                  : selectedCountry === c.origin_iso3
                  ? 'rgba(30, 58, 95, 0.9)'
                  : 'rgba(92, 75, 55, 0.55)');

              const strokeW = Math.max(1, Math.min(2.5, (c.predicted_influx || c.weight || 10000) / 400000));

              return (
                <g key={`arc-${c.origin_iso3}-${c.dest_iso3}-${idx}`}>
                  <path
                    d={`M ${p1.x} ${p1.y} Q ${midX} ${midY} ${p2.x} ${p2.y}`}
                    stroke={strokeColor}
                    strokeWidth={strokeW}
                    strokeDasharray="3 4"
                    opacity={0.9}
                  />
                  <circle
                    cx={p2.x}
                    cy={p2.y}
                    r={2.5}
                    fill={strokeColor}
                    opacity={0.95}
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
                    <circle cx={pt.x} cy={pt.y} r="14" fill="none" stroke="rgba(139, 38, 38, 0.4)" strokeWidth="1.5" strokeDasharray="3 3" />
                    <circle cx={pt.x} cy={pt.y} r="6" fill="rgba(139, 38, 38, 0.25)" stroke="#8b2626" strokeWidth="1.5" />
                    <circle cx={pt.x} cy={pt.y} r="2.5" fill="#8b2626" />
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
                    fill={isEpicenter ? '#8b2626' : '#1e3a5f'}
                    stroke="#ffffff"
                    strokeWidth={1.4}
                  />
                  <text
                    x={pt.x}
                    y={pt.y - 8}
                    textAnchor="middle"
                    fill="#1e3a5f"
                    fontSize="9.5"
                    fontWeight="700"
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
          className="absolute z-30 pointer-events-none bg-[#ffffff] border border-[#d8ccb8] px-3 py-2 rounded-lg shadow-xl text-xs space-y-1 transform -translate-x-1/2 -translate-y-full"
          style={{
            left: Math.max(70, Math.min(hoverPos.x, 920)),
            top: Math.max(50, hoverPos.y - 10),
          }}
        >
          <div className="flex items-center gap-1.5 font-bold text-[#1e293b] text-xs">
            <span>{hoveredCountry.flag}</span>
            <span>{hoveredCountry.name}</span>
            <span className="text-[9.5px] bg-[#f4efe6] text-[#6d5c46] border border-[#ddd2be] px-1.5 py-0.2 rounded font-mono">
              {hoveredCountry.iso3}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-x-3 text-[10px] text-[#645d52]">
            <div>Region: <span className="text-[#1e293b] font-medium">{hoveredCountry.un_region}</span></div>
            <div>Income: <span className="text-[#1e293b] font-medium">{hoveredCountry.income_group}</span></div>
          </div>
          {nodesData[hoveredCountry.iso3] && (
            <div className="pt-1 border-t border-[#f0e8db] space-y-0.5 text-[9.5px] text-[#4a3d2e] font-mono">
              <div className="flex justify-between gap-2">
                <span>PageRank: {(nodesData[hoveredCountry.iso3].pagerank || 0).toFixed(4)}</span>
                <span>Religion: <strong className="text-[#1e3a5f]">{nodesData[hoveredCountry.iso3].dominant_religion || 'Unknown'}</strong></span>
              </div>
              <div className="flex justify-between gap-2">
                <span>MIPEX: {Number(nodesData[hoveredCountry.iso3].integration_policy_index || 0.5).toFixed(2)}</span>
                <span>Openness: {Number(nodesData[hoveredCountry.iso3].cultural_openness_index || 0.5).toFixed(2)}</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
