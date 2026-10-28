import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { COUNTRY_GEO_MAP, getCountryGeo, projectLatLng } from '../lib/countryGeo';
import { WORLD_COUNTRIES_SVG } from '../lib/worldGeoSvg';
import {
  ACCENT, LAND, RELIGION_COLORS, STRAIN_COLORS, categorical, fmtCompact, fmtNum, sequential,
  type StrainLevel,
} from '../lib/theme';
import type { MapMode, NodeRow } from '../lib/types';
import { Dot } from './ui';

export interface MapArc {
  origin: string;
  dest: string;
  value: number;
  color?: string;
}

export interface SimulationOverlay {
  epicenter: string;
  recipients: Record<string, { influx: number; level: StrainLevel }>;
}

interface Props {
  mode: MapMode;
  nodes: Record<string, NodeRow>;
  arcs: MapArc[];
  selected?: string | null;
  onSelect?: (iso3: string | null) => void;
  simulation?: SimulationOverlay | null;
  className?: string;
}

const W = 1000;
const H = 500;
const MIN_W = W / 10;
const EPICENTER = '#7f1d1d';

type ViewBox = { x: number; y: number; w: number; h: number };
const FULL: ViewBox = { x: 0, y: 0, w: W, h: H };

function clampView(v: ViewBox): ViewBox {
  const w = Math.max(MIN_W, Math.min(W, v.w));
  const h = (w * H) / W;
  return {
    w,
    h,
    x: Math.max(0, Math.min(W - w, v.x)),
    y: Math.max(0, Math.min(H - h, v.y)),
  };
}

const CountryLayer = React.memo(function CountryLayer({
  fills,
  selected,
}: {
  fills: Record<string, { fill: string; opacity?: number }>;
  selected?: string | null;
}) {
  return (
    <g>
      {WORLD_COUNTRIES_SVG.map((c) => {
        const f = fills[c.iso3];
        const isSelected = selected === c.iso3;
        return (
          <path
            key={c.iso3 + c.name}
            d={c.d}
            data-iso={c.iso3}
            fill={f?.fill ?? LAND}
            fillOpacity={f?.opacity ?? 1}
            stroke={isSelected ? '#18181b' : '#ffffff'}
            strokeWidth={isSelected ? 1.4 : 0.5}
            vectorEffect="non-scaling-stroke"
            className="cursor-pointer transition-opacity hover:opacity-75"
          />
        );
      })}
    </g>
  );
});

export default function WorldMap({ mode, nodes, arcs, selected, onSelect, simulation, className = 'h-[460px]' }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [view, setView] = useState<ViewBox>(FULL);
  const [hover, setHover] = useState<{ iso3: string; x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; y: number; view: ViewBox; scale: number; moved: boolean } | null>(null);

  const toSvgPoint = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    return pt.matrixTransform(ctm.inverse());
  }, []);

  const zoomAt = useCallback((factor: number, sx: number, sy: number) => {
    setView((v) => {
      const w = Math.max(MIN_W, Math.min(W, v.w / factor));
      const k = w / v.w;
      return clampView({ x: sx - (sx - v.x) * k, y: sy - (sy - v.y) * k, w, h: (w * H) / W });
    });
  }, []);

  const zoomCenter = (factor: number) => zoomAt(factor, view.x + view.w / 2, view.y + view.h / 2);

  // React registers wheel listeners as passive, so preventDefault() would be
  // ignored and the page itself would zoom. Attach a native listener instead.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const p = toSvgPoint(e.clientX, e.clientY);
      if (p) zoomAt(Math.exp(-e.deltaY * 0.0025), p.x, p.y);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [toSvgPoint, zoomAt]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const ctm = svgRef.current?.getScreenCTM();
    drag.current = { x: e.clientX, y: e.clientY, view, scale: ctm?.a || 1, moved: false };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (d) {
      const dx = e.clientX - d.x;
      const dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) > 3) {
        d.moved = true;
        (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
      }
      if (d.moved) {
        setView(clampView({ ...d.view, x: d.view.x - dx / d.scale, y: d.view.y - dy / d.scale }));
        setHover(null);
        return;
      }
    }
    const target = (e.target as Element).closest?.('path[data-iso]');
    const rect = containerRef.current?.getBoundingClientRect();
    if (target && rect) {
      setHover({ iso3: target.getAttribute('data-iso')!, x: e.clientX - rect.left, y: e.clientY - rect.top });
    } else if (hover) {
      setHover(null);
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (d?.moved || !onSelect) return;
    const target = (e.target as Element).closest?.('path[data-iso]');
    const iso = target?.getAttribute('data-iso');
    if (iso && COUNTRY_GEO_MAP[iso]) onSelect(iso === selected ? null : iso);
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    const p = toSvgPoint(e.clientX, e.clientY);
    if (p) zoomAt(1.8, p.x, p.y);
  };

  // ── Fills per mode ────────────────────────────────────────────────────────
  const fills = useMemo(() => {
    const out: Record<string, { fill: string; opacity?: number }> = {};
    if (mode === 'centrality') {
      const ranked = Object.values(nodes)
        .filter((n) => n.pagerank != null)
        .sort((a, b) => (a.pagerank ?? 0) - (b.pagerank ?? 0));
      ranked.forEach((n, i) => (out[n.iso3] = { fill: sequential(i / Math.max(1, ranked.length - 1)) }));
    } else if (mode === 'communities') {
      for (const n of Object.values(nodes)) {
        if (n.louvain_community != null) out[n.iso3] = { fill: categorical(n.louvain_community) };
      }
    } else if (mode === 'religion') {
      for (const n of Object.values(nodes)) {
        const c = n.dominant_religion ? RELIGION_COLORS[n.dominant_religion] : undefined;
        if (c) out[n.iso3] = { fill: c, opacity: 0.85 };
      }
    } else if (mode === 'simulation' && simulation) {
      for (const [iso, r] of Object.entries(simulation.recipients)) {
        out[iso] = { fill: STRAIN_COLORS[r.level], opacity: 0.7 };
      }
      out[simulation.epicenter] = { fill: EPICENTER };
    }
    if (selected && mode === 'corridors') out[selected] = { fill: ACCENT };
    return out;
  }, [mode, nodes, simulation, selected]);

  // ── Arcs ──────────────────────────────────────────────────────────────────
  const arcPaths = useMemo(() => {
    const valid = arcs.filter((a) => COUNTRY_GEO_MAP[a.origin] && COUNTRY_GEO_MAP[a.dest] && a.value > 0);
    if (valid.length === 0) return [];
    const logs = valid.map((a) => Math.log(a.value));
    const lo = Math.min(...logs);
    const hi = Math.max(...logs);
    return valid.map((a, i) => {
      const o = getCountryGeo(a.origin);
      const d = getCountryGeo(a.dest);
      const p1 = projectLatLng(o.lat, o.lng, W, H);
      const p2 = projectLatLng(d.lat, d.lng, W, H);
      const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y);
      const cx = (p1.x + p2.x) / 2;
      const cy = Math.min(p1.y, p2.y) - Math.min(90, dist * 0.28);
      const t = hi > lo ? (logs[i] - lo) / (hi - lo) : 1;
      return {
        key: `${a.origin}-${a.dest}-${i}`,
        d: `M ${p1.x} ${p1.y} Q ${cx} ${cy} ${p2.x} ${p2.y}`,
        end: p2,
        width: 0.8 + 2.4 * t,
        color: a.color ?? ACCENT,
      };
    });
  }, [arcs]);

  const markerR = (2.6 * view.w) / W;
  const epicenterPt = simulation ? (() => {
    const g = getCountryGeo(simulation.epicenter);
    return projectLatLng(g.lat, g.lng, W, H);
  })() : null;

  const hoverInfo = hover ? { geo: getCountryGeo(hover.iso3), node: nodes[hover.iso3] } : null;
  const containerW = containerRef.current?.clientWidth ?? 800;
  const isZoomed = view.w < W;

  return (
    <div
      ref={containerRef}
      className={`relative w-full select-none overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50 ${className}`}
    >
      <svg
        ref={svgRef}
        viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
        preserveAspectRatio="xMidYMid meet"
        className={`h-full w-full touch-none ${drag.current?.moved ? 'cursor-grabbing' : 'cursor-grab'}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          drag.current = null;
          setHover(null);
        }}
        onDoubleClick={onDoubleClick}
        role="img"
        aria-label="World map of migration"
      >
        <CountryLayer fills={fills} selected={selected} />

        <g fill="none" pointerEvents="none">
          {arcPaths.map((a) => (
            <g key={a.key}>
              <path d={a.d} stroke={a.color} strokeWidth={a.width} strokeOpacity={0.7} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
              <circle cx={a.end.x} cy={a.end.y} r={markerR} fill={a.color} />
            </g>
          ))}
        </g>

        {epicenterPt && (
          <g pointerEvents="none">
            <circle cx={epicenterPt.x} cy={epicenterPt.y} r={markerR * 4} fill={EPICENTER} fillOpacity={0.15} />
            <circle cx={epicenterPt.x} cy={epicenterPt.y} r={markerR * 1.4} fill={EPICENTER} stroke="#fff" strokeWidth={1} vectorEffect="non-scaling-stroke" />
          </g>
        )}
      </svg>

      {/* Controls */}
      <div className="absolute right-3 top-3 flex flex-col overflow-hidden rounded-lg border border-zinc-200 bg-white/95 shadow-sm backdrop-blur">
        <button aria-label="Zoom in" onClick={() => zoomCenter(1.5)} className="h-8 w-8 text-zinc-700 hover:bg-zinc-50">+</button>
        <button aria-label="Zoom out" onClick={() => zoomCenter(1 / 1.5)} className="h-8 w-8 border-t border-zinc-200 text-zinc-700 hover:bg-zinc-50">−</button>
        <button
          aria-label="Reset view"
          onClick={() => setView(FULL)}
          disabled={!isZoomed}
          className="h-8 w-8 border-t border-zinc-200 text-xs text-zinc-700 hover:bg-zinc-50 disabled:text-zinc-300"
        >
          ⟲
        </button>
      </div>

      <div className="pointer-events-none absolute bottom-3 right-3 hidden text-[11px] text-zinc-400 sm:block">
        Drag to pan · ⌘/Ctrl + scroll to zoom
      </div>

      <Legend mode={mode} nodes={nodes} />

      {hoverInfo && (
        <div
          className="pointer-events-none absolute z-10 min-w-[180px] rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(hover!.x + 14, containerW - 200),
            top: Math.max(8, hover!.y - 12),
          }}
        >
          <div className="mb-1 flex items-center gap-1.5 font-semibold text-zinc-900">
            <span>{hoverInfo.geo.flag}</span>
            <span>{hoverInfo.node?.country_name ?? hoverInfo.geo.name}</span>
          </div>
          <TooltipBody mode={mode} iso3={hover!.iso3} node={hoverInfo.node} simulation={simulation} />
        </div>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 text-zinc-500">
      <span>{k}</span>
      <span className="tabular-nums text-zinc-900">{v}</span>
    </div>
  );
}

function TooltipBody({
  mode,
  iso3,
  node,
  simulation,
}: {
  mode: MapMode;
  iso3: string;
  node?: NodeRow;
  simulation?: SimulationOverlay | null;
}) {
  if (mode === 'simulation' && simulation) {
    if (iso3 === simulation.epicenter) return <div className="text-red-700">Shock epicenter</div>;
    const r = simulation.recipients[iso3];
    if (!r) return <div className="text-zinc-500">Outside top recipients</div>;
    return (
      <>
        <Row k="Projected arrivals" v={fmtCompact(r.influx)} />
        <Row k="Capacity strain" v={<span style={{ color: STRAIN_COLORS[r.level] }}>{r.level}</span>} />
      </>
    );
  }
  if (!node) return <div className="text-zinc-500">No data</div>;
  switch (mode) {
    case 'centrality':
      return (
        <>
          <Row k="PageRank" v={fmtNum(node.pagerank, 4)} />
          <Row k="Betweenness" v={fmtNum(node.betweenness_centrality, 4)} />
        </>
      );
    case 'communities':
      return <div className="max-w-[220px] whitespace-normal text-zinc-600">{node.community_label ?? '—'}</div>;
    case 'religion':
      return <Row k="Dominant religion" v={node.dominant_religion ?? '—'} />;
    default:
      return (
        <>
          <Row k="Immigrant stock" v={fmtCompact(node.in_strength)} />
          <Row k="Emigrant stock" v={fmtCompact(node.out_strength)} />
        </>
      );
  }
}

function Legend({ mode, nodes }: { mode: MapMode; nodes: Record<string, NodeRow> }) {
  const communities = useMemo(() => {
    if (mode !== 'communities') return [];
    const seen = new Map<number, string>();
    for (const n of Object.values(nodes)) {
      if (n.louvain_community != null && !seen.has(n.louvain_community)) {
        seen.set(n.louvain_community, (n.community_label ?? '').replace(/^Cluster \d+:\s*/, ''));
      }
    }
    return Array.from(seen).sort((a, b) => a[0] - b[0]);
  }, [mode, nodes]);

  let body: React.ReactNode;
  if (mode === 'centrality') {
    body = (
      <div className="w-40">
        <div className="h-1.5 rounded-full" style={{ background: `linear-gradient(to right, ${sequential(0)}, ${sequential(0.5)}, ${sequential(1)})` }} />
        <div className="mt-1 flex justify-between text-zinc-500">
          <span>Low</span>
          <span>PageRank</span>
          <span>High</span>
        </div>
      </div>
    );
  } else if (mode === 'communities') {
    body = (
      <div className="grid max-w-[340px] grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2">
        {communities.map(([id, label]) => (
          <span key={id} className="flex items-center gap-1.5 truncate text-zinc-600">
            <Dot color={categorical(id)} />
            <span className="truncate">{label || `Cluster ${id}`}</span>
          </span>
        ))}
      </div>
    );
  } else if (mode === 'religion') {
    body = (
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {Object.entries(RELIGION_COLORS).map(([k, c]) => (
          <span key={k} className="flex items-center gap-1.5 text-zinc-600"><Dot color={c} />{k}</span>
        ))}
      </div>
    );
  } else if (mode === 'simulation') {
    body = (
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <span className="flex items-center gap-1.5 text-zinc-600"><Dot color={EPICENTER} />Epicenter</span>
        {(Object.keys(STRAIN_COLORS) as StrainLevel[]).map((k) => (
          <span key={k} className="flex items-center gap-1.5 text-zinc-600"><Dot color={STRAIN_COLORS[k]} />{k} strain</span>
        ))}
      </div>
    );
  } else {
    body = (
      <span className="flex items-center gap-2 text-zinc-600">
        <span className="inline-block h-0.5 w-6 rounded" style={{ backgroundColor: ACCENT }} />
        Corridor · width ∝ migrant stock
      </span>
    );
  }

  return (
    <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-zinc-200 bg-white/95 px-3 py-2 text-[11px] shadow-sm backdrop-blur">
      {body}
    </div>
  );
}
