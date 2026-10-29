import { animate, motion, useInView } from 'framer-motion';
import React, { useEffect, useId, useRef, useState } from 'react';

const EASE = [0.22, 1, 0.36, 1] as const;

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
  bodyClassName = 'p-5',
  delay = 0,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  delay?: number;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: EASE, delay }}
      className={`rounded-xl border border-zinc-200 bg-white shadow-[0_1px_2px_rgb(0_0_0/0.03)] transition-shadow hover:shadow-[0_4px_24px_-8px_rgb(0_0_0/0.10)] dark:shadow-none dark:hover:shadow-[0_0_0_1px_rgb(96_165_250/0.12)] ${className}`}
    >
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-zinc-100 px-5 py-4">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-zinc-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </motion.section>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  size = 'sm',
  ariaLabel,
}: {
  options: readonly { value: NoInfer<T>; label: React.ReactNode }[];
  value: T;
  onChange: (v: NoInfer<T>) => void;
  size?: 'xs' | 'sm';
  ariaLabel?: string;
}) {
  const pad = size === 'xs' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';
  const id = useId();
  return (
    <div role="tablist" aria-label={ariaLabel} className="inline-flex flex-wrap rounded-lg bg-zinc-100 p-0.5">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={String(o.value)}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(o.value)}
            className={`${pad} relative rounded-md font-medium transition-colors ${
              active ? 'text-zinc-900' : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            {active && (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 rounded-md bg-white shadow-sm dark:bg-zinc-300/30"
                transition={{ type: 'spring', stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Tweens from the previous value to the new one; renders the final value with no motion preference. */
export function CountUp({ value, format }: { value: number | null | undefined; format: (n: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const prev = useRef<number>(0);
  const inView = useInView(ref, { once: true });
  const [display, setDisplay] = useState<number | null>(value ?? null);

  useEffect(() => {
    if (value == null || !Number.isFinite(value)) {
      setDisplay(null);
      return;
    }
    if (!inView) return;
    const controls = animate(prev.current, value, {
      duration: 0.9,
      ease: EASE,
      onUpdate: (v) => setDisplay(v),
    });
    prev.current = value;
    return () => controls.stop();
  }, [value, inView]);

  return <span ref={ref}>{display == null ? '—' : format(display)}</span>;
}

export function Stat({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  hint?: React.ReactNode;
  tone?: 'default' | 'accent' | 'danger';
}) {
  const color =
    tone === 'accent' ? 'text-blue-600 dark:text-blue-400' : tone === 'danger' ? 'text-red-600 dark:text-red-400' : 'text-zinc-900';
  return (
    <div className="min-w-0">
      <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`mt-1 truncate text-lg font-semibold tabular-nums ${color}`}>{value}</div>
      {hint && <div className="mt-0.5 truncate text-xs text-zinc-500">{hint}</div>}
    </div>
  );
}

export function Badge({ children, color }: { children: React.ReactNode; color?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium"
      style={color ? { backgroundColor: `${color}1f`, color } : undefined}
    >
      {!color ? <span className="text-zinc-600">{children}</span> : children}
    </span>
  );
}

export function Dot({ color, size = 8 }: { color: string; size?: number }) {
  return (
    <span className="inline-block shrink-0 rounded-full" style={{ backgroundColor: color, width: size, height: size }} />
  );
}

export function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-center justify-between text-xs font-medium text-zinc-600">
        <span>{label}</span>
        {hint && <span className="tabular-nums text-zinc-900">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

export const selectClass =
  'w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20';

/** Horizontal share bars: a lighter, more legible alternative to pie charts. */
export function ShareBars({
  data,
  colorFor,
  format,
  limit,
}: {
  data: { label: string; value: number }[];
  colorFor: (label: string, index: number) => string;
  format: (v: number) => string;
  limit?: number;
}) {
  const total = data.reduce((acc, d) => acc + d.value, 0) || 1;
  const rows = limit ? data.slice(0, limit) : data;
  const max = Math.max(...rows.map((d) => d.value), 1);
  return (
    <ul className="space-y-2.5">
      {rows.map((d, i) => (
        <li key={d.label}>
          <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
            <span className="flex min-w-0 items-center gap-2 text-zinc-700">
              <Dot color={colorFor(d.label, i)} />
              <span className="truncate">{d.label}</span>
            </span>
            <span className="shrink-0 tabular-nums text-zinc-500">
              {format(d.value)} <span className="ml-1 font-medium text-zinc-900">{((d.value / total) * 100).toFixed(1)}%</span>
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-zinc-100">
            <motion.div
              className="h-1.5 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(d.value / max) * 100}%` }}
              transition={{ duration: 0.7, ease: EASE, delay: i * 0.04 }}
              style={{ backgroundColor: colorFor(d.label, i) }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="py-10 text-center text-sm text-zinc-500">{children}</div>;
}

/** Loading placeholder with a moving sheen. */
export function Skeleton({ className = '' }: { className?: string }) {
  return (
    <div className={`relative overflow-hidden rounded-xl bg-zinc-100 ${className}`}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/40 to-transparent dark:via-white/5" />
    </div>
  );
}

export function Table({ children }: { children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">{children}</table>
    </div>
  );
}

export const th = 'whitespace-nowrap px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500';
export const td = 'whitespace-nowrap px-3 py-2.5 text-zinc-700';
