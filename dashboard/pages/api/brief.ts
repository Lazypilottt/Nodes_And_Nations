import type { NextApiRequest, NextApiResponse } from 'next';
import { COUNTRY_GEO_MAP } from '../../lib/countryGeo';
import { readDataCSV } from '../../lib/csv';
import { migrationLong, nodes } from '../../lib/server/data';
import { potentialRows } from './potential';

/**
 * GET /api/brief?iso3=IND
 *
 * Everything the country brief needs that is not already in /api/dataset:
 * top corridors in the latest year, diaspora gaps, where people would go if a
 * crisis displaced them from this country, and which current displacement
 * crises are most likely to reach it.
 */
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const iso3 = String(req.query.iso3 ?? '').toUpperCase();
  if (!COUNTRY_GEO_MAP[iso3]) return res.status(400).json({ error: `Unknown country '${iso3}'` });

  const all = nodes();
  const latestYear = Math.max(...all.map((n) => n.year));
  const latest = new Map(all.filter((n) => n.year === latestYear).map((n) => [n.iso3, n]));

  const inbound = new Map<string, number>();
  const outbound = new Map<string, number>();
  for (const m of migrationLong()) {
    if (m.year !== latestYear) continue;
    if (m.dest === iso3) inbound.set(m.origin, (inbound.get(m.origin) ?? 0) + m.stock);
    if (m.origin === iso3) outbound.set(m.dest, (outbound.get(m.dest) ?? 0) + m.stock);
  }
  const top = (m: Map<string, number>) =>
    Array.from(m, ([k, v]) => ({ iso3: k, stock: v })).sort((a, b) => b.stock - a.stock).slice(0, 8);

  const potential = potentialRows();
  const gaps = (role: 'origin' | 'dest', status: 'under' | 'over') =>
    potential
      .filter((r) => (role === 'origin' ? r.origin_iso3 : r.dest_iso3) === iso3 && r.status === status)
      .sort((a, b) => (status === 'under' ? a.gap - b.gap : b.gap - a.gap))
      .slice(0, 5);

  const shares = readDataCSV('exports/displacement_shares.csv');
  const ifDisplaced = shares
    .filter((r) => r.origin_iso3 === iso3)
    .slice(0, 8)
    .map((r) => ({ iso3: String(r.dest_iso3), share: Number(r.share) }));

  // Current displacement from each origin, scaled by this country's forecast share
  const exposure = shares
    .filter((r) => r.dest_iso3 === iso3)
    .map((r) => {
      const o = latest.get(String(r.origin_iso3));
      const abroad = o?.refugees_abroad_per_1k != null && o.population ? (o.refugees_abroad_per_1k * o.population) / 1000 : 0;
      return { iso3: String(r.origin_iso3), share: Number(r.share), displaced_abroad: abroad, expected: abroad * Number(r.share) };
    })
    .filter((r) => r.displaced_abroad >= 10_000 && r.share >= 0.01)
    .sort((a, b) => b.expected - a.expected)
    .slice(0, 6);

  res.setHeader('Cache-Control', 'public, max-age=300');
  return res.status(200).json({
    iso3,
    year: latestYear,
    inbound: top(inbound),
    outbound: top(outbound),
    gaps: {
      diaspora_under: gaps('origin', 'under'),
      diaspora_over: gaps('origin', 'over'),
      immigrant_under: gaps('dest', 'under'),
      immigrant_over: gaps('dest', 'over'),
    },
    if_displaced: ifDisplaced,
    exposure,
  });
}
