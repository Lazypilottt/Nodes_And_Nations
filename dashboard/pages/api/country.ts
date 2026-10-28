import type { NextApiRequest, NextApiResponse } from 'next';
import { migrationLong } from '../../lib/server/data';

const ISO3 = /^[A-Z]{3}$/;

/** Largest inbound and outbound corridors for one country in one snapshot year. */
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const iso3 = String(req.query.iso3 ?? '').toUpperCase();
  const year = Number(req.query.year);
  const limit = Math.min(50, Math.max(1, Number(req.query.limit ?? 10)));
  if (!ISO3.test(iso3) || !Number.isInteger(year)) {
    return res.status(400).json({ error: 'iso3 (ISO3 code) and year are required' });
  }

  const inbound = new Map<string, number>();
  const outbound = new Map<string, number>();
  let totalIn = 0;
  let totalOut = 0;
  for (const m of migrationLong()) {
    if (m.year !== year) continue;
    if (m.dest === iso3) {
      inbound.set(m.origin, (inbound.get(m.origin) ?? 0) + m.stock);
      totalIn += m.stock;
    } else if (m.origin === iso3) {
      outbound.set(m.dest, (outbound.get(m.dest) ?? 0) + m.stock);
      totalOut += m.stock;
    }
  }

  const top = (acc: Map<string, number>) =>
    Array.from(acc, ([iso, stock]) => ({ iso3: iso, stock }))
      .sort((a, b) => b.stock - a.stock)
      .slice(0, limit);

  res.setHeader('Cache-Control', 'private, max-age=300');
  return res.status(200).json({
    iso3,
    year,
    total_inbound: totalIn,
    total_outbound: totalOut,
    inbound: top(inbound),
    outbound: top(outbound),
  });
}
