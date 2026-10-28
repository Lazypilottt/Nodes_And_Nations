import type { NextApiRequest, NextApiResponse } from 'next';
import { migrationLong } from '../../lib/server/data';
import { SNAPSHOT_YEARS } from '../../lib/types';

const ISO3 = /^[A-Z]{3}$/;

/** Bilateral stock in both directions for every snapshot year. */
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const a = String(req.query.a ?? '').toUpperCase();
  const b = String(req.query.b ?? '').toUpperCase();
  if (!ISO3.test(a) || !ISO3.test(b)) {
    return res.status(400).json({ error: 'Query parameters a and b must be ISO3 codes' });
  }

  const aToB = new Map<number, number>();
  const bToA = new Map<number, number>();
  for (const m of migrationLong()) {
    if (m.origin === a && m.dest === b) aToB.set(m.year, (aToB.get(m.year) ?? 0) + m.stock);
    else if (m.origin === b && m.dest === a) bToA.set(m.year, (bToA.get(m.year) ?? 0) + m.stock);
  }

  res.setHeader('Cache-Control', 'private, max-age=300');
  return res.status(200).json({
    a,
    b,
    series: SNAPSHOT_YEARS.map((year) => ({ year, a_to_b: aToB.get(year) ?? 0, b_to_a: bToA.get(year) ?? 0 })),
  });
}
