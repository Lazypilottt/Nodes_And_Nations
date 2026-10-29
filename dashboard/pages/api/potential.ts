import type { NextApiRequest, NextApiResponse } from 'next';
import { readDataCSV } from '../../lib/csv';

export interface PotentialRow {
  origin_iso3: string;
  dest_iso3: string;
  migrant_stock: number;
  expected: number;
  ratio: number;
  log_ratio: number;
  gap: number;
  status: 'under' | 'over' | 'as_expected';
  dist_km: number;
  contig: number;
  comlang_ethno: number;
  colony: number;
}

export function potentialRows(): PotentialRow[] {
  return readDataCSV('exports/corridor_potential.csv') as unknown as PotentialRow[];
}

/**
 * GET /api/potential?iso3=IND&role=origin|dest|any&limit=15
 *
 * Corridors much smaller ("under") or larger ("over") than a gravity model
 * predicts (notebooks/09_corridor_potential.py), ranked by the absolute gap in
 * people, plus a scatter sample of large corridors.
 */
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  const iso3 = typeof req.query.iso3 === 'string' ? req.query.iso3.toUpperCase() : null;
  const role = req.query.role === 'origin' || req.query.role === 'dest' ? req.query.role : 'any';
  const limit = Math.min(50, Math.max(1, Number(req.query.limit) || 15));

  let rows = potentialRows();
  if (rows.length === 0) return res.status(404).json({ error: 'Run notebooks/09_corridor_potential.py first' });
  if (iso3) {
    rows = rows.filter(
      (r) => (role !== 'dest' && r.origin_iso3 === iso3) || (role !== 'origin' && r.dest_iso3 === iso3)
    );
  }

  const under = rows.filter((r) => r.status === 'under').sort((a, b) => a.gap - b.gap).slice(0, limit);
  const over = rows.filter((r) => r.status === 'over').sort((a, b) => b.gap - a.gap).slice(0, limit);
  const scatter = rows
    .filter((r) => Math.max(r.migrant_stock, r.expected) >= (iso3 ? 1_000 : 25_000))
    .map(({ origin_iso3, dest_iso3, migrant_stock, expected, status }) => ({ origin_iso3, dest_iso3, migrant_stock, expected, status }));

  res.setHeader('Cache-Control', 'public, max-age=300');
  return res.status(200).json({
    iso3,
    counts: {
      under: rows.filter((r) => r.status === 'under').length,
      over: rows.filter((r) => r.status === 'over').length,
      total: rows.length,
    },
    under,
    over,
    scatter,
  });
}
