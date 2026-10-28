import type { NextApiRequest, NextApiResponse } from 'next';
import { buildDataset } from '../../lib/server/data';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).json({ error: 'Method not allowed' });
  }
  try {
    const dataset = buildDataset();
    if (dataset.nodes.length === 0) {
      return res.status(404).json({ error: 'Pipeline exports not found. Run scripts/build_pipeline.py first.' });
    }
    res.setHeader('Cache-Control', 'private, max-age=60');
    return res.status(200).json(dataset);
  } catch (err: any) {
    console.error('GET /api/dataset failed:', err);
    return res.status(500).json({ error: err?.message ?? 'Internal error' });
  }
}
