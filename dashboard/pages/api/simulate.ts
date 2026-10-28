import type { NextApiRequest, NextApiResponse } from 'next';
import { COUNTRY_GEO_MAP, getCountryGeo } from '../../lib/countryGeo';
import { migrationLong, nodes } from '../../lib/server/data';
import { mipexTier, type StrainLevel } from '../../lib/theme';
import type { NodeRow } from '../../lib/types';

export const SCENARIOS = ['conflict', 'climate', 'cultural_kinship', 'visa', 'economic'] as const;
export const POLICIES = ['universal', 'kinship_first', 'cultural_kinship', 'integration_havens', 'strict_border'] as const;
type Scenario = (typeof SCENARIOS)[number];
type Policy = (typeof POLICIES)[number];

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const rad = Math.PI / 180;
  const dLat = (lat2 - lat1) * rad;
  const dLon = (lon2 - lon1) * rad;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function median(values: number[]): number {
  const s = values.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
}

function strainLevel(perThousand: number): StrainLevel {
  if (perThousand >= 12) return 'Critical';
  if (perThousand >= 4) return 'High';
  if (perThousand >= 1.2) return 'Moderate';
  return 'Low';
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed, use POST' });
  }

  const body = req.body ?? {};
  const scenario = (body.scenario ?? 'conflict') as Scenario;
  const policy = (body.border_policy ?? 'universal') as Policy;
  const epicenter = String(body.epicenter_iso3 ?? 'UKR').toUpperCase();
  const intensity = Number(body.intensity ?? 80);
  const displaced = Number(body.displaced_scale ?? 2_000_000);

  if (!SCENARIOS.includes(scenario)) return res.status(400).json({ error: `Unknown scenario '${scenario}'` });
  if (!POLICIES.includes(policy)) return res.status(400).json({ error: `Unknown border_policy '${policy}'` });
  if (!COUNTRY_GEO_MAP[epicenter]) return res.status(400).json({ error: `Unknown epicenter '${epicenter}'` });
  if (!Number.isFinite(intensity) || intensity < 10 || intensity > 100) {
    return res.status(400).json({ error: 'intensity must be between 10 and 100' });
  }
  if (!Number.isFinite(displaced) || displaced < 10_000 || displaced > 20_000_000) {
    return res.status(400).json({ error: 'displaced_scale must be between 10,000 and 20,000,000' });
  }

  try {
    const allNodes = nodes();
    if (allNodes.length === 0) return res.status(404).json({ error: 'Pipeline exports not found' });

    const latestYear = Math.max(...allNodes.map((n) => n.year));
    const latest = new Map<string, NodeRow>();
    for (const n of allNodes) if (n.year === latestYear) latest.set(n.iso3, n);

    const medGdp = median(Array.from(latest.values(), (n) => n.gdp_per_capita ?? NaN));
    const medPop = median(Array.from(latest.values(), (n) => n.population ?? NaN));

    // Existing diaspora of the epicenter in every destination (latest snapshot)
    const diaspora = new Map<string, number>();
    for (const m of migrationLong()) {
      if (m.origin === epicenter && m.year === latestYear) diaspora.set(m.dest, (diaspora.get(m.dest) ?? 0) + m.stock);
    }

    const epiGeo = getCountryGeo(epicenter);
    const epi = latest.get(epicenter);
    const epiGdp = epi?.gdp_per_capita ?? medGdp;
    const epiReligion = epi?.dominant_religion ?? null;
    const epiOpenness = epi?.cultural_openness_index ?? 0.5;

    // Severity steepens distance decay: acute shocks push people to the nearest
    // safe state; slower-burning crises allow longer journeys.
    const s = intensity / 100;
    const decayScale = 0.6 + 0.8 * s;

    const candidates = Object.values(COUNTRY_GEO_MAP)
      .filter((g) => g.iso3 !== epicenter)
      .map((g) => {
        const n = latest.get(g.iso3);
        const gdp = n?.gdp_per_capita ?? medGdp;
        const pop = n?.population ?? medPop;
        const integration = n?.integration_policy_index ?? 0.5;
        const openness = n?.cultural_openness_index ?? 0.5;
        const distKm = Math.max(50, haversineKm(epiGeo.lat, epiGeo.lng, g.lat, g.lng));
        const existing = diaspora.get(g.iso3) ?? 0;
        const isNeighbor = distKm < 1800;
        const sameContinent = g.continent === epiGeo.continent;
        const sameCommunity = n?.louvain_community != null && n.louvain_community === epi?.louvain_community;
        const sameReligion = !!epiReligion && n?.dominant_religion === epiReligion;
        const decay = (k: number) => Math.pow(distKm, -k * decayScale);

        let score: number;
        switch (scenario) {
          case 'conflict':
            score =
              decay(0.65) *
              Math.pow(existing + 500, 0.42) *
              Math.pow(Math.max(1000, gdp), 0.28) *
              (isNeighbor ? 1 + 1.8 * s : 1) *
              (sameContinent ? 1.4 : 0.8) *
              (sameCommunity ? 1.3 : 1) *
              (sameReligion ? 1.25 : 0.95) *
              Math.pow(Math.max(0.1, integration), 0.35);
            break;
          case 'climate':
            score =
              decay(0.45) *
              Math.pow(Math.max(2000, gdp), 0.45) *
              Math.pow(Math.max(0.1, integration), 0.4) *
              (sameContinent ? 1.5 : 1);
            break;
          case 'cultural_kinship':
            score =
              Math.pow(existing + 200, 0.45) *
              (sameReligion ? 3 : 0.4) *
              Math.pow(Math.max(0.1, openness), 0.6) *
              decay(0.5);
            break;
          case 'visa':
            score =
              Math.pow(existing + 200, 0.5) *
              (g.income_group === 'High income' ? 2.2 : 1) *
              Math.pow(Math.max(0.1, integration), 0.5) *
              (sameContinent ? 1.6 : 0.9) *
              decay(0.2);
            break;
          default: // economic
            score =
              Math.pow(Math.max(0.5, gdp / Math.max(500, epiGdp)), 0.6) *
              decay(0.5) *
              Math.pow(existing + 100, 0.5);
        }

        switch (policy) {
          case 'kinship_first':
            score *= Math.pow(existing + 100, 0.2) * (sameReligion ? 1.3 : 1);
            break;
          case 'cultural_kinship':
            score *= (sameReligion ? 2 : 0.45) * Math.pow(Math.max(0.1, openness), 0.5);
            break;
          case 'integration_havens':
            score *= Math.pow(Math.max(0.1, integration), 1.2);
            break;
          case 'strict_border':
            if (!isNeighbor) score *= 0.3;
            break;
        }

        return {
          iso3: g.iso3,
          country_name: g.name,
          continent: g.continent,
          un_region: g.un_region,
          income_group: n?.income_group ?? g.income_group,
          dominant_religion: n?.dominant_religion ?? null,
          integration_policy_index: n?.integration_policy_index ?? null,
          lat: g.lat,
          lng: g.lng,
          dist_km: Math.round(distKm),
          existing_diaspora: Math.round(existing),
          population: pop,
          score: Math.max(1e-12, score),
        };
      });

    const total = candidates.reduce((acc, c) => acc + c.score, 0);
    const recipients = candidates
      .map(({ score, ...c }) => {
        const influx = (score / total) * displaced;
        const strainPer1k = (influx / Math.max(500_000, c.population)) * 1000;
        return {
          ...c,
          share: score / total,
          predicted_influx: Math.round(influx),
          strain_per_1k: Number(strainPer1k.toFixed(2)),
          strain_level: strainLevel(strainPer1k),
        };
      })
      .sort((a, b) => b.predicted_influx - a.predicted_influx);

    const breakdown = (key: (r: (typeof recipients)[number]) => string | null) => {
      const acc = new Map<string, number>();
      for (const r of recipients) {
        const k = key(r) ?? 'Unknown';
        acc.set(k, (acc.get(k) ?? 0) + r.predicted_influx);
      }
      return Array.from(acc, ([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);
    };

    const top = recipients.slice(0, 20);
    const weightedDistance = recipients.reduce((acc, r) => acc + r.dist_km * r.share, 0);

    return res.status(200).json({
      params: { scenario, epicenter_iso3: epicenter, intensity, displaced_scale: displaced, border_policy: policy },
      epicenter: {
        iso3: epicenter,
        name: epiGeo.name,
        flag: epiGeo.flag,
        population: epi?.population ?? null,
        dominant_religion: epiReligion,
      },
      reference_year: latestYear,
      summary: {
        total_displaced: displaced,
        mean_distance_km: Math.round(weightedDistance),
        top5_share: recipients.slice(0, 5).reduce((acc, r) => acc + r.share, 0),
        critical_states: recipients.filter((r) => r.strain_level === 'Critical').length,
      },
      top_recipients: top,
      breakdowns: {
        continent: breakdown((r) => r.continent),
        income: breakdown((r) => r.income_group),
        religion: breakdown((r) => r.dominant_religion),
        mipex: breakdown((r) => mipexTier(r.integration_policy_index)),
      },
    });
  } catch (err: any) {
    console.error('POST /api/simulate failed:', err);
    return res.status(500).json({ error: err?.message ?? 'Internal error' });
  }
}
