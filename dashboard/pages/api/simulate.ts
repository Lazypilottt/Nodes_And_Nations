import type { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';
import { COUNTRY_GEO_MAP, getCountryGeo } from '../../lib/countryGeo';

function haversineDistanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim().replace(/^"|"$/g, ''));
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim().replace(/^"|"$/g, ''));
  return result;
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed, use POST' });
  }

  try {
    const {
      scenario = 'conflict',
      epicenter_iso3 = 'UKR',
      intensity = 80, // 0 to 100
      displaced_scale = 2000000, // Total population displaced
      border_policy = 'universal', // 'universal' | 'kinship_first' | 'strict_border'
    } = req.body || {};

    const exportsDir = path.resolve(process.cwd(), '../data/exports');
    const nodesPath = path.join(exportsDir, 'nodes_master.csv');
    const edgesPath = path.join(exportsDir, 'migration_full_flat.csv');

    if (!fs.existsSync(nodesPath) || !fs.existsSync(edgesPath)) {
      return res.status(404).json({ error: 'Required pipeline exports not found' });
    }

    const epicGeo = getCountryGeo(epicenter_iso3);

    // Read node master for latest indicators (GDP, population, communities)
    const nodeLines = fs.readFileSync(nodesPath, 'utf8').split(/\r?\n/).filter((l) => l.trim().length > 0);
    const nodeHeader = parseCSVLine(nodeLines[0]);
    const latestNodes: Record<string, any> = {};

    for (let i = 1; i < nodeLines.length; i++) {
      const parts = parseCSVLine(nodeLines[i]);
      const obj: any = {};
      for (let j = 0; j < nodeHeader.length; j++) {
        obj[nodeHeader[j]] = parts[j] ?? '';
      }
      if (Number(obj.year) === 2025 || (!latestNodes[obj.iso3] && Number(obj.year) === 2020)) {
        latestNodes[obj.iso3] = {
          iso3: obj.iso3,
          country_name: obj.country_name || getCountryGeo(obj.iso3).name,
          continent: obj.continent || getCountryGeo(obj.iso3).continent,
          un_region: obj.un_region || getCountryGeo(obj.iso3).un_region,
          income_group: obj.income_group || getCountryGeo(obj.iso3).income_group,
          gdp_per_capita: parseFloat(obj.gdp_per_capita) || 3000,
          population: parseFloat(obj.population) || 10000000,
          louvain_community: obj.louvain_community,
          community_label: obj.community_label || 'General Community',
          is_boundary_node: obj.is_boundary_node === 'True' || obj.is_boundary_node === 'true',
        };
      }
    }

    // Read historical bilateral diaspora weights for the epicenter
    const edgeLines = fs.readFileSync(edgesPath, 'utf8').split(/\r?\n/).filter((l) => l.trim().length > 0);
    const edgeHeader = parseCSVLine(edgeLines[0]);
    const origIdx = edgeHeader.indexOf('origin_iso3');
    const destIdx = edgeHeader.indexOf('dest_iso3');
    const yearIdx = edgeHeader.indexOf('year');
    const weightIdx = edgeHeader.indexOf('weight');
    const diasporaWeights: Record<string, number> = {};

    for (let i = 1; i < edgeLines.length; i++) {
      const parts = parseCSVLine(edgeLines[i]);
      const orig = parts[origIdx >= 0 ? origIdx : 0]?.trim();
      const dest = parts[destIdx >= 0 ? destIdx : 1]?.trim();
      const yr = Number(parts[yearIdx >= 0 ? yearIdx : 2]?.trim());
      const w = parseFloat(parts[weightIdx >= 0 ? weightIdx : 3]?.trim()) || 0;
      if (orig === epicenter_iso3 && (yr === 2020 || yr === 2025)) {
        diasporaWeights[dest] = (diasporaWeights[dest] || 0) + w;
      }
    }

    const epicNode = latestNodes[epicenter_iso3] || {
      iso3: epicenter_iso3,
      country_name: epicGeo.name,
      continent: epicGeo.continent,
      un_region: epicGeo.un_region,
      gdp_per_capita: 3500,
      population: 40000000,
      louvain_community: '0',
    };

    // Calculate routing gravity score for every other country in the world
    const candidateDestinations: any[] = [];
    const intensityFactor = Math.max(0.1, intensity / 100);

    for (const iso3 of Object.keys(COUNTRY_GEO_MAP)) {
      if (iso3 === epicenter_iso3) continue;
      const targetGeo = getCountryGeo(iso3);
      const targetNode = latestNodes[iso3] || {
        iso3,
        country_name: targetGeo.name,
        continent: targetGeo.continent,
        un_region: targetGeo.un_region,
        income_group: targetGeo.income_group,
        gdp_per_capita: 5000,
        population: 15000000,
        louvain_community: '0',
      };

      const distKm = Math.max(
        50,
        haversineDistanceKm(epicGeo.lat, epicGeo.lng, targetGeo.lat, targetGeo.lng)
      );

      const existingDiaspora = diasporaWeights[iso3] || 100;
      const isNeighbor = distKm < 1800;
      const isSameContinent = targetGeo.continent === epicGeo.continent;
      const isSameCommunity = targetNode.louvain_community === epicNode.louvain_community;

      // Gravity Model Components
      let gravityScore = 0;

      if (scenario === 'conflict') {
        // War/Conflict Model: Heavy geographic contiguity + historical diaspora pull + income capacity
        const distDecay = Math.pow(distKm, -0.65);
        const diasporaPull = Math.pow(existingDiaspora + 500, 0.42);
        const gdpPull = Math.pow(Math.max(1000, targetNode.gdp_per_capita), 0.28);
        const neighborBonus = isNeighbor ? 2.8 : 1.0;
        const continentBonus = isSameContinent ? 1.4 : 0.8;
        const communityBonus = isSameCommunity ? 1.3 : 1.0;

        gravityScore =
          distDecay *
          diasporaPull *
          gdpPull *
          neighborBonus *
          continentBonus *
          communityBonus *
          intensityFactor;

        // Policy shock modifiers
        if (border_policy === 'kinship_first') {
          gravityScore *= Math.pow(existingDiaspora + 100, 0.2);
        } else if (border_policy === 'strict_border') {
          if (!isNeighbor) gravityScore *= 0.3;
        }
      } else if (scenario === 'climate') {
        // Climate Displacement: Moves towards high climate-resilience / high-income destinations
        const distDecay = Math.pow(distKm, -0.45);
        const gdpPull = Math.pow(Math.max(2000, targetNode.gdp_per_capita), 0.45);
        gravityScore = distDecay * gdpPull * (isSameContinent ? 1.5 : 1.0) * intensityFactor;
      } else if (scenario === 'visa') {
        // Visa Liberalization / Border Free Movement: Sharp increase to high-income & alliance partners
        const diasporaPull = Math.pow(existingDiaspora + 200, 0.5);
        const incomeBonus = targetNode.income_group === 'High income' ? 2.2 : 1.0;
        gravityScore = diasporaPull * incomeBonus * (isSameContinent ? 1.6 : 0.9) * intensityFactor;
      } else {
        // Economic Asymmetry: Gradient between origin and dest GDP per capita
        const gdpRatio = Math.max(0.5, targetNode.gdp_per_capita / Math.max(500, epicNode.gdp_per_capita));
        const distDecay = Math.pow(distKm, -0.5);
        gravityScore = Math.pow(gdpRatio, 0.6) * distDecay * (existingDiaspora + 100);
      }

      candidateDestinations.push({
        iso3,
        country_name: targetGeo.name,
        continent: targetGeo.continent,
        un_region: targetGeo.un_region,
        income_group: targetGeo.income_group,
        lat: targetGeo.lat,
        lng: targetGeo.lng,
        flag: targetGeo.flag,
        dist_km: Math.round(distKm),
        existing_diaspora: Math.round(existingDiaspora),
        gdp_per_capita: targetNode.gdp_per_capita,
        population: targetNode.population,
        is_boundary_node: targetNode.is_boundary_node,
        raw_score: Math.max(0.000001, gravityScore),
      });
    }

    // Normalize probabilities
    const totalScore = candidateDestinations.reduce((acc, c) => acc + c.raw_score, 0);
    let cumulativeDisplaced = 0;

    const projectedDestinations = candidateDestinations.map((c) => {
      const probability = c.raw_score / totalScore;
      const predictedInflux = Math.round(probability * displaced_scale);
      cumulativeDisplaced += predictedInflux;

      // Strain: Influx per 1,000 local citizens
      const strainPer1k = (predictedInflux / Math.max(500000, c.population)) * 1000;
      let strainCategory = 'Low';
      let strainColor = '#10b981'; // emerald
      if (strainPer1k >= 12) {
        strainCategory = 'Critical';
        strainColor = '#ef4444'; // red
      } else if (strainPer1k >= 4) {
        strainCategory = 'High';
        strainColor = '#f97316'; // orange
      } else if (strainPer1k >= 1.2) {
        strainCategory = 'Moderate';
        strainColor = '#f59e0b'; // amber
      }

      return {
        ...c,
        probability,
        predicted_influx: predictedInflux,
        strain_per_1k: parseFloat(strainPer1k.toFixed(2)),
        strain_category: strainCategory,
        strain_color: strainColor,
      };
    });

    // Sort by projected influx descending
    projectedDestinations.sort((a, b) => b.predicted_influx - a.predicted_influx);

    // Continent breakdown for Interactive Donut/Pie Charts
    const continentBreakdown: Record<string, number> = {};
    const regionBreakdown: Record<string, number> = {};
    const incomeBreakdown: Record<string, number> = {};

    for (const dest of projectedDestinations) {
      continentBreakdown[dest.continent] = (continentBreakdown[dest.continent] || 0) + dest.predicted_influx;
      regionBreakdown[dest.un_region] = (regionBreakdown[dest.un_region] || 0) + dest.predicted_influx;
      incomeBreakdown[dest.income_group] = (incomeBreakdown[dest.income_group] || 0) + dest.predicted_influx;
    }

    // Top 15 key recipient corridors for map visualization
    const topCorridors = projectedDestinations.slice(0, 15).map((d) => ({
      origin_iso3: epicenter_iso3,
      origin_name: epicGeo.name,
      origin_lat: epicGeo.lat,
      origin_lng: epicGeo.lng,
      origin_flag: epicGeo.flag,
      dest_iso3: d.iso3,
      dest_name: d.country_name,
      dest_lat: d.lat,
      dest_lng: d.lng,
      dest_flag: d.flag,
      predicted_influx: d.predicted_influx,
      strain_category: d.strain_category,
      strain_color: d.strain_color,
      dist_km: d.dist_km,
    }));

    return res.status(200).json({
      success: true,
      simulation_metadata: {
        scenario,
        epicenter: {
          iso3: epicenter_iso3,
          name: epicGeo.name,
          continent: epicGeo.continent,
          lat: epicGeo.lat,
          lng: epicGeo.lng,
          flag: epicGeo.flag,
          population: epicNode.population,
        },
        intensity,
        displaced_scale,
        border_policy,
        total_modeled_displacement: cumulativeDisplaced,
      },
      top_recipients: projectedDestinations.slice(0, 20),
      top_corridors: topCorridors,
      continent_breakdown: continentBreakdown,
      region_breakdown: regionBreakdown,
      income_breakdown: incomeBreakdown,
    });
  } catch (err: any) {
    console.error('Simulation Error:', err);
    res.status(500).json({ success: false, error: err.message });
  }
}
