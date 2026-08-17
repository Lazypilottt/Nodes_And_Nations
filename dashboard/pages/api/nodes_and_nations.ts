import type { NextApiRequest, NextApiResponse } from 'next';
import fs from 'fs';
import path from 'path';
import { COUNTRY_GEO_MAP, getCountryGeo } from '../../lib/countryGeo';

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

function parseCSV(text: string): { cols: string[]; rows: any[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return { cols: [], rows: [] };
  const header = parseCSVLine(lines[0]);
  const rows = lines.slice(1).map((l) => {
    const cells = parseCSVLine(l);
    const obj: any = {};
    for (let i = 0; i < header.length; i++) {
      let val = cells[i] !== undefined && cells[i] !== '' ? cells[i] : null;
      if (val !== null) {
        const num = Number(val);
        if (!isNaN(num) && !isNaN(parseFloat(val))) {
          obj[header[i]] = num;
          continue;
        }
        if (val.toLowerCase() === 'true') {
          obj[header[i]] = true;
          continue;
        }
        if (val.toLowerCase() === 'false') {
          obj[header[i]] = false;
          continue;
        }
      }
      obj[header[i]] = val;
    }
    return obj;
  });
  return { cols: header, rows };
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    const exportsDir = path.resolve(process.cwd(), '../data/exports');
    const processedDir = path.resolve(process.cwd(), '../data/processed');

    const readCSVFile = (dir: string, file: string) => {
      const p = path.join(dir, file);
      if (!fs.existsSync(p)) return [];
      const txt = fs.readFileSync(p, 'utf8');
      return parseCSV(txt).rows;
    };

    const nodesMaster = readCSVFile(exportsDir, 'nodes_master.csv');
    const edgesFlat = readCSVFile(exportsDir, 'migration_full_flat.csv');
    const networkSummary = readCSVFile(exportsDir, 'network_summary.csv');
    const summaryStats = readCSVFile(exportsDir, 'summary_stats.csv');
    const modularityScores = readCSVFile(exportsDir, 'modularity_scores.csv');
    const boundaryNodes = readCSVFile(exportsDir, 'boundary_nodes.csv');
    const regressionCoeffs = readCSVFile(exportsDir, 'regression_coefficients.csv');
    const regressionComparison = readCSVFile(exportsDir, 'regression_model_comparison.csv');

    // Enrich edges with lat/lng coordinates for origin and destination
    const enrichedEdges = edgesFlat.map((e) => {
      const oGeo = getCountryGeo(e.origin_iso3);
      const dGeo = getCountryGeo(e.dest_iso3);
      return {
        ...e,
        origin_lat: oGeo.lat,
        origin_lng: oGeo.lng,
        origin_flag: oGeo.flag,
        dest_lat: dGeo.lat,
        dest_lng: dGeo.lng,
        dest_flag: dGeo.flag,
      };
    });

    // Unique countries from metadata
    const countryMetaList = Object.values(COUNTRY_GEO_MAP);

    res.status(200).json({
      success: true,
      country_geo_map: COUNTRY_GEO_MAP,
      country_list: countryMetaList,
      nodes_master: nodesMaster,
      edges: enrichedEdges,
      network_summary: networkSummary,
      summary_stats: summaryStats,
      modularity_scores: modularityScores,
      boundary_nodes: boundaryNodes,
      regression_coefficients: regressionCoeffs,
      regression_comparison: regressionComparison,
    });
  } catch (err: any) {
    console.error('API Error /api/nodes_and_nations:', err);
    res.status(500).json({ success: false, error: err.message });
  }
}
