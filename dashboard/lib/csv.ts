import fs from 'fs';
import path from 'path';

export type Row = Record<string, string | number | boolean | null>;

export const DATA_ROOT = path.resolve(process.cwd(), '..', 'data');

function parseLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (ch === ',' && !quoted) {
      out.push(cur);
      cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function coerce(raw: string): string | number | boolean | null {
  const v = raw.trim();
  if (v === '') return null;
  if (v === 'True' || v === 'true') return true;
  if (v === 'False' || v === 'false') return false;
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}

export function parseCSV(text: string): Row[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return [];
  const header = parseLine(lines[0]).map((h) => h.trim());
  const rows: Row[] = new Array(lines.length - 1);
  for (let i = 1; i < lines.length; i++) {
    const cells = parseLine(lines[i]);
    const row: Row = {};
    for (let j = 0; j < header.length; j++) row[header[j]] = coerce(cells[j] ?? '');
    rows[i - 1] = row;
  }
  return rows;
}

const cache = new Map<string, { mtimeMs: number; rows: Row[] }>();

/**
 * Read a CSV under data/ and memoise it until the file changes on disk, so
 * API routes don't re-parse multi-megabyte exports on every request.
 */
export function readDataCSV(relPath: string): Row[] {
  const file = path.join(DATA_ROOT, relPath);
  if (!fs.existsSync(file)) return [];
  const { mtimeMs } = fs.statSync(file);
  const hit = cache.get(file);
  if (hit && hit.mtimeMs === mtimeMs) return hit.rows;
  const rows = parseCSV(fs.readFileSync(file, 'utf8'));
  cache.set(file, { mtimeMs, rows });
  return rows;
}
