import type { NextApiRequest, NextApiResponse } from 'next'
import fs from 'fs'
import path from 'path'

function parseCSV(text: string){
  const lines = text.split('\n').filter(Boolean)
  if(lines.length === 0) return {cols: [], rows: []}
  const header = lines[0].split(',').map(h=>h.trim())
  const rows = lines.slice(1).map(l => {
    // naive CSV split — good enough for these exports
    const cells = l.split(',')
    const obj: any = {}
    for(let i=0;i<header.length;i++) obj[header[i]] = cells[i] !== undefined ? cells[i].trim() : null
    return obj
  })
  return {cols: header, rows}
}

export default function handler(req: NextApiRequest, res: NextApiResponse){
  const csvPath = path.resolve(process.cwd(), '../data/exports/network_edges.csv')
  if(!fs.existsSync(csvPath)){
    return res.status(404).json({error: 'network_edges.csv not found; run pipeline first'})
  }
  const txt = fs.readFileSync(csvPath, 'utf8')
  const parsed = parseCSV(txt)
  res.status(200).json(parsed)
}
