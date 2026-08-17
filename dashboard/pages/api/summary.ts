import type { NextApiRequest, NextApiResponse } from 'next'
import fs from 'fs'
import path from 'path'

export default function handler(req: NextApiRequest, res: NextApiResponse){
  const exportsDir = path.resolve(process.cwd(), '../data/exports')
  if(!fs.existsSync(exportsDir)) return res.status(404).json({error: 'exports dir not found'})
  const files = fs.readdirSync(exportsDir).map(f => {
    const st = fs.statSync(path.join(exportsDir, f))
    return {name: f, size: st.size}
  })
  res.status(200).json({files})
}
