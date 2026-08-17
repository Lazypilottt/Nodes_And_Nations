import { useEffect, useState } from 'react'

export default function ComparePanel({factors}:{factors:any[]}){
  const [left, setLeft] = useState('')
  const [right, setRight] = useState('')

  const isoList = Array.from(new Set(factors.map((r:any)=> r.iso3))).slice(0,200)

  const getSeries = (iso:string, col:string)=> {
    return factors.filter((r:any)=> r.iso3 === iso).map((r:any)=> ({x: r.year, y: parseFloat(r[col] || '0')})).sort((a:any,b:any)=> a.x - b.x)
  }

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="font-semibold">Compare countries</h3>
      <div className="flex gap-3 mt-2">
        <select value={left} onChange={e=>setLeft(e.target.value)} className="p-2 border rounded">
          <option value="">Select left</option>
          {isoList.map(i=> <option key={i} value={i}>{i}</option>)}
        </select>
        <select value={right} onChange={e=>setRight(e.target.value)} className="p-2 border rounded">
          <option value="">Select right</option>
          {isoList.map(i=> <option key={i} value={i}>{i}</option>)}
        </select>
      </div>
      <p className="text-sm text-slate-500 mt-2">Pick two countries to compare their GDP/population/indices over time (uses loaded factors data).</p>
    </div>
  )
}
