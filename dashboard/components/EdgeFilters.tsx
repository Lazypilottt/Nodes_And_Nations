import { useState } from 'react'

export default function EdgeFilters({onFilter}:{onFilter: (opts:any)=>void}){
  const [q, setQ] = useState('')
  const [year, setYear] = useState('')

  return (
    <div className="bg-white p-4 rounded-lg shadow flex gap-3 items-end">
      <div className="flex-1">
        <label className="block text-sm text-slate-600">Search (origin/dest)</label>
        <input value={q} onChange={e=>setQ(e.target.value)} className="mt-1 w-full p-2 border rounded" placeholder="Type ISO3 or country name" />
      </div>
      <div>
        <label className="block text-sm text-slate-600">Year</label>
        <input value={year} onChange={e=>setYear(e.target.value)} className="mt-1 p-2 border rounded w-28" placeholder="e.g. 2019" />
      </div>
      <div>
        <button onClick={()=>onFilter({q, year})} className="px-4 py-2 bg-blue-600 text-white rounded">Apply</button>
      </div>
    </div>
  )
}
