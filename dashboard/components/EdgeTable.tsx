type Edge = {
  origin_iso3: string,
  dest_iso3: string,
  year: string | number,
  weight: string | number,
}

export default function EdgeTable({rows}:{rows: Edge[]}){
  return (
    <div className="overflow-x-auto bg-white p-4 rounded-lg shadow">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="p-2">Origin</th>
            <th className="p-2">Destination</th>
            <th className="p-2">Year</th>
            <th className="p-2">Weight</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0,200).map((r,i)=> (
            <tr key={i} className={i%2? 'bg-slate-50':'bg-white'}>
              <td className="p-2 font-medium">{r.origin_iso3}</td>
              <td className="p-2">{r.dest_iso3}</td>
              <td className="p-2">{r.year}</td>
              <td className="p-2">{r.weight}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-slate-500 mt-2">Showing up to 200 rows</p>
    </div>
  )
}
