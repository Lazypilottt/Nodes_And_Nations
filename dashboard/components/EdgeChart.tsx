import { Line } from 'react-chartjs-2'
import { Chart, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js'
Chart.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend)

export default function EdgeChart({data}:{data:{labels:string[], datasets:{label:string, data:number[], borderColor?:string}[]}}){
  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <Line data={data} options={{responsive:true, maintainAspectRatio: false}} height={300} />
    </div>
  )
}
