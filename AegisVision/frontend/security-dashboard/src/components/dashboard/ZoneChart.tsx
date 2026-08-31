import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const colors = ['#2557D6', '#2E7A50', '#B4711F', '#AE372B', '#5B6472'];

export default function ZoneChart({ data }: { data: { zone: string; value: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="#DADFE1" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: '#8A93A0' }} axisLine={false} tickLine={false} />
        <YAxis
          dataKey="zone"
          type="category"
          width={80}
          tick={{ fontSize: 12, fill: '#12181F' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: '#EDEFEA' }}
          contentStyle={{ borderRadius: 4, border: '1px solid #DADFE1', fontSize: 12 }}
        />
        <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={16}>
          {data.map((_, i) => (
            <Cell key={i} fill={colors[i % colors.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
