import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { TrafficPoint } from '../../types';

export default function TrafficChart({ data }: { data: TrafficPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="motionFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2557D6" stopOpacity={0.18} />
            <stop offset="100%" stopColor="#2557D6" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#DADFE1" vertical={false} />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 11, fill: '#8A93A0', fontFamily: 'JetBrains Mono' }}
          axisLine={{ stroke: '#DADFE1' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 11, fill: '#8A93A0', fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            borderRadius: 4,
            border: '1px solid #DADFE1',
            fontSize: 12,
            fontFamily: 'Inter',
          }}
        />
        <Area
          type="monotone"
          dataKey="motion"
          stroke="#2557D6"
          strokeWidth={2}
          fill="url(#motionFill)"
          name="Motion events"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
