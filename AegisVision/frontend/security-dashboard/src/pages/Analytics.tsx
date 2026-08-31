import StatCard from '../components/ui/StatCard';
import TrafficChart from '../components/dashboard/TrafficChart';
import ZoneChart from '../components/dashboard/ZoneChart';
import { traffic, zoneBreakdown } from '../data/mockData';
import { Activity, TrendingUp, Clock, Percent } from 'lucide-react';

export default function Analytics() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total events today" value="1,284" delta="+8.2%" deltaTone="up" icon={Activity} />
        <StatCard label="Peak hour" value="18:00" delta="74 events" deltaTone="flat" icon={Clock} />
        <StatCard label="Avg. response time" value="2m 14s" delta="-19s" deltaTone="up" icon={TrendingUp} />
        <StatCard label="False alarm rate" value="4.1%" delta="+0.6%" deltaTone="down" icon={Percent} />
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded border border-line bg-white p-5">
          <h2 className="font-display text-base font-semibold text-ink mb-4">
            Motion &amp; events, last 24 hours
          </h2>
          <TrafficChart data={traffic} />
        </div>
        <div className="rounded border border-line bg-white p-5">
          <h2 className="font-display text-base font-semibold text-ink mb-4">Events by zone</h2>
          <ZoneChart data={zoneBreakdown} />
        </div>
      </div>
    </div>
  );
}
