import { Link } from 'react-router-dom';
import { Video, BellRing, ShieldAlert, Wifi } from 'lucide-react';
import StatCard from '../components/ui/StatCard';
import CameraTile from '../components/dashboard/CameraTile';
import AlertsPanel from '../components/dashboard/AlertsPanel';
import TrafficChart from '../components/dashboard/TrafficChart';
import { cameras, alerts, traffic, devices } from '../data/mockData';

export default function Overview() {
  const onlineCameras = cameras.filter((c) => c.status === 'online').length;
  const openAlerts = alerts.filter((a) => !a.acknowledged).length;
  const offlineDevices = devices.filter((d) => d.status === 'offline').length;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Cameras online" value={`${onlineCameras}/${cameras.length}`} delta="+1 vs. yesterday" deltaTone="up" icon={Video} />
        <StatCard label="Open alerts" value={String(openAlerts)} delta="2 critical" deltaTone="down" icon={BellRing} />
        <StatCard label="Devices offline" value={String(offlineDevices)} delta="No change" deltaTone="flat" icon={ShieldAlert} />
        <StatCard label="Network uptime" value="99.94%" delta="+0.02%" deltaTone="up" icon={Wifi} />
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded border border-line bg-white p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display text-base font-semibold text-ink">Activity, last 24 hours</h2>
            <span className="text-xs text-ink-faint font-mono">updated 30s ago</span>
          </div>
          <TrafficChart data={traffic} />
        </div>

        <div className="rounded border border-line bg-white p-5">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-display text-base font-semibold text-ink">Recent alerts</h2>
            <Link to="/alerts" className="text-xs text-signal hover:underline underline-offset-2">
              View all
            </Link>
          </div>
          <AlertsPanel alerts={alerts} compact />
        </div>
      </div>

      <div className="rounded border border-line bg-white p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display text-base font-semibold text-ink">Live cameras</h2>
          <Link to="/cameras" className="text-xs text-signal hover:underline underline-offset-2">
            View all cameras
          </Link>
        </div>
        <div className="grid sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {cameras.slice(0, 4).map((camera) => (
            <CameraTile key={camera.id} camera={camera} />
          ))}
        </div>
      </div>
    </div>
  );
}
