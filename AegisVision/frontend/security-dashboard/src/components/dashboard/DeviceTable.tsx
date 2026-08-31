import type { Device } from '../../types';
import StatusBadge from '../ui/StatusBadge';

const kindLabel: Record<Device['kind'], string> = {
  camera: 'Camera',
  sensor: 'Sensor',
  'access-panel': 'Access panel',
  nvr: 'Recorder',
};

export default function DeviceTable({ devices }: { devices: Device[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-ink-faint border-b border-line">
            <th className="font-medium py-2.5 pr-4">Device</th>
            <th className="font-medium py-2.5 pr-4">Type</th>
            <th className="font-medium py-2.5 pr-4">Zone</th>
            <th className="font-medium py-2.5 pr-4">IP address</th>
            <th className="font-medium py-2.5 pr-4">Last seen</th>
            <th className="font-medium py-2.5">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {devices.map((d) => (
            <tr key={d.id} className="hover:bg-paper-dim/60 transition-colors">
              <td className="py-3 pr-4">
                <p className="text-ink font-medium">{d.name}</p>
                <p className="text-xs text-ink-faint font-mono">{d.id}</p>
              </td>
              <td className="py-3 pr-4 text-ink-soft">{kindLabel[d.kind]}</td>
              <td className="py-3 pr-4 text-ink-soft">{d.zone}</td>
              <td className="py-3 pr-4 text-ink-soft font-mono text-xs">{d.ip}</td>
              <td className="py-3 pr-4 text-ink-soft">{d.lastSeen}</td>
              <td className="py-3"><StatusBadge status={d.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
