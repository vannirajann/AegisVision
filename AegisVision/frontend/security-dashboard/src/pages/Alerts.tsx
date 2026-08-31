import { useMemo, useState } from 'react';
import AlertsPanel from '../components/dashboard/AlertsPanel';
import { alerts } from '../data/mockData';
import type { AlertSeverity } from '../types';

const filters: { label: string; value: AlertSeverity | 'all' }[] = [
  { label: 'All', value: 'all' },
  { label: 'Critical', value: 'critical' },
  { label: 'Warning', value: 'warning' },
  { label: 'Info', value: 'info' },
];

export default function Alerts() {
  const [filter, setFilter] = useState<AlertSeverity | 'all'>('all');
  const filtered = useMemo(
    () => (filter === 'all' ? alerts : alerts.filter((a) => a.severity === filter)),
    [filter]
  );

  return (
    <div className="rounded border border-line bg-white p-5">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h2 className="font-display text-base font-semibold text-ink">
          Alert history
        </h2>
        <div className="flex items-center gap-1.5">
          {filters.map((f) => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
                filter === f.value
                  ? 'bg-ink text-paper'
                  : 'bg-paper border border-line text-ink-soft hover:text-ink'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>
      <AlertsPanel alerts={filtered} />
    </div>
  );
}
