import { useState } from 'react';
import { AlertTriangle, AlertOctagon, Info, Check } from 'lucide-react';
import type { AlertItem, AlertSeverity } from '../../types';

const severityConfig: Record<AlertSeverity, { icon: typeof AlertTriangle; text: string; bg: string }> = {
  critical: { icon: AlertOctagon, text: 'text-red', bg: 'bg-red-dim' },
  warning: { icon: AlertTriangle, text: 'text-amber', bg: 'bg-amber-dim' },
  info: { icon: Info, text: 'text-signal', bg: 'bg-signal-dim' },
};

export default function AlertsPanel({ alerts, compact = false }: { alerts: AlertItem[]; compact?: boolean }) {
  const [acked, setAcked] = useState<Set<string>>(
    new Set(alerts.filter((a) => a.acknowledged).map((a) => a.id))
  );

  function toggle(id: string) {
    setAcked((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const list = compact ? alerts.slice(0, 4) : alerts;

  return (
    <ul className="divide-y divide-line">
      {list.map((alert) => {
        const { icon: Icon, text, bg } = severityConfig[alert.severity];
        const isAcked = acked.has(alert.id);
        return (
          <li key={alert.id} className="py-3.5 flex items-start gap-3">
            <span className={`shrink-0 mt-0.5 w-7 h-7 rounded flex items-center justify-center ${bg}`}>
              <Icon size={14} className={text} strokeWidth={2} />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-medium text-ink truncate">{alert.title}</p>
                <span className="shrink-0 text-xs font-mono text-ink-faint">{alert.time}</span>
              </div>
              <p className="text-xs text-ink-soft mt-0.5 leading-relaxed">{alert.detail}</p>
              <p className="text-xs text-ink-faint mt-1">{alert.zone} · {alert.id}</p>
            </div>
            <button
              onClick={() => toggle(alert.id)}
              aria-pressed={isAcked}
              className={`shrink-0 mt-0.5 flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium border transition-colors ${
                isAcked
                  ? 'border-line text-ink-faint bg-paper'
                  : 'border-ink/15 text-ink hover:bg-paper-dim'
              }`}
            >
              <Check size={12} />
              {isAcked ? 'Acked' : 'Ack'}
            </button>
          </li>
        );
      })}
    </ul>
  );
}
