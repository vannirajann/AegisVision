import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaTone?: 'up' | 'down' | 'flat';
  icon: LucideIcon;
}

const toneClasses: Record<string, string> = {
  up: 'text-green',
  down: 'text-red',
  flat: 'text-ink-faint',
};

export default function StatCard({ label, value, delta, deltaTone = 'flat', icon: Icon }: StatCardProps) {
  return (
    <div className="rounded border border-line bg-white p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-ink-soft">{label}</span>
        <Icon size={16} className="text-ink-faint" strokeWidth={2} />
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="font-display text-2xl font-semibold text-ink">{value}</span>
        {delta && (
          <span className={`text-xs font-medium ${toneClasses[deltaTone]}`}>{delta}</span>
        )}
      </div>
    </div>
  );
}
