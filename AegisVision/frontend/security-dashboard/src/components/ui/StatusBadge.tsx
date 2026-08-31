interface StatusBadgeProps {
  status: string;
}

const map: Record<string, { bg: string; text: string; dot: string; label: string }> = {
  online: { bg: 'bg-green-dim', text: 'text-green', dot: 'bg-green', label: 'Online' },
  active: { bg: 'bg-green-dim', text: 'text-green', dot: 'bg-green', label: 'Active' },
  offline: { bg: 'bg-red-dim', text: 'text-red', dot: 'bg-red', label: 'Offline' },
  suspended: { bg: 'bg-red-dim', text: 'text-red', dot: 'bg-red', label: 'Suspended' },
  warning: { bg: 'bg-amber-dim', text: 'text-amber', dot: 'bg-amber', label: 'Warning' },
  maintenance: { bg: 'bg-amber-dim', text: 'text-amber', dot: 'bg-amber', label: 'Maintenance' },
};

export default function StatusBadge({ status }: StatusBadgeProps) {
  const style = map[status] ?? map.offline;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${style.bg} ${style.text}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  );
}
