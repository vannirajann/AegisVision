import { NavLink } from 'react-router-dom';
import {
  ShieldCheck,
  LayoutGrid,
  Video,
  BellRing,
  BarChart3,
  Users,
} from 'lucide-react';

const navItems = [
  { to: '/overview', label: 'Overview', icon: LayoutGrid },
  { to: '/cameras', label: 'Cameras', icon: Video },
  { to: '/alerts', label: 'Alerts', icon: BellRing },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/users-devices', label: 'Users & devices', icon: Users },
];

export default function Sidebar() {
  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-line bg-white">
      <div className="h-16 flex items-center gap-2.5 px-6 border-b border-line">
        <ShieldCheck size={20} strokeWidth={2} className="text-ink" />
        <span className="font-display font-semibold text-[15px] tracking-tight text-ink">
          Perimeter
        </span>
      </div>

      <nav className="flex-1 px-3 py-5 space-y-0.5">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded px-3 py-2 text-sm transition-colors ${
                isActive
                  ? 'bg-signal-dim text-signal font-medium'
                  : 'text-ink-soft hover:bg-paper-dim hover:text-ink'
              }`
            }
          >
            <Icon size={17} strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-6 py-4 border-t border-line">
        <p className="text-[11px] font-mono text-ink-faint">v1.0 · 6 sites</p>
      </div>
    </aside>
  );
}
