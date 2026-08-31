import { Search, LogOut } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const titles: Record<string, string> = {
  '/overview': 'Overview',
  '/cameras': 'Cameras',
  '/alerts': 'Alerts',
  '/analytics': 'Analytics',
  '/users-devices': 'Users & devices',
};

export default function Topbar() {
  const { pathname } = useLocation();
  const { userName, logout } = useAuth();
  const title = titles[pathname] ?? 'Dashboard';

  return (
    <header className="h-16 shrink-0 border-b border-line bg-white flex items-center justify-between px-6 gap-4">
      <div className="flex items-center gap-3">
        <h1 className="font-display text-lg font-semibold text-ink">{title}</h1>
        <span className="hidden sm:flex items-center gap-1.5 rounded-full bg-green-dim px-2.5 py-1 text-xs font-medium text-green">
          <span className="w-1.5 h-1.5 rounded-full bg-green" />
          All systems normal
        </span>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden sm:flex items-center gap-2 rounded border border-line bg-paper px-3 py-1.5 w-56">
          <Search size={15} className="text-ink-faint" />
          <input
            type="text"
            placeholder="Search cameras, alerts…"
            className="bg-transparent text-sm text-ink placeholder:text-ink-faint focus:outline-none w-full"
          />
        </div>

        <div className="flex items-center gap-2.5 pl-3 border-l border-line">
          <div className="w-8 h-8 rounded-full bg-ink text-paper flex items-center justify-center text-xs font-medium uppercase">
            {userName ? userName.charAt(0) : 'O'}
          </div>
          <span className="hidden md:block text-sm text-ink capitalize">
            {userName || 'Operator'}
          </span>
          <button
            onClick={logout}
            aria-label="Sign out"
            className="text-ink-faint hover:text-ink transition-colors"
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>
    </header>
  );
}
