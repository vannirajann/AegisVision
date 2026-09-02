import { NavLink } from 'react-router-dom'
import './Sidebar.css'

// Small inline icon set so we don't need an extra dependency yet.
// Each icon is a plain 20x20 stroke SVG that inherits currentColor.
const icons = {
  dashboard: (
    <svg viewBox="0 0 20 20" fill="none">
      <rect x="2.5" y="2.5" width="6.5" height="6.5" rx="1.2" stroke="currentColor" strokeWidth="1.5" />
      <rect x="11" y="2.5" width="6.5" height="9.5" rx="1.2" stroke="currentColor" strokeWidth="1.5" />
      <rect x="2.5" y="11.5" width="6.5" height="6" rx="1.2" stroke="currentColor" strokeWidth="1.5" />
      <rect x="11" y="14.5" width="6.5" height="3" rx="1.2" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  ),
  camera: (
    <svg viewBox="0 0 20 20" fill="none">
      <path d="M2.5 6.8A1.8 1.8 0 0 1 4.3 5h2l.9-1.4a1.2 1.2 0 0 1 1-.6h3.6a1.2 1.2 0 0 1 1 .6L13.7 5h2A1.8 1.8 0 0 1 17.5 6.8v6.9a1.8 1.8 0 0 1-1.8 1.8H4.3a1.8 1.8 0 0 1-1.8-1.8V6.8Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <circle cx="10" cy="10.2" r="3" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  ),
  bell: (
    <svg viewBox="0 0 20 20" fill="none">
      <path d="M5 8a5 5 0 0 1 10 0c0 3.2 1 4.4 1.5 5H3.5C4 12.4 5 11.2 5 8Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8 15.5a2 2 0 0 0 4 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  history: (
    <svg viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10.5" r="7" stroke="currentColor" strokeWidth="1.5" />
      <path d="M10 6.5v4l2.8 1.6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M6.5 2.6 4 4.6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  plate: (
    <svg viewBox="0 0 20 20" fill="none">
      <rect x="2.5" y="5.5" width="15" height="9" rx="1.6" stroke="currentColor" strokeWidth="1.5" />
      <path d="M5.5 10.2h2.4M9.5 10.2h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  analytics: (
    <svg viewBox="0 0 20 20" fill="none">
      <path d="M3 17V8.5M8.3 17V3M13.7 17v-6M19 17H1" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  shield: (
    <svg viewBox="0 0 20 20" fill="none">
      <path d="M10 2.3 16.5 5v4.4c0 4.3-2.7 7.2-6.5 8.3-3.8-1.1-6.5-4-6.5-8.3V5L10 2.3Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  ),
}

const navItems = [
  { to: '/', label: 'Dashboard', icon: 'dashboard', end: true },
  { to: '/live-monitoring', label: 'Live Monitoring', icon: 'camera' },
  { to: '/alerts', label: 'Alerts', icon: 'bell' },
  { to: '/event-history', label: 'Event History', icon: 'history' },
  { to: '/anpr', label: 'ANPR', icon: 'plate' },
  { to: '/analytics', label: 'Analytics', icon: 'analytics' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-brand-icon">{icons.shield}</span>
        <div>
          <div className="sidebar-brand-name">AegisVision</div>
          <div className="sidebar-brand-tag">Surveillance Console</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => 'sidebar-link' + (isActive ? ' is-active' : '')}
          >
            <span className="sidebar-link-icon">{icons[item.icon]}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="sidebar-status-dot" />
        System monitoring active
      </div>
    </aside>
  )
}
