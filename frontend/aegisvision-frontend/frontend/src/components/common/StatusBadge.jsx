import './StatusBadge.css'

export default function StatusBadge({ status }) {
  // status: 'online' | 'offline'
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-badge-dot" />
      {status === 'online' ? 'Online' : 'Offline'}
    </span>
  )
}
