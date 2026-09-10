import './StatCard.css'

/**
 * A single metric tile, e.g. "Active Cameras: 14".
 * `tone` colors the accent bar + value to match what the number means.
 */
export default function StatCard({ label, value, tone = 'neutral', hint }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <span className="stat-card-bar" />
      <div className="stat-card-body">
        <div className="stat-card-label">{label}</div>
        <div className="stat-card-value">{value}</div>
        {hint && <div className="stat-card-hint">{hint}</div>}
      </div>
    </div>
  )
}
