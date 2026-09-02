import './SeverityBadge.css'

const labels = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
}

export default function SeverityBadge({ level }) {
  return <span className={`severity-badge severity-${level}`}>{labels[level] || level}</span>
}
