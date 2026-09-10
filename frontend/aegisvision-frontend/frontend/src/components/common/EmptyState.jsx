import './EmptyState.css'

export default function EmptyState({ text = 'Nothing to show yet.' }) {
  return <div className="empty-state">{text}</div>
}
