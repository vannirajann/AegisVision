import { useState } from 'react'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { mockEvents } from '../../data/mockEvents.js'
import { formatTimestamp } from '../../utils/formatTime.js'
import './EventHistory.css'

export default function EventHistory() {
  const [query, setQuery] = useState('')

  const filtered = mockEvents.filter((event) => {
    const haystack = `${event.type} ${event.camera}`.toLowerCase()
    return haystack.includes(query.toLowerCase())
  })

  return (
    <div className="event-history">
      <div className="event-history-toolbar">
        <input
          className="event-history-search"
          type="text"
          placeholder="Search by event type or camera name"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="event-history-count">{filtered.length} events</span>
      </div>

      {filtered.length === 0 ? (
        <EmptyState text="No events match your search." />
      ) : (
        <table className="event-table">
          <thead>
            <tr>
              <th>Event</th>
              <th>Camera</th>
              <th>Time</th>
              <th>Severity</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((event) => (
              <tr key={event.id}>
                <td className="event-table-type">{event.type}</td>
                <td>{event.camera}</td>
                <td className="event-table-time">{formatTimestamp(event.timestamp)}</td>
                <td><SeverityBadge level={event.severity} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
