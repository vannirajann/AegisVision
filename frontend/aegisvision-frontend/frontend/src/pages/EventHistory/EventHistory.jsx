import { useEffect, useState } from 'react'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { formatTimestamp } from '../../utils/formatTime.js'
import './EventHistory.css'

const BACKEND_URL = 'http://127.0.0.1:8001'

export default function EventHistory() {
  const [events, setEvents] = useState([])
  const [query, setQuery] = useState('')

  const loadEvents = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/events`)
      const data = await response.json()
      setEvents(data.events || [])
    } catch (error) {
      console.error('Failed to load events:', error)
    }
  }

  useEffect(() => {
    loadEvents()

    const interval = setInterval(loadEvents, 5000)

    return () => clearInterval(interval)
  }, [])

  const filtered = events.filter((event) => {
    const haystack =
      `${event.event_type} ${event.source}`.toLowerCase()

    return haystack.includes(query.toLowerCase())
  })

  return (
    <div className="event-history">
      <div className="event-history-toolbar">
        <input
          className="event-history-search"
          type="text"
          placeholder="Search by event type or source"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />

        <span className="event-history-count">
          {filtered.length} events
        </span>
      </div>

      {filtered.length === 0 ? (
        <EmptyState text="No events found." />
      ) : (
        <table className="event-table">
          <thead>
            <tr>
              <th>Event</th>
              <th>Source</th>
              <th>Time</th>
              <th>Severity</th>
            </tr>
          </thead>

          <tbody>
            {filtered.map((event) => (
              <tr key={event.id}>
                <td className="event-table-type">
                  {event.event_type}
                </td>

                <td>{event.source}</td>

                <td className="event-table-time">
                  {formatTimestamp(event.timestamp)}
                </td>

                <td>
                  <SeverityBadge level={event.severity} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}