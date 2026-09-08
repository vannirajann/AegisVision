import { useEffect, useState } from 'react'
import './Analytics.css'

const BACKEND_URL = 'http://127.0.0.1:8001'

function countBy(items, key) {
  const counts = {}

  items.forEach((item) => {
    const value = item[key]
    if (value) {
      counts[value] = (counts[value] || 0) + 1
    }
  })

  return Object.entries(counts).sort((a, b) => b[1] - a[1])
}

export default function Analytics() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)

  const loadEvents = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/events`)
      const data = await response.json()

      setEvents(data.events || [])
      setLoading(false)
    } catch (error) {
      console.error('Could not load analytics events:', error)
      setLoading(false)
    }
  }

  useEffect(() => {
    loadEvents()

    const interval = setInterval(loadEvents, 5000)

    return () => clearInterval(interval)
  }, [])

  const alerts = events.filter(
    (event) => event.severity === 'high'
  )

  const alertsByType = countBy(alerts, 'event_type')
  const eventsBySeverity = countBy(events, 'severity')
  const eventsBySource = countBy(events, 'source')

  const maxAlertCount =
    alertsByType.length > 0
      ? Math.max(...alertsByType.map(([, count]) => count))
      : 1

  const maxSourceCount =
    eventsBySource.length > 0
      ? Math.max(...eventsBySource.map(([, count]) => count))
      : 1

  const totalSeverity = eventsBySeverity.reduce(
    (sum, [, count]) => sum + count,
    0
  )

  return (
    <div className="analytics-page">

      <div className="analytics-panel">
        <div className="analytics-panel-title">
          Alerts by Type
        </div>

        {loading ? (
          <p>Loading real events...</p>
        ) : alertsByType.length === 0 ? (
          <p>No high-severity alerts yet.</p>
        ) : (
          alertsByType.map(([type, count]) => (
            <div className="analytics-bar-row" key={type}>
              <span>{type}</span>

              <div className="analytics-bar">
                <div
                  className="analytics-bar-fill"
                  style={{
                    width: `${(count / maxAlertCount) * 100}%`
                  }}
                />
              </div>

              <strong>{count}</strong>
            </div>
          ))
        )}
      </div>


      <div className="analytics-panel">
        <div className="analytics-panel-title">
          Events by Severity
        </div>

        {eventsBySeverity.map(([severity, count]) => {
          const percentage =
            totalSeverity > 0
              ? Math.round((count / totalSeverity) * 100)
              : 0

          return (
            <div className="analytics-severity-row" key={severity}>
              <span>● {severity}</span>
              <strong>
                {count} ({percentage}%)
              </strong>
            </div>
          )
        })}
      </div>


      <div className="analytics-panel analytics-panel-wide">
        <div className="analytics-panel-title">
          Events by Source
        </div>

        {eventsBySource.length === 0 ? (
          <p>No events available.</p>
        ) : (
          eventsBySource.slice(0, 5).map(([source, count]) => (
            <div className="analytics-bar-row" key={source}>
              <span>{source}</span>

              <div className="analytics-bar">
                <div
                  className="analytics-bar-fill"
                  style={{
                    width: `${(count / maxSourceCount) * 100}%`
                  }}
                />
              </div>

              <strong>{count}</strong>
            </div>
          ))
        )}
      </div>


      <div className="analytics-real-data-message">
        Showing real events from AegisVision backend and SQLite database.
        Updates automatically every 5 seconds.
      </div>

    </div>
  )
}