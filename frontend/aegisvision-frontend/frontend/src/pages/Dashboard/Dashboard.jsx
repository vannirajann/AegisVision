import { useEffect, useState } from 'react'
import StatCard from '../../components/common/StatCard.jsx'
import Panel from '../../components/common/Panel.jsx'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { timeAgo, formatTimestamp } from '../../utils/formatTime.js'
import './Dashboard.css'

const BACKEND_URL = 'http://127.0.0.1:8001'

export default function Dashboard() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadEvents = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/events`)
        const data = await response.json()

        setEvents(data.events || [])
      } catch (error) {
        console.error('Could not connect to backend:', error)
      } finally {
        setLoading(false)
      }
    }

    loadEvents()

    const interval = setInterval(loadEvents, 5000)

    return () => clearInterval(interval)
  }, [])

  const totalCameras = 1
  const activeCameras = 1

  const activeAlerts = events.filter(
    (event) => event.severity === 'high' && !event.acknowledged
  )

  const highSeverityAlerts = activeAlerts.length

  const recentEvents = events.slice(-6).reverse()

  return (
    <div className="dashboard">

      <div className="dashboard-stats">

        <StatCard
          label="Total Cameras"
          value={totalCameras}
          tone="neutral"
          hint="1 registered"
        />

        <StatCard
          label="Active Cameras"
          value={activeCameras}
          tone="online"
          hint="Online"
        />

        <StatCard
          label="Recent Events"
          value={events.length}
          tone="info"
          hint="From live backend"
        />

        <StatCard
          label="Active Alerts"
          value={activeAlerts.length}
          tone="medium"
          hint="Awaiting acknowledgement"
        />

        <StatCard
          label="High-Severity Alerts"
          value={highSeverityAlerts}
          tone="high"
          hint="Needs immediate review"
        />

      </div>

      <div className="dashboard-grid">

        <Panel title="Recent Activity" className="dashboard-activity">

          {loading ? (
            <EmptyState text="Loading events..." />
          ) : recentEvents.length === 0 ? (
            <EmptyState text="No activity recorded yet." />
          ) : (
            <ul className="activity-list">

              {recentEvents.map((event) => (

                <li
                  key={event.id}
                  className={`activity-item sev-${event.severity}`}
                >

                  <div className="activity-item-main">

                    <span className="activity-item-type">
                      {event.event_type}
                    </span>

                    <span className="activity-item-camera">
                      {event.source}
                    </span>

                  </div>

                  <span className="activity-item-time">
                    {timeAgo(event.timestamp)}
                  </span>

                </li>

              ))}

            </ul>
          )}

        </Panel>

        <Panel title="Active Alerts" className="dashboard-alerts">

          {activeAlerts.length === 0 ? (
            <EmptyState text="No unacknowledged alerts right now." />
          ) : (

            <ul className="alert-list">

              {activeAlerts.slice(-5).reverse().map((event) => (

                <li
                  key={event.id}
                  className="alert-list-item"
                >

                  <div className="alert-list-top">

                    <span className="alert-list-type">
                      {event.event_type}
                    </span>

                    <SeverityBadge level={event.severity} />

                  </div>

                  <p className="alert-list-message">
                    {event.data?.object
                      ? `${event.data.object} detected crossing the restricted zone.`
                      : 'Security event detected.'}
                  </p>

                  <div className="alert-list-meta">

                    <span>
                      {event.source}
                    </span>

                    <span>
                      {formatTimestamp(event.timestamp)}
                    </span>

                  </div>

                </li>

              ))}

            </ul>

          )}

        </Panel>

      </div>

    </div>
  )
}