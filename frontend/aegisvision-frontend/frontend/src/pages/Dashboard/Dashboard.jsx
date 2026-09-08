import { useEffect, useState } from 'react'
import StatCard from '../../components/common/StatCard.jsx'
import Panel from '../../components/common/Panel.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import { formatTimestamp, timeAgo } from '../../utils/formatTime.js'
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
    (event) =>
      event.severity === 'high' &&
      !event.acknowledged
  )

  const highSeverityAlerts = activeAlerts.length

  const recentEvents = events.slice(-6).reverse()

  const getEventLabel = (event) => {
    switch (event.event_type) {
      case 'anpr':
        return 'ANPR Detection'

      case 'face_detected':
        return 'Face Detected'

      case 'night_movement':
        return 'Night Movement'

      case 'loitering':
        return 'Loitering'

      case 'suspicious_activity':
        return 'Suspicious Activity'

      case 'intrusion':
        return 'Intrusion'

      default:
        return event.event_type
    }
  }

  const getEventSource = (event) => {
    if (event.event_type === 'anpr') {
      return event.data?.plate_number
        ? `Plate: ${event.data.plate_number}`
        : 'ANPR'
    }

    if (event.source === 'analytics') {
      return 'Analytics'
    }

    return event.source || 'Unknown'
  }

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

        <Panel
          title="Recent Activity"
          className="dashboard-activity"
        >

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
                      {getEventLabel(event)}
                    </span>

                    <span className="activity-item-camera">
                      {getEventSource(event)}
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

        <Panel
          title="Active Alerts"
          className="dashboard-alerts"
        >

          {activeAlerts.length === 0 ? (

            <EmptyState
              text="No unacknowledged alerts right now."
            />

          ) : (

            <ul className="alert-list">

              {activeAlerts
                .slice(-5)
                .reverse()
                .map((event) => (

                  <li
                    key={event.id}
                    className="alert-list-item"
                  >

                    <div className="alert-list-top">

                      <span className="alert-list-type">
                        {getEventLabel(event)}
                      </span>

                      <SeverityBadge
                        level={event.severity}
                      />

                    </div>

                    <p className="alert-list-message">

                      {event.event_type === 'intrusion'
                        ? `${event.data?.object || 'Object'} detected crossing the restricted zone.`

                        : event.event_type === 'anpr'
                        ? `Vehicle plate detected: ${event.data?.plate_number || 'Unknown'}`

                        : event.event_type === 'face_detected'
                        ? `Face detected by the analytics system.`

                        : event.event_type === 'night_movement'
                        ? `Movement detected during low-light conditions.`

                        : event.event_type === 'loitering'
                        ? `Person remained in the area for too long.`

                        : event.event_type === 'suspicious_activity'
                        ? `Suspicious activity detected.`

                        : 'Security event detected.'
                      }

                    </p>

                    <div className="alert-list-meta">

                      <span>
                        {event.source || 'Unknown'}
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