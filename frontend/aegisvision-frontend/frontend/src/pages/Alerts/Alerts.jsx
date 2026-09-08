import { useEffect, useState } from 'react'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { formatTimestamp } from '../../utils/formatTime.js'
import './Alerts.css'

const BACKEND_URL = 'http://127.0.0.1:8001'

export default function Alerts() {
  const [alerts, setAlerts] = useState([])

  const loadAlerts = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/events`)
      const data = await response.json()

      const highSeverityAlerts = (data.events || []).filter(
        (event) =>
          event.severity === 'high' &&
          event.acknowledged === false
      )

      setAlerts(highSeverityAlerts)
    } catch (error) {
      console.error('Failed to load alerts:', error)
    }
  }

  useEffect(() => {
    loadAlerts()

    const interval = setInterval(loadAlerts, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="alerts-page">
      <div className="alerts-toolbar">
        <span>{alerts.length} active alerts</span>
      </div>

      {alerts.length === 0 ? (
        <EmptyState text="No active alerts." />
      ) : (
        <table className="event-table">
          <thead>
            <tr>
              <th>Alert</th>
              <th>Source</th>
              <th>Time</th>
              <th>Severity</th>
            </tr>
          </thead>

          <tbody>
            {alerts.map((alert) => (
              <tr key={alert.id}>
                <td className="event-table-type">
                  {alert.event_type}
                </td>

                <td>{alert.source}</td>

                <td className="event-table-time">
                  {formatTimestamp(alert.timestamp)}
                </td>

                <td>
                  <SeverityBadge level={alert.severity} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}