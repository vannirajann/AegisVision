import { useState } from 'react'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { mockAlerts } from '../../data/mockAlerts.js'
import { formatTimestamp } from '../../utils/formatTime.js'
import './Alerts.css'

const filters = ['All', 'Intrusion', 'ANPR', 'Loitering', 'Night Movement']

export default function Alerts() {
  // Local copy so "Acknowledge" can update the UI immediately.
  // Once the backend exists, this becomes state fetched from
  // getAlerts() and acknowledgeAlert() will send the PATCH/POST.
  const [alerts, setAlerts] = useState(mockAlerts)
  const [activeFilter, setActiveFilter] = useState('All')

  function acknowledge(id) {
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)))
  }

  const visibleAlerts = alerts.filter((a) => activeFilter === 'All' || a.type === activeFilter)

  return (
    <div className="alerts-page">
      <div className="alerts-filters">
        {filters.map((f) => (
          <button
            key={f}
            className={'alerts-filter' + (f === activeFilter ? ' is-active' : '')}
            onClick={() => setActiveFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      {visibleAlerts.length === 0 ? (
        <EmptyState text="No alerts match this filter." />
      ) : (
        <div className="alerts-table">
          <div className="alerts-row alerts-row-head">
            <span>Type</span>
            <span>Camera</span>
            <span>Severity</span>
            <span>Time</span>
            <span>Message</span>
            <span>Status</span>
          </div>
          {visibleAlerts.map((alert) => (
            <div key={alert.id} className={'alerts-row' + (alert.acknowledged ? ' is-acknowledged' : '')}>
              <span className="alerts-cell-type">{alert.type}</span>
              <span>{alert.camera}</span>
              <span><SeverityBadge level={alert.severity} /></span>
              <span className="alerts-cell-time">{formatTimestamp(alert.timestamp)}</span>
              <span className="alerts-cell-message">{alert.message}</span>
              <span>
                {alert.acknowledged ? (
                  <span className="alerts-ack-label">Acknowledged</span>
                ) : (
                  <button className="alerts-ack-btn" onClick={() => acknowledge(alert.id)}>
                    Acknowledge
                  </button>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
