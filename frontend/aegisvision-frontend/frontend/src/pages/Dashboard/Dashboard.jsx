import StatCard from '../../components/common/StatCard.jsx'
import Panel from '../../components/common/Panel.jsx'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { mockCameras } from '../../data/mockCameras.js'
import { mockAlerts } from '../../data/mockAlerts.js'
import { mockEvents } from '../../data/mockEvents.js'
import { timeAgo, formatTimestamp } from '../../utils/formatTime.js'
import './Dashboard.css'

export default function Dashboard() {
  const totalCameras = mockCameras.length
  const activeCameras = mockCameras.filter((c) => c.status === 'online').length
  const activeAlerts = mockAlerts.filter((a) => !a.acknowledged).length
  const highSeverityAlerts = mockAlerts.filter((a) => !a.acknowledged && a.severity === 'high').length
  const recentEvents = mockEvents.slice(0, 6)
  const activeAlertList = mockAlerts.filter((a) => !a.acknowledged).slice(0, 5)

  return (
    <div className="dashboard">
      <div className="dashboard-stats">
        <StatCard label="Total Cameras" value={totalCameras} tone="neutral" hint={`${totalCameras} registered`} />
        <StatCard label="Active Cameras" value={activeCameras} tone="online" hint={`${totalCameras - activeCameras} offline`} />
        <StatCard label="Recent Events" value={mockEvents.length} tone="info" hint="Last 24 hours" />
        <StatCard label="Active Alerts" value={activeAlerts} tone="medium" hint="Awaiting acknowledgement" />
        <StatCard label="High-Severity Alerts" value={highSeverityAlerts} tone="high" hint="Needs immediate review" />
      </div>

      <div className="dashboard-grid">
        <Panel title="Recent Activity" className="dashboard-activity">
          {recentEvents.length === 0 ? (
            <EmptyState text="No activity recorded yet." />
          ) : (
            <ul className="activity-list">
              {recentEvents.map((event) => (
                <li key={event.id} className={`activity-item sev-${event.severity}`}>
                  <div className="activity-item-main">
                    <span className="activity-item-type">{event.type}</span>
                    <span className="activity-item-camera">{event.camera}</span>
                  </div>
                  <span className="activity-item-time">{timeAgo(event.timestamp)}</span>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Active Alerts" className="dashboard-alerts">
          {activeAlertList.length === 0 ? (
            <EmptyState text="No unacknowledged alerts right now." />
          ) : (
            <ul className="alert-list">
              {activeAlertList.map((alert) => (
                <li key={alert.id} className="alert-list-item">
                  <div className="alert-list-top">
                    <span className="alert-list-type">{alert.type}</span>
                    <SeverityBadge level={alert.severity} />
                  </div>
                  <p className="alert-list-message">{alert.message}</p>
                  <div className="alert-list-meta">
                    <span>{alert.camera}</span>
                    <span>{formatTimestamp(alert.timestamp)}</span>
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
