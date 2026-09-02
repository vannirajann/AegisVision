import Panel from '../../components/common/Panel.jsx'
import { mockAlerts } from '../../data/mockAlerts.js'
import { mockEvents } from '../../data/mockEvents.js'
import './Analytics.css'

function countBy(list, key) {
  const counts = {}
  for (const item of list) {
    counts[item[key]] = (counts[item[key]] || 0) + 1
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1])
}

const severityColor = {
  low: 'var(--severity-low)',
  medium: 'var(--severity-medium)',
  high: 'var(--severity-high)',
}

export default function Analytics() {
  const alertsByType = countBy(mockAlerts, 'type')
  const eventsBySeverity = countBy(mockEvents, 'severity')
  const eventsByCamera = countBy(mockEvents, 'camera').slice(0, 5)

  const maxAlertCount = Math.max(...alertsByType.map(([, count]) => count))
  const maxCameraCount = Math.max(...eventsByCamera.map(([, count]) => count))
  const totalSeverity = eventsBySeverity.reduce((sum, [, count]) => sum + count, 0)

  return (
    <div className="analytics-page">
      <div className="analytics-grid">
        <Panel title="Alerts by Type">
          <div className="bar-list">
            {alertsByType.map(([type, count]) => (
              <div className="bar-row" key={type}>
                <span className="bar-row-label">{type}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${(count / maxAlertCount) * 100}%` }} />
                </div>
                <span className="bar-row-value">{count}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Events by Severity">
          <div className="severity-donut-wrap">
            <div className="severity-legend">
              {eventsBySeverity.map(([severity, count]) => (
                <div className="severity-legend-row" key={severity}>
                  <span className="severity-legend-dot" style={{ background: severityColor[severity] }} />
                  <span className="severity-legend-label">{severity}</span>
                  <span className="severity-legend-value">
                    {count} <span className="severity-legend-pct">({Math.round((count / totalSeverity) * 100)}%)</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="Most Active Cameras" className="analytics-span-2">
          <div className="bar-list">
            {eventsByCamera.map(([camera, count]) => (
              <div className="bar-row" key={camera}>
                <span className="bar-row-label">{camera}</span>
                <div className="bar-track">
                  <div className="bar-fill bar-fill-online" style={{ width: `${(count / maxCameraCount) * 100}%` }} />
                </div>
                <span className="bar-row-value">{count}</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <p className="analytics-note">
        Charts use the mock event and alert data for now. Once Person 5's backend is
        connected, these will run on real detection history instead.
      </p>
    </div>
  )
}
