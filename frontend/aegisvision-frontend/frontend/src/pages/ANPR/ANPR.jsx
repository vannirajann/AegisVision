import { mockAnpr } from '../../data/mockAnpr.js'
import { formatTimestamp } from '../../utils/formatTime.js'
import './ANPR.css'

const statusLabels = {
  allowed: 'Allow-listed',
  blocked: 'Blocked',
  unknown: 'Unrecognized',
}

export default function ANPR() {
  return (
    <div className="anpr-page">
      <table className="anpr-table">
        <thead>
          <tr>
            <th>Plate Number</th>
            <th>Vehicle</th>
            <th>Camera</th>
            <th>Time</th>
            <th>Confidence</th>
            <th>List Status</th>
          </tr>
        </thead>
        <tbody>
          {mockAnpr.map((entry) => (
            <tr key={entry.id}>
              <td className="anpr-plate">{entry.plate}</td>
              <td>{entry.vehicleType}</td>
              <td>{entry.camera}</td>
              <td className="anpr-time">{formatTimestamp(entry.timestamp)}</td>
              <td className="anpr-confidence">{Math.round(entry.confidence * 100)}%</td>
              <td>
                <span className={`anpr-status anpr-status-${entry.listStatus}`}>
                  {statusLabels[entry.listStatus]}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
