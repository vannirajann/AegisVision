import { useEffect, useState } from 'react'
import { formatTimestamp } from '../../utils/formatTime.js'
import './ANPR.css'

const BACKEND_URL = 'http://127.0.0.1:8001'

export default function ANPR() {
  const [plates, setPlates] = useState([])

  const loadANPR = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/events`)
      const data = await response.json()

      const anprEvents = (data.events || [])
        .filter((event) => {
          if (event.event_type !== 'anpr') {
            return false
          }

          const plate = event.data?.plate_number

          if (typeof plate !== 'string') {
            return false
          }

          if (plate.includes("[{'text'")) {
            return false
          }

          return plate.trim().length > 0
        })
        .sort(
          (a, b) =>
            new Date(b.timestamp) - new Date(a.timestamp)
        )

      setPlates(anprEvents)
    } catch (error) {
      console.error('Failed to load ANPR events:', error)
    }
  }

  useEffect(() => {
    loadANPR()

    const interval = setInterval(loadANPR, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="anpr-page">

      <div className="anpr-toolbar">
        <span>{plates.length} ANPR detections</span>
      </div>

      {plates.length === 0 ? (
        <div className="anpr-empty">
          No ANPR detections found.
        </div>
      ) : (
        <table className="anpr-table">

          <thead>
            <tr>
              <th>Plate Number</th>
              <th>Source</th>
              <th>Time</th>
              <th>Confidence</th>
              <th>Status</th>
            </tr>
          </thead>

          <tbody>
            {plates.map((event) => {

              const data = event.data || {}

              const plateNumber =
                data.plate_number || 'Unknown'

              const confidence =
                typeof data.confidence === 'number'
                  ? Math.round(data.confidence * 100)
                  : 0

              const status =
                data.plate_valid_format
                  ? 'Recognized'
                  : 'Detected'

              return (
                <tr key={event.id}>

                  <td className="anpr-plate">
                    {plateNumber}
                  </td>

                  <td>
                    {event.source || 'anpr'}
                  </td>

                  <td className="anpr-time">
                    {formatTimestamp(event.timestamp)}
                  </td>

                  <td className="anpr-confidence">
                    {confidence}%
                  </td>

                  <td>
                    <span
                      className={`anpr-status ${
                        data.plate_valid_format
                          ? 'anpr-status-allowed'
                          : 'anpr-status-unknown'
                      }`}
                    >
                      {status}
                    </span>
                  </td>

                </tr>
              )
            })}
          </tbody>

        </table>
      )}

    </div>
  )
}