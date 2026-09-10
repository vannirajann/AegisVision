import { mockCameras } from '../../data/mockCameras.js'
import CameraTile from './CameraTile.jsx'
import './LiveMonitoring.css'

export default function LiveMonitoring() {
  const onlineCount = mockCameras.filter((c) => c.status === 'online').length

  return (
    <div className="live-monitoring">
      <div className="live-monitoring-toolbar">
        <span>{onlineCount} of {mockCameras.length} cameras streaming</span>
      </div>

      <div className="camera-grid">
        {mockCameras.map((camera) => (
          <CameraTile key={camera.id} camera={camera} />
        ))}
      </div>
    </div>
  )
}
