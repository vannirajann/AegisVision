import './LiveMonitoring.css'

const DETECTION_URL = 'http://127.0.0.1:8000'

const cameras = [
  { id: 1, name: 'Camera 01' },
  { id: 2, name: 'Camera 02' },
  { id: 3, name: 'Camera 03' },
  { id: 4, name: 'Camera 04' },
  { id: 5, name: 'Camera 05' },
  { id: 6, name: 'Camera 06' },
]

export default function LiveMonitoring() {
  return (
    <div className="live-monitoring">

      <div className="live-monitoring-toolbar">
        <span>6 of 6 cameras streaming</span>
      </div>

      <div className="camera-grid">

        {cameras.map((camera) => (
          <div className="camera-tile" key={camera.id}>

            <div className="camera-tile-header">
              <span>{camera.name}</span>
              <span>🟢 LIVE</span>
            </div>

            <div className="camera-video">
              <img
                src={`${DETECTION_URL}/video-feed?camera=${camera.id}`}
                alt={`${camera.name} live camera feed`}
              />
            </div>

            <div className="camera-tile-footer">
              <span>Source: test.mp4</span>
              <span>Detection + Tracking</span>
            </div>

          </div>
        ))}

      </div>

    </div>
  )
}