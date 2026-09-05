import './LiveMonitoring.css'

const DETECTION_URL = 'http://127.0.0.1:8000'

export default function LiveMonitoring() {
  return (
    <div className="live-monitoring">

      <div className="live-monitoring-toolbar">
        <span>1 of 1 cameras streaming</span>
      </div>

      <div className="camera-grid">

        <div className="camera-tile">
          <div className="camera-tile-header">
            <span>Camera 01</span>
            <span>🟢 LIVE</span>
          </div>

          <div className="camera-video">
            <img
              src={`${DETECTION_URL}/video-feed`}
              alt="AegisVision live camera feed"
            />
          </div>

          <div className="camera-tile-footer">
            <span>Source: test.mp4</span>
            <span>Detection + Tracking</span>
          </div>
        </div>

      </div>

    </div>
  )
}