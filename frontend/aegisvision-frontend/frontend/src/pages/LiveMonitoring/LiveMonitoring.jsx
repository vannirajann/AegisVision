import './LiveMonitoring.css'

export default function LiveMonitoring() {
  const videoFeedUrl = 'http://127.0.0.1:8000/video-feed'

  return (
    <div className="live-monitoring">

      <div className="live-monitoring-toolbar">
        <span>4 of 4 cameras streaming</span>

        <span className="live-status">
          <span className="live-dot"></span>
          LIVE
        </span>
      </div>

      <div className="camera-grid">

        <div className="camera-feed">
          <div className="camera-label">
            <span>CAM-01</span>
            <span>LIVE</span>
          </div>

          <img
            src={videoFeedUrl}
            alt="AegisVision four-camera surveillance feed"
          />
        </div>

      </div>

    </div>
  )
}