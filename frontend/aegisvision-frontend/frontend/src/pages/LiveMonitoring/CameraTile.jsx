import StatusBadge from '../../components/common/StatusBadge.jsx'

export default function CameraTile({ camera }) {
  const isOnline = camera.status === 'online'

  return (
    <div className="camera-tile">
      <div className="camera-tile-feed">
        {isOnline ? (
          <>
            <span className="camera-tile-scanline" />
            <span className="camera-tile-feed-label">No live feed connected yet</span>
          </>
        ) : (
          <span className="camera-tile-feed-label muted">Camera offline</span>
        )}
      </div>

      <div className="camera-tile-info">
        <div className="camera-tile-heading">
          <span className="camera-tile-name">{camera.name}</span>
          <StatusBadge status={camera.status} />
        </div>
        <div className="camera-tile-location">
          <span>{camera.location}</span>
          <span className="camera-tile-resolution">{camera.resolution}</span>
        </div>

        <div className="camera-tile-detections">
          <span>{isOnline ? camera.detections.people : '—'} people</span>
          <span>{isOnline ? camera.detections.vehicles : '—'} vehicles</span>
        </div>
      </div>
    </div>
  )
}
