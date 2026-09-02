// Placeholder event log. Alerts are events too, but this list also
// includes lower-signal events that never became an alert.
export const mockEvents = [
  { id: 'EV-3311', type: 'Person Detected', camera: 'Main Gate', timestamp: '2026-09-01T06:42:10', severity: 'high' },
  { id: 'EV-3310', type: 'Vehicle Detected', camera: 'Parking Lot A', timestamp: '2026-09-01T06:35:52', severity: 'low' },
  { id: 'EV-3309', type: 'Plate Recognized', camera: 'Main Gate', timestamp: '2026-09-01T06:20:45', severity: 'medium' },
  { id: 'EV-3308', type: 'Loitering', camera: 'Visitor Lobby', timestamp: '2026-09-01T05:58:02', severity: 'low' },
  { id: 'EV-3307', type: 'Camera Reconnected', camera: 'East Perimeter', timestamp: '2026-09-01T04:10:00', severity: 'low' },
  { id: 'EV-3306', type: 'Night Movement', camera: 'Warehouse Floor', timestamp: '2026-09-01T02:14:37', severity: 'medium' },
  { id: 'EV-3305', type: 'Camera Offline', camera: 'East Perimeter', timestamp: '2026-08-31T23:52:00', severity: 'high' },
  { id: 'EV-3304', type: 'Person Detected', camera: 'East Perimeter', timestamp: '2026-08-31T23:51:19', severity: 'high' },
  { id: 'EV-3303', type: 'Vehicle Detected', camera: 'Loading Dock', timestamp: '2026-08-31T20:05:11', severity: 'low' },
  { id: 'EV-3302', type: 'Plate Recognized', camera: 'Parking Lot A', timestamp: '2026-08-31T18:30:05', severity: 'low' },
  { id: 'EV-3301', type: 'Person Detected', camera: 'Warehouse Floor', timestamp: '2026-08-31T15:12:44', severity: 'low' },
  { id: 'EV-3300', type: 'Loitering', camera: 'Main Gate', timestamp: '2026-08-31T11:03:27', severity: 'medium' },
]
