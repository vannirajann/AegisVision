// Placeholder alerts. Real alerts will arrive two ways later:
//  - GET /api/alerts on page load
//  - a WebSocket push for anything new (see src/services/websocket.js)
export const mockAlerts = [
  {
    id: 'AL-1042',
    type: 'Intrusion',
    camera: 'Rear Fence Line',
    severity: 'high',
    timestamp: '2026-09-01T06:42:10',
    message: 'Person detected crossing the restricted fence line.',
    acknowledged: false,
  },
  {
    id: 'AL-1041',
    type: 'ANPR',
    camera: 'Main Gate',
    severity: 'medium',
    timestamp: '2026-09-01T06:20:45',
    message: 'Unrecognized plate KA-05-MX-4471 entered the premises.',
    acknowledged: false,
  },
  {
    id: 'AL-1040',
    type: 'Loitering',
    camera: 'Visitor Lobby',
    severity: 'low',
    timestamp: '2026-09-01T05:58:02',
    message: 'A person has remained in frame for over 12 minutes.',
    acknowledged: true,
  },
  {
    id: 'AL-1039',
    type: 'Night Movement',
    camera: 'Warehouse Floor',
    severity: 'medium',
    timestamp: '2026-09-01T02:14:37',
    message: 'Movement detected after hours with no scheduled staff.',
    acknowledged: true,
  },
  {
    id: 'AL-1038',
    type: 'Intrusion',
    camera: 'East Perimeter',
    severity: 'high',
    timestamp: '2026-08-31T23:51:19',
    message: 'Camera went offline immediately after a fence-line trigger.',
    acknowledged: false,
  },
  {
    id: 'AL-1037',
    type: 'ANPR',
    camera: 'Parking Lot A',
    severity: 'low',
    timestamp: '2026-08-31T18:30:05',
    message: 'Plate MH-12-AB-7788 matched an allow-listed vehicle.',
    acknowledged: true,
  },
]
