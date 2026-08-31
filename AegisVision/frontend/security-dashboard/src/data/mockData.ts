import type { Camera, AlertItem, Device, AppUser, TrafficPoint } from '../types';

export const cameras: Camera[] = [
  { id: 'CAM-014', name: 'Main Entrance', zone: 'Perimeter', status: 'online', resolution: '4K', lastMotion: '2 min ago' },
  { id: 'CAM-022', name: 'Loading Dock A', zone: 'Warehouse', status: 'online', resolution: '1080p', lastMotion: '11 min ago' },
  { id: 'CAM-007', name: 'Parking Level 2', zone: 'Garage', status: 'warning', resolution: '1080p', lastMotion: '1 min ago' },
  { id: 'CAM-031', name: 'Server Room', zone: 'Restricted', status: 'online', resolution: '4K', lastMotion: '48 min ago' },
  { id: 'CAM-018', name: 'North Corridor', zone: 'Office', status: 'offline', resolution: '1080p', lastMotion: '3 hrs ago' },
  { id: 'CAM-009', name: 'Rear Gate', zone: 'Perimeter', status: 'online', resolution: '4K', lastMotion: '6 min ago' },
  { id: 'CAM-026', name: 'Lobby', zone: 'Office', status: 'online', resolution: '1080p', lastMotion: '30 sec ago' },
  { id: 'CAM-003', name: 'Loading Dock B', zone: 'Warehouse', status: 'online', resolution: '1080p', lastMotion: '19 min ago' },
];

export const alerts: AlertItem[] = [
  { id: 'A-2291', title: 'Motion after hours', detail: 'Movement detected outside scheduled activity window.', zone: 'Server Room', severity: 'critical', time: '2 min ago', acknowledged: false },
  { id: 'A-2290', title: 'Camera signal degraded', detail: 'Frame drops exceeding threshold for 4 minutes.', zone: 'Parking Level 2', severity: 'warning', time: '9 min ago', acknowledged: false },
  { id: 'A-2288', title: 'Door held open', detail: 'Rear gate has been open for over 90 seconds.', zone: 'Perimeter', severity: 'warning', time: '22 min ago', acknowledged: true },
  { id: 'A-2285', title: 'Camera offline', detail: 'Connection lost, retry attempts exhausted.', zone: 'North Corridor', severity: 'critical', time: '3 hrs ago', acknowledged: false },
  { id: 'A-2281', title: 'Badge denied — 3 attempts', detail: 'Repeated failed access attempts on the same credential.', zone: 'Restricted', severity: 'critical', time: '4 hrs ago', acknowledged: true },
  { id: 'A-2277', title: 'Firmware update available', detail: 'NVR-02 has a pending security patch.', zone: 'Warehouse', severity: 'info', time: '6 hrs ago', acknowledged: true },
];

export const devices: Device[] = [
  { id: 'CAM-014', name: 'Main Entrance', kind: 'camera', zone: 'Perimeter', status: 'online', ip: '10.20.1.14', lastSeen: 'now' },
  { id: 'CAM-018', name: 'North Corridor', kind: 'camera', zone: 'Office', status: 'offline', ip: '10.20.1.18', lastSeen: '3 hrs ago' },
  { id: 'SEN-004', name: 'Rear Gate Contact', kind: 'sensor', zone: 'Perimeter', status: 'online', ip: '10.20.2.04', lastSeen: 'now' },
  { id: 'ACP-001', name: 'Restricted Wing Panel', kind: 'access-panel', zone: 'Restricted', status: 'online', ip: '10.20.3.01', lastSeen: 'now' },
  { id: 'NVR-02', name: 'Warehouse Recorder', kind: 'nvr', zone: 'Warehouse', status: 'maintenance', ip: '10.20.0.02', lastSeen: '1 hr ago' },
  { id: 'SEN-011', name: 'Server Room Motion', kind: 'sensor', zone: 'Restricted', status: 'online', ip: '10.20.2.11', lastSeen: 'now' },
];

export const users: AppUser[] = [
  { id: 'U-01', name: 'Priya Raman', email: 'priya.raman@perimeter.io', role: 'Administrator', lastActive: '5 min ago', status: 'active' },
  { id: 'U-02', name: 'Daniel Osei', email: 'daniel.osei@perimeter.io', role: 'Operator', lastActive: '38 min ago', status: 'active' },
  { id: 'U-03', name: 'Mei Lin Tan', email: 'mei.tan@perimeter.io', role: 'Operator', lastActive: '2 hrs ago', status: 'active' },
  { id: 'U-04', name: 'Carlos Vega', email: 'carlos.vega@perimeter.io', role: 'Viewer', lastActive: '1 day ago', status: 'suspended' },
  { id: 'U-05', name: 'Anya Petrova', email: 'anya.petrova@perimeter.io', role: 'Administrator', lastActive: '1 hr ago', status: 'active' },
];

export const traffic: TrafficPoint[] = [
  { time: '00:00', events: 4, motion: 12 },
  { time: '03:00', events: 2, motion: 6 },
  { time: '06:00', events: 5, motion: 18 },
  { time: '09:00', events: 14, motion: 52 },
  { time: '12:00', events: 21, motion: 68 },
  { time: '15:00', events: 18, motion: 61 },
  { time: '18:00', events: 26, motion: 74 },
  { time: '21:00', events: 11, motion: 34 },
];

export const zoneBreakdown = [
  { zone: 'Perimeter', value: 32 },
  { zone: 'Office', value: 21 },
  { zone: 'Warehouse', value: 18 },
  { zone: 'Restricted', value: 9 },
  { zone: 'Garage', value: 14 },
];
