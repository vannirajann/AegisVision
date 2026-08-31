export type CameraStatus = 'online' | 'offline' | 'warning';

export interface Camera {
  id: string;
  name: string;
  zone: string;
  status: CameraStatus;
  resolution: string;
  lastMotion: string;
}

export type AlertSeverity = 'critical' | 'warning' | 'info';

export interface AlertItem {
  id: string;
  title: string;
  detail: string;
  zone: string;
  severity: AlertSeverity;
  time: string;
  acknowledged: boolean;
}

export type DeviceKind = 'camera' | 'sensor' | 'access-panel' | 'nvr';
export type DeviceStatus = 'online' | 'offline' | 'maintenance';

export interface Device {
  id: string;
  name: string;
  kind: DeviceKind;
  zone: string;
  status: DeviceStatus;
  ip: string;
  lastSeen: string;
}

export type UserRole = 'Administrator' | 'Operator' | 'Viewer';

export interface AppUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  lastActive: string;
  status: 'active' | 'suspended';
}

export interface TrafficPoint {
  time: string;
  events: number;
  motion: number;
}
