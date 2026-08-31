# Perimeter — Security Surveillance Dashboard

A React + TypeScript + Tailwind CSS security operations console: live camera
grid, alerts, analytics, and device/user management. Built with mock data —
no real cameras or backend are connected.

## Getting started

```bash
npm install
npm run dev
```

Open the printed local URL. Sign in with any email/password — auth is a
local mock for demo purposes.

## Build

```bash
npm run build
npm run preview
```

## Project structure

```
src/
  components/
    layout/       Sidebar, Topbar, DashboardLayout (page shell)
    dashboard/     CameraGrid, CameraTile, AlertsPanel, charts, tables
    ui/            StatCard, StatusBadge (shared primitives)
    ProtectedRoute.tsx
  context/         AuthContext (mock auth)
  data/            mockData.ts — sample cameras, alerts, devices, users
  pages/           Login, Overview, Cameras, Alerts, Analytics, UsersDevices
  types/           shared TypeScript types
  App.tsx          routes
  main.tsx         entry point
```

## Stack

- Vite + React 18 + TypeScript
- Tailwind CSS (custom color/type tokens in `tailwind.config.js`)
- react-router-dom
- recharts (analytics charts)
- lucide-react (icons)
