import { Routes, Route, Navigate } from 'react-router-dom'

import Layout from './components/layout/Layout.jsx'
import Login from './pages/Login/Login.jsx'

import Dashboard from './pages/Dashboard/Dashboard.jsx'
import LiveMonitoring from './pages/LiveMonitoring/LiveMonitoring.jsx'
import Alerts from './pages/Alerts/Alerts.jsx'
import EventHistory from './pages/EventHistory/EventHistory.jsx'
import ANPR from './pages/ANPR/ANPR.jsx'
import Analytics from './pages/Analytics/Analytics.jsx'

export default function App() {

  // Check login status from sessionStorage
  const accessToken = sessionStorage.getItem('access_token')
  const isLoggedIn = Boolean(accessToken)

  return (
    <Routes>

      {/* =========================
          LOGIN PAGE
      ========================= */}
      <Route
        path="/login"
        element={
          isLoggedIn
            ? <Navigate to="/" replace />
            : <Login />
        }
      />

      {/* =========================
          PROTECTED APPLICATION
      ========================= */}
      <Route
        element={
          isLoggedIn
            ? <Layout />
            : <Navigate to="/login" replace />
        }
      >

        {/* Dashboard */}
        <Route
          path="/"
          element={<Dashboard />}
        />

        {/* Live Monitoring */}
        <Route
          path="/live-monitoring"
          element={<LiveMonitoring />}
        />

        {/* Alerts */}
        <Route
          path="/alerts"
          element={<Alerts />}
        />

        {/* Event History */}
        <Route
          path="/event-history"
          element={<EventHistory />}
        />

        {/* ANPR */}
        <Route
          path="/anpr"
          element={<ANPR />}
        />

        {/* Analytics */}
        <Route
          path="/analytics"
          element={<Analytics />}
        />

      </Route>

      {/* =========================
          UNKNOWN URL
      ========================= */}
      <Route
        path="*"
        element={<Navigate to="/login" replace />}
      />

    </Routes>
  )
}