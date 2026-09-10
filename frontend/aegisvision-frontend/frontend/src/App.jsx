import { Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout.jsx'
import Dashboard from './pages/Dashboard/Dashboard.jsx'
import LiveMonitoring from './pages/LiveMonitoring/LiveMonitoring.jsx'
import Alerts from './pages/Alerts/Alerts.jsx'
import EventHistory from './pages/EventHistory/EventHistory.jsx'
import ANPR from './pages/ANPR/ANPR.jsx'
import Analytics from './pages/Analytics/Analytics.jsx'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live-monitoring" element={<LiveMonitoring />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/event-history" element={<EventHistory />} />
        <Route path="/anpr" element={<ANPR />} />
        <Route path="/analytics" element={<Analytics />} />
      </Route>
    </Routes>
  )
}
