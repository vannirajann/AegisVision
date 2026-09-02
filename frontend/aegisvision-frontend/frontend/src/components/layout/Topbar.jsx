import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import './Topbar.css'

const pageTitles = {
  '/': ['Dashboard', 'Overview of the entire camera network'],
  '/live-monitoring': ['Live Monitoring', 'Real-time feeds and detection status'],
  '/alerts': ['Alerts', 'Events that need attention'],
  '/event-history': ['Event History', 'A searchable log of everything detected'],
  '/anpr': ['ANPR', 'Automatic number plate recognition log'],
  '/analytics': ['Analytics', 'Trends across cameras and event types'],
}

function useClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

export default function Topbar() {
  const location = useLocation()
  const now = useClock()
  const [title, subtitle] = pageTitles[location.pathname] || ['AegisVision', '']

  const time = now.toLocaleTimeString('en-GB', { hour12: false })
  const date = now.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' })

  return (
    <header className="topbar">
      <div>
        <h1 className="topbar-title">{title}</h1>
        <p className="topbar-subtitle">{subtitle}</p>
      </div>
      <div className="topbar-right">
        <div className="topbar-clock">
          <span className="topbar-clock-time">{time}</span>
          <span className="topbar-clock-date">{date}</span>
        </div>
        <div className="topbar-operator">
          <div className="topbar-operator-avatar">SO</div>
          <div>
            <div className="topbar-operator-name">Security Operator</div>
            <div className="topbar-operator-role">Morning shift</div>
          </div>
        </div>
      </div>
    </header>
  )
}
