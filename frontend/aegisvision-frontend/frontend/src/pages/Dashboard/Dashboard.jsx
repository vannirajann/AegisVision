import { useEffect, useState } from 'react'
import StatCard from '../../components/common/StatCard.jsx'
import Panel from '../../components/common/Panel.jsx'
import SeverityBadge from '../../components/common/SeverityBadge.jsx'
import EmptyState from '../../components/common/EmptyState.jsx'
import { mockCameras } from '../../data/mockCameras.js'
import { mockAlerts } from '../../data/mockAlerts.js'
import { mockEvents } from '../../data/mockEvents.js'
import { timeAgo, formatTimestamp } from '../../utils/formatTime.js'
import './Dashboard.css'

export default function Dashboard() {
  const totalCameras = mockCameras.length
  const activeCameras = mockCameras.filter(
    (c) => c.status === 'online'
  ).length

  const activeAlerts = mockAlerts.filter(
    (a) => !a.acknowledged
  ).length

  const highSeverityAlerts = mockAlerts.filter(
    (a) => !a.acknowledged && a.severity === 'high'
  ).length

  const recentEvents = mockEvents.slice(0, 6)

  const activeAlertList = mockAlerts
    .filter((a) => !a.acknowledged)
    .slice(0, 5)

  const [botMessage, setBotMessage] = useState('Monitoring perimeter...')
  const [botMood, setBotMood] = useState('normal')
  const [isReacting, setIsReacting] = useState(false)
  const [mouseX, setMouseX] = useState(0)

  /*
   * AI CHARACTER REACTION
   */

  const react = (message, mood = 'normal') => {
    setBotMessage(message)
    setBotMood(mood)
    setIsReacting(true)

    setTimeout(() => {
      setIsReacting(false)
    }, 900)

    setTimeout(() => {
      setBotMessage('Monitoring perimeter...')
      setBotMood('normal')
    }, 2600)
  }

  /*
   * React whenever the user clicks somewhere
   */

  useEffect(() => {
    const handleClick = (event) => {
      const target = event.target

      if (target.closest('.alert-list-item')) {
        react('Threat detected!', 'alert')
        return
      }

      if (target.closest('.dashboard-stats')) {
        react('Analyzing statistics...', 'scan')
        return
      }

      if (target.closest('.activity-item')) {
        react('Reviewing activity...', 'scan')
        return
      }

      react('Interaction detected!', 'happy')
    }

    document.addEventListener('click', handleClick)

    return () => {
      document.removeEventListener('click', handleClick)
    }
  }, [])

  /*
   * Mouse tracking
   */

  useEffect(() => {
    const handleMouseMove = (event) => {
      const center = window.innerWidth / 2
      const difference = event.clientX - center

      const direction = Math.max(
        -8,
        Math.min(8, difference / 80)
      )

      setMouseX(direction)
    }

    window.addEventListener('mousemove', handleMouseMove)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
    }
  }, [])

  return (
    <div className="dashboard">

      {/* =====================================================
          HERO
      ===================================================== */}

      <div className="dashboard-hero">

        <div className="hero-content">

          <div className="hero-status">
            <span className="hero-status-dot"></span>
            AI SURVEILLANCE SYSTEM
          </div>

          <h1>
            AegisVision
            <span>Command Center</span>
          </h1>

          <p>
            Intelligent CCTV monitoring, threat detection and
            perimeter analytics.
          </p>

        </div>

        <div className="hero-radar">

          <div className="radar-ring radar-ring-1"></div>
          <div className="radar-ring radar-ring-2"></div>
          <div className="radar-ring radar-ring-3"></div>

          <div className="radar-sweep"></div>

          <div className="radar-dot radar-dot-1"></div>
          <div className="radar-dot radar-dot-2"></div>
          <div className="radar-dot radar-dot-3"></div>

          <div className="radar-center"></div>

        </div>

      </div>

      {/* =====================================================
          STATISTICS
      ===================================================== */}

      <div
        className="dashboard-stats"
        onMouseEnter={() =>
          react('Checking system status...', 'scan')
        }
      >

        <StatCard
          label="Total Cameras"
          value={totalCameras}
          tone="neutral"
          hint={`${totalCameras} registered`}
        />

        <StatCard
          label="Active Cameras"
          value={activeCameras}
          tone="online"
          hint={`${totalCameras - activeCameras} offline`}
        />

        <StatCard
          label="Recent Events"
          value={mockEvents.length}
          tone="info"
          hint="Last 24 hours"
        />

        <StatCard
          label="Active Alerts"
          value={activeAlerts}
          tone="medium"
          hint="Awaiting acknowledgement"
        />

        <StatCard
          label="High-Severity Alerts"
          value={highSeverityAlerts}
          tone="high"
          hint="Needs immediate review"
        />

      </div>

      {/* =====================================================
          MAIN PANELS
      ===================================================== */}

      <div className="dashboard-grid">

        <Panel
          title="Recent Activity"
          className="dashboard-activity"
        >

          {recentEvents.length === 0 ? (
            <EmptyState text="No activity recorded yet." />
          ) : (

            <ul className="activity-list">

              {recentEvents.map((event) => (

                <li
                  key={event.id}
                  className={`activity-item sev-${event.severity}`}
                  onMouseEnter={() =>
                    react('Reviewing event...', 'scan')
                  }
                >

                  <div className="activity-item-main">

                    <span className="activity-item-type">
                      {event.type}
                    </span>

                    <span className="activity-item-camera">
                      {event.camera}
                    </span>

                  </div>

                  <span className="activity-item-time">
                    {timeAgo(event.timestamp)}
                  </span>

                </li>

              ))}

            </ul>

          )}

        </Panel>

        <Panel
          title="Active Alerts"
          className="dashboard-alerts"
        >

          {activeAlertList.length === 0 ? (

            <EmptyState
              text="No unacknowledged alerts right now."
            />

          ) : (

            <ul className="alert-list">

              {activeAlertList.map((alert) => (

                <li
                  key={alert.id}
                  className="alert-list-item"
                  onMouseEnter={() =>
                    react('Threat detected!', 'alert')
                  }
                >

                  <div className="alert-list-top">

                    <span className="alert-list-type">
                      {alert.type}
                    </span>

                    <SeverityBadge
                      level={alert.severity}
                    />

                  </div>

                  <p className="alert-list-message">
                    {alert.message}
                  </p>

                  <div className="alert-list-meta">

                    <span>
                      {alert.camera}
                    </span>

                    <span>
                      {formatTimestamp(alert.timestamp)}
                    </span>

                  </div>

                </li>

              ))}

            </ul>

          )}

        </Panel>

      </div>

      {/* =====================================================
          AI CARTOON ASSISTANT
      ===================================================== */}

      <div
        className={`ai-assistant ${botMood} ${
          isReacting ? 'reacting' : ''
        }`}
        style={{
          '--eye-follow': `${mouseX}px`
        }}
        onMouseEnter={() =>
          react('Hello, commander!', 'happy')
        }
        onMouseLeave={() =>
          react('Resuming surveillance...', 'normal')
        }
        onClick={(event) => {
          event.stopPropagation()
          react('Systems are operational!', 'happy')
        }}
      >

        {/* Speech bubble */}

        <div className="ai-speech">
          <span>{botMessage}</span>
          <i></i>
        </div>

        {/* Notification waves */}

        <div className="assistant-wave wave-one"></div>
        <div className="assistant-wave wave-two"></div>
        <div className="assistant-wave wave-three"></div>

        {/* Antenna */}

        <div className="bot-antenna">

          <div className="antenna-line"></div>

          <div className="antenna-light"></div>

        </div>

        {/* Head */}

        <div className="bot-head">

          <div className="bot-head-shine"></div>

          <div className="bot-eyes">

            <span
              style={{
                transform: `translateX(var(--eye-follow))`
              }}
            ></span>

            <span
              style={{
                transform: `translateX(var(--eye-follow))`
              }}
            ></span>

          </div>

          <div className="bot-face-line"></div>

          <div className="bot-camera">

            <div className="camera-glass"></div>

          </div>

        </div>

        {/* Body */}

        <div className="bot-body">

          <div className="bot-chest-screen">

            <div className="chest-ai">
              AI
            </div>

            <div className="chest-signal">
              <span></span>
              <span></span>
              <span></span>
              <span></span>
            </div>

          </div>

          <div className="bot-body-light"></div>

        </div>

        {/* Arms */}

        <div className="bot-arm bot-arm-left">
          <div className="bot-hand"></div>
        </div>

        <div className="bot-arm bot-arm-right">
          <div className="bot-hand"></div>
        </div>

        {/* Feet */}

        <div className="bot-feet">

          <span></span>
          <span></span>

        </div>

        {/* Ground */}

        <div className="bot-ground"></div>

      </div>

      {/* =====================================================
          FOOTER
      ===================================================== */}

      <div className="dashboard-footer-status">

        <span className="footer-line"></span>

        <span>
          AEGISVISION AI CORE
        </span>

        <span className="footer-line"></span>

      </div>

    </div>
  )
}