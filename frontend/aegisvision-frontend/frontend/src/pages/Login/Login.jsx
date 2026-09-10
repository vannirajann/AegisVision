import { useState } from 'react'
import './Login.css'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [scanning, setScanning] = useState(false)

  const handleLogin = (e) => {
    e.preventDefault()

    setError('')
    setScanning(true)

    // DEMO LOGIN
    const demoUsername = 'admin'
    const demoPassword = 'admin123'

    setTimeout(() => {
      if (
        username.trim() === demoUsername &&
        password === demoPassword
      ) {
        sessionStorage.setItem(
          'access_token',
          'aegisvision-demo-token'
        )

        sessionStorage.setItem(
          'role',
          'operator'
        )

        sessionStorage.setItem(
          'full_name',
          'Border Security Operator'
        )

        console.log('LOGIN SUCCESS')
        console.log('Redirecting to dashboard...')

        window.location.href = '/'
      } else {
        setScanning(false)

        setError(
          'ACCESS DENIED - Invalid Operator ID or Security Key'
        )
      }
    }, 1200)
  }

  return (
    <div className="login-page">

      <div className="grid-background"></div>

      <div className="radar">
        <div className="radar-circle circle-1"></div>
        <div className="radar-circle circle-2"></div>
        <div className="radar-circle circle-3"></div>
        <div className="radar-line"></div>

        <span className="radar-dot dot-1"></span>
        <span className="radar-dot dot-2"></span>
        <span className="radar-dot dot-3"></span>
      </div>

      <div className="brand">

        <div className="shield">
          AEGIS
        </div>

        <div>
          <h1>AEGISVISION</h1>
          <p>AI BORDER SURVEILLANCE SYSTEM</p>
        </div>

      </div>

      <div
        className={`login-container ${
          scanning ? 'scanning' : ''
        }`}
      >

        <div className="camera-icon">

          <div className="camera-body">
            <div className="camera-lens"></div>
          </div>

          <div className="camera-stand"></div>

        </div>

        <div className="security-status">

          <span
            className={`status-dot ${
              scanning ? 'active' : ''
            }`}
          ></span>

          {scanning
            ? 'VERIFYING IDENTITY...'
            : 'SECURE ACCESS PORTAL'}

        </div>

        <h2>Command Access</h2>

        <p className="subtitle">
          Authorized Personnel Only
        </p>

        <form onSubmit={handleLogin}>

          <div className="input-group">

            <label>
              OPERATOR ID
            </label>

            <input
              type="text"
              placeholder="Enter operator ID"
              value={username}
              onChange={(e) =>
                setUsername(e.target.value)
              }
              required
              autoComplete="username"
            />

          </div>

          <div className="input-group">

            <label>
              SECURITY KEY
            </label>

            <input
              type="password"
              placeholder="Enter security key"
              value={password}
              onChange={(e) =>
                setPassword(e.target.value)
              }
              required
              autoComplete="current-password"
            />

          </div>

          {error && (
            <div className="error-message">
              ⚠ {error}
            </div>
          )}

          <button
            type="submit"
            disabled={scanning}
          >

            <span>
              {scanning
                ? 'AUTHENTICATING...'
                : 'AUTHENTICATE'}
            </span>

            <span className="button-arrow">
              →
            </span>

          </button>

        </form>

        <div className="security-footer">

          <span>
            ● ENCRYPTED CONNECTION
          </span>

          <span>
            ● AI MONITORING ACTIVE
          </span>

        </div>

      </div>

      <div className="system-info">

        <span>
          SYSTEM: AEGIS-V1
        </span>

        <span>
          NETWORK: SECURE
        </span>

        <span>
          THREAT MONITOR: ACTIVE
        </span>

      </div>

    </div>
  )
}