import { useState } from 'react'
import './Login.css'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [scanning, setScanning] = useState(false)

  const handleLogin = async (e) => {
    e.preventDefault()

    setError('')
    setScanning(true)

    try {
      const formData = new URLSearchParams()

      formData.append('username', username.trim())
      formData.append('password', password)

      const response = await fetch(
        'http://127.0.0.1:8000/auth/login',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formData.toString(),
        }
      )

      const text = await response.text()

      let data

      try {
        data = JSON.parse(text)
      } catch {
        throw new Error(
          `Server returned an invalid response (${response.status})`
        )
      }

      if (!response.ok) {
        throw new Error(
          data.detail || `Login failed (${response.status})`
        )
      }

      console.log('LOGIN SUCCESS:', data)

      if (!data.access_token) {
        throw new Error(
          'Login succeeded but no access token was returned'
        )
      }

      // Save login information for this browser session
      sessionStorage.setItem(
        'access_token',
        data.access_token
      )

      sessionStorage.setItem(
        'role',
        data.role || 'operator'
      )

      sessionStorage.setItem(
        'full_name',
        data.full_name || 'Border Security Operator'
      )

      console.log('LOGIN COMPLETE')
      console.log('Redirecting to dashboard...')

      setScanning(false)

      // Go to Dashboard
      window.location.href = '/'

    } catch (err) {
      console.error('LOGIN ERROR:', err)

      setScanning(false)

      setError(
        `ACCESS DENIED - ${err.message}`
      )
    }
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