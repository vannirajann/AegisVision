// ---------------------------------------------------------------------
// Central place for every call to Person 5's FastAPI backend.
// Right now nothing here is used yet — pages read from src/data/*.
// When the backend is ready, swap the mock imports in each page for
// the matching function below, one page at a time.
// ---------------------------------------------------------------------

// Set this to the backend's real address once it exists.
// Example: http://localhost:8000
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    throw new Error(`Request to ${path} failed with status ${res.status}`)
  }
  return res.json()
}

export function getCameras() {
  return request('/api/cameras')
}

export function getDashboardStats() {
  return request('/api/dashboard/stats')
}

export function getAlerts() {
  return request('/api/alerts')
}

export function acknowledgeAlert(alertId) {
  return request(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' })
}

export function getEvents() {
  return request('/api/events')
}

export function getAnprLog() {
  return request('/api/anpr')
}
