// ---------------------------------------------------------------------
// Real-time alert channel: Backend WebSocket -> this file -> UI.
// Not wired into any page yet. When the backend exposes a WebSocket
// endpoint, use it like this from a page or a top-level component:
//
//   import { connectAlertSocket } from '../services/websocket'
//
//   useEffect(() => {
//     const socket = connectAlertSocket((alert) => {
//       // e.g. prepend the new alert to state
//     })
//     return () => socket.close()
//   }, [])
// ---------------------------------------------------------------------

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/alerts'

export function connectAlertSocket(onAlert) {
  const socket = new WebSocket(WS_URL)

  socket.onmessage = (event) => {
    try {
      const alert = JSON.parse(event.data)
      onAlert(alert)
    } catch (err) {
      console.error('Could not parse incoming alert:', err)
    }
  }

  socket.onerror = (err) => {
    console.error('Alert socket error:', err)
  }

  return socket
}
