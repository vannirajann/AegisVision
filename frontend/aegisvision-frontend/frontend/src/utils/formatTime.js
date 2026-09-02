/**
 * Format an ISO timestamp for display in tables and lists.
 * Example: "01 Sep, 06:42:10"
 */
export function formatTimestamp(isoString) {
  const d = new Date(isoString)
  const day = d.toLocaleDateString(undefined, { day: '2-digit', month: 'short' })
  const time = d.toLocaleTimeString('en-GB', { hour12: false })
  return `${day}, ${time}`
}

/**
 * Rough "x minutes/hours ago" label for recent-activity style lists.
 */
export function timeAgo(isoString) {
  const diffMs = Date.now() - new Date(isoString).getTime()
  const minutes = Math.floor(diffMs / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}
