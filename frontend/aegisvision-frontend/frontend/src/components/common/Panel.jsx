import './Panel.css'

export default function Panel({ title, action, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || action) && (
        <div className="panel-header">
          {title && <h2 className="panel-title">{title}</h2>}
          {action}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </section>
  )
}
