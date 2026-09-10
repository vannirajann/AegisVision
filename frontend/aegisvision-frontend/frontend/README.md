# AegisVision — Frontend Web Dashboard

This is the frontend module of AegisVision, an AI-powered surveillance
platform. This folder (`frontend/`) is a standalone React + Vite app.
It currently runs on **mock data** — no backend is required to see the
whole dashboard working.

---

## 1. What you need installed first

- **Node.js** version 18 or newer (Vite requires it). Check with:
  ```
  node -v
  ```
  If that command fails, install Node.js from https://nodejs.org (the
  "LTS" version).

---

## 2. Where to run commands

Open **this folder** (`frontend/`) in VS Code, then open a terminal
inside VS Code: `Terminal → New Terminal`. Every command below is run
from inside `frontend/` — check your terminal prompt shows you're in
that folder before typing anything.

---

## 3. Install dependencies

```
npm install
```

**What this does:** reads `package.json` and downloads every library
the project needs (React, React Router, Vite) into a new
`node_modules/` folder. You only need to run this once, and again
any time `package.json` changes. It needs an internet connection.

---

## 4. Run the app

```
npm run dev
```

**What this does:** starts Vite's local development server. It will
print a URL, normally:

```
http://localhost:5173
```

Open that URL in your browser. The dashboard should load with mock
cameras, alerts, and events already filled in. Leave this terminal
running while you work — every time you save a file, the browser
updates automatically (this is called Hot Module Reload).

To stop the server: click into the terminal and press `Ctrl + C`.

---

## 5. Folder structure

```
frontend/
├── index.html                 Vite's HTML entry point
├── package.json                Project dependencies & scripts
├── vite.config.js              Vite configuration
├── .env.example                 Template for backend URLs (copy to .env later)
└── src/
    ├── main.jsx                 React entry point, wraps App in a Router
    ├── App.jsx                  All page routes are declared here
    ├── index.css                 Global styles
    ├── styles/
    │   └── variables.css         Color, font, and spacing tokens
    ├── components/
    │   ├── layout/                Sidebar, Topbar, Layout (page frame)
    │   └── common/                 Reusable pieces: StatCard, SeverityBadge,
    │                                StatusBadge, Panel, EmptyState
    ├── pages/
    │   ├── Dashboard/
    │   ├── LiveMonitoring/
    │   ├── Alerts/
    │   ├── EventHistory/
    │   ├── ANPR/
    │   └── Analytics/
    ├── data/                      Mock data files (stand in for the backend)
    │   ├── mockCameras.js
    │   ├── mockAlerts.js
    │   ├── mockEvents.js
    │   └── mockAnpr.js
    ├── services/
    │   ├── api.js                  Every future call to Person 5's FastAPI backend
    │   └── websocket.js            Real-time alert socket connection
    └── utils/
        └── formatTime.js            Timestamp formatting helpers
```

**Rule of thumb for adding new code:** if it's a page, it goes in
`src/pages/<PageName>/`. If it's used on more than one page, it goes
in `src/components/common/`. If it's fake data standing in for the
backend, it goes in `src/data/`.

---

## 6. How each page currently gets its data

Every page imports directly from `src/data/*.js` right now, for
example:

```js
import { mockAlerts } from '../../data/mockAlerts.js'
```

This is intentional — it lets you build and test the whole UI before
Person 5's backend exists.

---

## 7. Connecting to the real backend later (steps 16–17)

When Person 5's FastAPI backend is ready:

1. Copy `.env.example` to a new file named `.env` in this same folder,
   and fill in the real backend URL.
2. In `src/services/api.js`, the request functions
   (`getCameras`, `getAlerts`, `getEvents`, `getAnprLog`,
   `acknowledgeAlert`) are already written and pointed at
   `VITE_API_BASE_URL`.
3. Go into **one page at a time** (start with Dashboard), replace its
   `mockX` import with a `useState` + `useEffect` that calls the
   matching function from `api.js`, and test that page before moving
   to the next. Don't switch every page over at once.
4. For real-time alerts, `src/services/websocket.js` exports
   `connectAlertSocket(onAlert)`. Call it inside a `useEffect` in the
   Alerts page (or a top-level component if you want alerts to pop up
   from any page) and prepend each incoming alert to your alerts
   state.

Nothing above needs to happen yet — the mock data is enough to build
and demo the entire dashboard today.

---

## 8. Rules this module follows

- Only files inside `frontend/` are touched by this module.
- Every page currently works from mock data; the backend is a later
  swap-in, not a dependency.
- Components in `components/common/` are written to be reusable
  across pages rather than duplicated.
