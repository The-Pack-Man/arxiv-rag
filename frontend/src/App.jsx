import { Routes, Route, NavLink } from 'react-router-dom'
import SearchPage from './pages/SearchPage'
import ChatPage from './pages/ChatPage'
import EvalPage from './pages/EvalPage'

const NAV = [
  { to: '/',      label: 'Search',    icon: '🔍' },
  { to: '/chat',  label: 'Chat',      icon: '💬' },
  { to: '/eval',  label: 'Eval',      icon: '📊' },
]

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-8">
          <div className="flex items-center gap-2">
            <span className="text-arxiv-500 font-bold text-lg tracking-tight">ArXiv RAG</span>
            <span className="text-gray-500 text-sm hidden sm:block">— research paper intelligence</span>
          </div>
          <nav className="flex gap-1 ml-auto">
            {NAV.map(({ to, label, icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-arxiv-500/20 text-arxiv-500'
                      : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'
                  }`
                }
              >
                <span className="mr-1">{icon}</span>{label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-8">
        <Routes>
          <Route path="/"      element={<SearchPage />} />
          <Route path="/chat"  element={<ChatPage />} />
          <Route path="/eval"  element={<EvalPage />} />
        </Routes>
      </main>

      <footer className="border-t border-gray-800 py-3 text-center text-xs text-gray-600">
        ArXiv RAG · Qdrant + RAGAS + MLflow · All models run locally
      </footer>
    </div>
  )
}
