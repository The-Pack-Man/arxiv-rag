import { useState, useRef, useEffect } from 'react'
import { chatWithPapers } from '../hooks/useApi'
import SourceCitation from '../components/SourceCitation'

function Message({ msg }) {
  const [showSources, setShowSources] = useState(false)

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bg-arxiv-500/20 border border-arxiv-500/30 rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-xl text-sm text-gray-100">
          {msg.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-2xl space-y-2">
        <div className="bg-gray-900 border border-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
          <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{msg.content}</p>
          {msg.latency_ms && (
            <p className="text-xs text-gray-600 mt-2">
              {msg.model} · {msg.latency_ms}ms · {msg.reranked ? 'reranked' : 'naive'}
            </p>
          )}
        </div>

        {msg.sources?.length > 0 && (
          <div>
            <button
              onClick={() => setShowSources(!showSources)}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1"
            >
              <svg className={`w-3 h-3 transition-transform ${showSources ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
              {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
            </button>
            {showSources && (
              <div className="mt-1 space-y-1">
                {msg.sources.map((s, i) => (
                  <SourceCitation key={s.arxiv_id} source={s} index={i + 1} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [useReranker, setUseReranker] = useState(true)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMsg = { role: 'user', content: input }
    setMessages(m => [...m, userMsg])
    setInput('')
    setLoading(true)

    try {
      const data = await chatWithPapers(input, useReranker)
      setMessages(m => [...m, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        model: data.model,
        latency_ms: data.latency_ms,
        reranked: data.reranked,
      }])
    } catch (err) {
      setMessages(m => [...m, {
        role: 'assistant',
        content: `Error: ${err.response?.data?.detail || err.message}`,
        sources: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-10rem)]">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Chat with Papers</h1>
          <p className="text-gray-500 text-sm mt-0.5">Ask questions — answers are grounded in retrieved ArXiv papers.</p>
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
          <div
            onClick={() => setUseReranker(!useReranker)}
            className={`relative w-9 h-5 rounded-full transition-colors ${useReranker ? 'bg-arxiv-500' : 'bg-gray-700'}`}
          >
            <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${useReranker ? 'translate-x-4' : 'translate-x-0.5'}`} />
          </div>
          Reranker
        </label>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {messages.length === 0 && (
          <div className="text-center py-16 text-gray-600">
            <p className="text-4xl mb-3">💬</p>
            <p className="text-sm">Ask a question about ML research.</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {[
                'How does LoRA reduce fine-tuning cost?',
                'What is RLHF and how does it work?',
                'Explain flash attention',
              ].map(q => (
                <button
                  key={q}
                  onClick={() => setInput(q)}
                  className="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1.5 rounded-full text-gray-400 hover:text-gray-200 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-900 border border-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1 items-center h-4">
                {[0, 1, 2].map(i => (
                  <div key={i} className="w-1.5 h-1.5 bg-arxiv-500 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={send} className="mt-3 flex gap-2">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about ML papers, methods, architectures..."
          className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-arxiv-500 focus:ring-1 focus:ring-arxiv-500/40 transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-arxiv-500 hover:bg-arxiv-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-4 py-2.5 rounded-xl transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </form>
    </div>
  )
}
