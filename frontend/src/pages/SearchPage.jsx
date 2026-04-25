import { useState } from 'react'
import { searchPapers } from '../hooks/useApi'
import PaperCard from '../components/PaperCard'

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await searchPapers(query, topK)
      setResults(data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Semantic Paper Search</h1>
        <p className="text-gray-500 text-sm mt-1">
          Search across {'>'}200 ArXiv papers (cs.LG · cs.AI · cs.CL) using dense vector retrieval.
        </p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-2">
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="e.g. attention mechanism transformers, LoRA fine-tuning, diffusion models..."
          className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-arxiv-500 focus:ring-1 focus:ring-arxiv-500/40 transition-colors"
        />
        <select
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2.5 text-sm text-gray-400 focus:outline-none focus:border-arxiv-500"
        >
          {[5, 10, 20].map(k => <option key={k} value={k}>Top {k}</option>)}
        </select>
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="bg-arxiv-500 hover:bg-arxiv-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
              </svg>
              Searching
            </span>
          ) : 'Search'}
        </button>
      </form>

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {results && (
        <div>
          <p className="text-xs text-gray-500 mb-3">
            {results.total} results for "{results.query}"
          </p>
          <div className="space-y-3">
            {results.results.map((paper) => (
              <PaperCard key={paper.arxiv_id} paper={paper} />
            ))}
          </div>
          {results.results.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              No papers found. Try a different query.
            </div>
          )}
        </div>
      )}

      {!results && !loading && (
        <div className="text-center py-16 text-gray-600">
          <p className="text-4xl mb-3">📄</p>
          <p className="text-sm">Enter a query to search the ArXiv paper index.</p>
        </div>
      )}
    </div>
  )
}
