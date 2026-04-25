import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  CartesianGrid, ResponsiveContainer
} from 'recharts'
import { getEvalHistory, runEval } from '../hooks/useApi'

const METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
const METRIC_COLORS = {
  faithfulness:       '#4a6cf7',
  answer_relevancy:   '#34d399',
  context_precision:  '#f59e0b',
  context_recall:     '#f87171',
}

function MetricCell({ value }) {
  if (value == null) return <span className="text-gray-600">—</span>
  const pct = Math.round(value * 100)
  const color =
    pct >= 75 ? 'text-green-400' :
    pct >= 55 ? 'text-yellow-400' :
                'text-red-400'
  return <span className={`font-mono font-semibold ${color}`}>{pct}%</span>
}

export default function EvalPage() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [variant, setVariant] = useState('reranked')
  const [notes, setNotes] = useState('')
  const [message, setMessage] = useState(null)

  const fetchHistory = async () => {
    try {
      const data = await getEvalHistory()
      setHistory(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchHistory() }, [])

  const handleRun = async () => {
    setRunning(true)
    setMessage(null)
    try {
      const data = await runEval(variant, notes)
      setMessage({ type: 'success', text: data.message || 'Evaluation started.' })
      // poll for results after 30s
      setTimeout(fetchHistory, 30_000)
    } catch (err) {
      setMessage({ type: 'error', text: err.response?.data?.detail || err.message })
    } finally {
      setRunning(false)
    }
  }

  // Prepare chart data: one point per run, chronological
  const chartData = [...history]
    .reverse()
    .map((r, i) => ({
      name: `${r.pipeline_variant} #${i + 1}`,
      faithfulness: r.faithfulness != null ? +(r.faithfulness * 100).toFixed(1) : null,
      answer_relevancy: r.answer_relevancy != null ? +(r.answer_relevancy * 100).toFixed(1) : null,
      context_precision: r.context_precision != null ? +(r.context_precision * 100).toFixed(1) : null,
      context_recall: r.context_recall != null ? +(r.context_recall * 100).toFixed(1) : null,
    }))

  const hasChartData = chartData.some(d =>
    METRICS.some(m => d[m] != null)
  )

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Evaluation Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          RAGAS metrics across pipeline variants — faithfulness, relevancy, precision, recall.
        </p>
      </div>

      {/* Run eval panel */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">Run New Evaluation</h2>
        <div className="flex flex-wrap gap-3 items-end">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Pipeline</label>
            <select
              value={variant}
              onChange={e => setVariant(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-arxiv-500"
            >
              <option value="naive">Naive (no reranker)</option>
              <option value="reranked">Reranked (full pipeline)</option>
            </select>
          </div>
          <div className="flex-1 min-w-48">
            <label className="text-xs text-gray-500 block mb-1">Notes (optional)</label>
            <input
              value={notes}
              onChange={e => setNotes(e.target.value)}
              placeholder="e.g. top-k=20, new embedding model"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-arxiv-500"
            />
          </div>
          <button
            onClick={handleRun}
            disabled={running}
            className="bg-arxiv-500 hover:bg-arxiv-600 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            {running ? 'Starting...' : 'Run Eval'}
          </button>
          <button
            onClick={fetchHistory}
            className="bg-gray-800 hover:bg-gray-700 text-gray-400 text-sm px-3 py-2 rounded-lg transition-colors"
            title="Refresh"
          >
            ↻
          </button>
        </div>
        {message && (
          <p className={`mt-3 text-sm ${message.type === 'error' ? 'text-red-400' : 'text-green-400'}`}>
            {message.text}
          </p>
        )}
        <p className="text-xs text-gray-600 mt-3">
          Requires <code className="bg-gray-800 px-1 rounded">OPENAI_API_KEY</code> for LLM-based RAGAS metrics.
          Runs asynchronously — refresh after ~2 min.
        </p>
      </div>

      {/* Chart */}
      {hasChartData && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Scores Over Runs (%)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#6b7280' }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#e5e7eb', fontSize: 11 }}
                itemStyle={{ fontSize: 11 }}
                formatter={(v) => v != null ? `${v}%` : '—'}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
              {METRICS.map(m => (
                <Line
                  key={m}
                  type="monotone"
                  dataKey={m}
                  stroke={METRIC_COLORS[m]}
                  strokeWidth={2}
                  dot={{ r: 3, fill: METRIC_COLORS[m] }}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* History table */}
      <div>
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Eval History</h2>
        {loading ? (
          <div className="text-center py-8 text-gray-600 text-sm">Loading...</div>
        ) : history.length === 0 ? (
          <div className="text-center py-12 text-gray-600">
            <p className="text-3xl mb-2">📊</p>
            <p className="text-sm">No evaluation runs yet. Run your first eval above.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800 text-xs text-gray-500 text-left">
                  <th className="pb-2 pr-4 font-medium">Date</th>
                  <th className="pb-2 pr-4 font-medium">Variant</th>
                  <th className="pb-2 pr-4 font-medium">Faithfulness</th>
                  <th className="pb-2 pr-4 font-medium">Relevancy</th>
                  <th className="pb-2 pr-4 font-medium">Precision</th>
                  <th className="pb-2 pr-4 font-medium">Recall</th>
                  <th className="pb-2 pr-4 font-medium">N</th>
                  <th className="pb-2 font-medium">Notes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {history.map((r) => (
                  <tr key={r.run_id} className="hover:bg-gray-900/50 transition-colors">
                    <td className="py-2.5 pr-4 text-gray-500 font-mono text-xs whitespace-nowrap">
                      {r.created_at?.slice(0, 19).replace('T', ' ')}
                    </td>
                    <td className="py-2.5 pr-4">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                        r.pipeline_variant === 'reranked'
                          ? 'bg-arxiv-500/20 text-arxiv-500'
                          : 'bg-gray-700 text-gray-400'
                      }`}>
                        {r.pipeline_variant}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4"><MetricCell value={r.faithfulness} /></td>
                    <td className="py-2.5 pr-4"><MetricCell value={r.answer_relevancy} /></td>
                    <td className="py-2.5 pr-4"><MetricCell value={r.context_precision} /></td>
                    <td className="py-2.5 pr-4"><MetricCell value={r.context_recall} /></td>
                    <td className="py-2.5 pr-4 text-gray-500 font-mono text-xs">{r.num_questions}</td>
                    <td className="py-2.5 text-gray-500 text-xs max-w-xs truncate">{r.notes || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
