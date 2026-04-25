import { useState } from 'react'

function ScoreBadge({ score }) {
  const pct = Math.round(score * 100)
  const color =
    pct >= 80 ? 'text-green-400 bg-green-400/10' :
    pct >= 60 ? 'text-yellow-400 bg-yellow-400/10' :
                'text-gray-400 bg-gray-700'
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${color}`}>
      {pct}%
    </span>
  )
}

export default function PaperCard({ paper }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 hover:border-gray-700 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <a
            href={paper.url}
            target="_blank"
            rel="noreferrer"
            className="text-arxiv-500 hover:text-arxiv-400 font-medium text-sm leading-snug line-clamp-2 transition-colors"
          >
            {paper.title}
          </a>
          <p className="text-gray-500 text-xs mt-1 truncate">
            {paper.authors?.slice(0, 4).join(', ')}
            {paper.authors?.length > 4 ? ` +${paper.authors.length - 4} more` : ''}
          </p>
        </div>
        <ScoreBadge score={paper.score} />
      </div>

      <p className={`text-gray-400 text-xs mt-2 leading-relaxed ${expanded ? '' : 'line-clamp-3'}`}>
        {paper.abstract}
      </p>

      <div className="flex items-center justify-between mt-3">
        <div className="flex gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            {expanded ? 'Show less' : 'Show more'}
          </button>
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            >
              PDF →
            </a>
          )}
        </div>
        <span className="text-xs text-gray-600 font-mono">
          {paper.arxiv_id} · {paper.published?.slice(0, 10)}
        </span>
      </div>
    </div>
  )
}
