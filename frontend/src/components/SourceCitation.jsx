import { useState } from 'react'

export default function SourceCitation({ source, index }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-800/50 transition-colors"
      >
        <span className="text-xs text-gray-400">
          <span className="text-arxiv-500 font-mono mr-2">[{index}]</span>
          <span className="font-medium text-gray-300">{source.title}</span>
        </span>
        <svg
          className={`w-3.5 h-3.5 text-gray-500 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && (
        <div className="px-3 pb-3 border-t border-gray-800">
          <p className="text-xs text-gray-500 mt-2">
            {source.authors?.slice(0, 3).join(', ')}
          </p>
          <p className="text-xs text-gray-400 mt-1 leading-relaxed bg-gray-800/50 rounded p-2 font-mono">
            "{source.excerpt}"
          </p>
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-arxiv-500 hover:text-arxiv-400 mt-1 inline-block transition-colors"
          >
            View on ArXiv →
          </a>
        </div>
      )}
    </div>
  )
}
