import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({ baseURL: BASE, timeout: 120_000 })

export async function searchPapers(query, topK = 10) {
  const { data } = await api.post('/search', { query, top_k: topK })
  return data
}

export async function chatWithPapers(query, useReranker = true) {
  const { data } = await api.post('/chat', { query, use_reranker: useReranker })
  return data
}

export async function getEvalHistory(limit = 50) {
  const { data } = await api.get(`/eval/history?limit=${limit}`)
  return data
}

export async function runEval(pipelineVariant = 'reranked', notes = '') {
  const { data } = await api.post('/eval/run', {
    pipeline_variant: pipelineVariant,
    notes,
  })
  return data
}

export async function triggerIngest() {
  const { data } = await api.post('/ingest')
  return data
}
