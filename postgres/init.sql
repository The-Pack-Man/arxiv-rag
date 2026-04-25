-- ── Query logs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    query_text  TEXT        NOT NULL,
    answer      TEXT,
    sources     JSONB,
    latency_ms  INTEGER,
    model       TEXT,
    top_k       INTEGER,
    reranked    BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs (created_at DESC);

-- ── Eval results ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_results (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mlflow_run_id       TEXT        UNIQUE,
    pipeline_variant    TEXT        NOT NULL,  -- 'naive' | 'reranked'
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_precision   FLOAT,
    context_recall      FLOAT,
    num_questions       INTEGER,
    params              JSONB,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_results_created_at ON eval_results (created_at DESC);
