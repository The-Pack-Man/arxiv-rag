import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from ragas.embeddings.base import BaseRagasEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fake OpenAI-compatible judge ─────────────────────────────────────────────
# Instructor (used by RAGAS) puts the expected JSON schema in the prompt. This
# server finds that schema and answers with a minimal valid instance, so the
# real RAGAS metric code runs end to end without a model.

def _find_schema(messages):
    decoder = json.JSONDecoder()
    best = None
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            continue
        for i, ch in enumerate(content):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(content[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and "properties" in obj:
                if best is None or len(json.dumps(obj)) > len(json.dumps(best)):
                    best = obj
    return best


def _instance(schema, defs):
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].split("/")[-1]], defs)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"]
            return _instance(options[0], defs)
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        return {k: _instance(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_instance(schema.get("items", {"type": "string"}), defs)]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    return "The context supports this statement."


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        schema = _find_schema(body.get("messages", []))
        content = json.dumps(_instance(schema, schema.get("$defs", {})) if schema else {})
        reply = {
            "id": "fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_judge():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()



class FakeEmbeddings(BaseRagasEmbedding):
    """Identical vectors for every text, so cosine similarity is always 1."""

    def embed_text(self, text, **kw):
        return [1.0, 0.0, 0.0]

    def embed_texts(self, texts, **kw):
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def aembed_text(self, text, **kw):
        return self.embed_text(text)

    async def aembed_texts(self, texts, **kw):
        return self.embed_texts(texts)
