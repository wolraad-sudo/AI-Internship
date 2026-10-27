# Week 1 — `/ask` Demo (5 stages)

Build a typed LLM endpoint step by step. Each stage is a standalone FastAPI app you can run and compare.

## Setup

This project uses the system-wide Python (no `.venv`). Start every Python tool with `python -m`,
so pip, uvicorn and streamlit always come from the same Python that runs the code:

```powershell
copy .env.example .env                       # then fill in OPENAI_API_KEY and PINECONE_API_KEY
python -m pip install -r requirements.txt    # re-run whenever requirements.txt changes
```

Not sure which Python you're on? This prints its path; it should be under `AppData\Local\Python`:

```powershell
python -c "import sys; print(sys.executable)"
```

## Demo stages

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `completions.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
python -m uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Streamlit demo runner

Interactive UI for all five stages:

```bash
python -m streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal.

## Test with curl

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG in one sentence?"}'
```

Stage 5 example (model + cost):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "gpt-4o-mini"}'
```

## Session 2 — Pinecone vector store

`rag.py` holds the vector store layer (chunking, `text-embedding-3-small` embeddings, upsert, query).
The same embedding model is used at ingest and query time, and the index is created with 1536 dimensions to match.

Add to `.env` (see `.env.example`):

```
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=session2-rag   # optional, created on first ingest
```

Check Pinecone is reachable (returns 200 when OK, 503 with an `error` field otherwise):

```bash
python rag.py
curl -s http://127.0.0.1:8000/health/pinecone
```

### Ingest a document

`POST /ingest` splits the text with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`, default 800; `CHUNK_OVERLAP`, default 100),
embeds each chunk, and upserts it with metadata `document_id`, `chunk_index`, `source` (plus any extra `metadata` you send).
Re-ingesting the same `document_id` replaces its chunks. Empty `text` or `document_id` returns a 400.

```bash
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"document_id": "handbook", "source": "handbook.md", "text": "Employees get 25 days of paid leave per year."}'
```

PowerShell (where `curl` is an alias for `Invoke-WebRequest`, so the command above fails):

```powershell
$body = @{
    document_id = "handbook"
    source      = "handbook.md"
    text        = "Employees get 25 days of paid leave per year."
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ingest" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

```json
{"document_id": "handbook", "chunks_indexed": 1, "status": "indexed", "tokens_used": 11, "cost_usd": 0.0}
```

### Check retrieval (no LLM call)

`GET /debug/retrieve?q=...&top_k=5` embeds the question and returns the nearest chunks with their
cosine similarity `score`, `document_id`, `chunk_index`, `source` and `text`. It only costs one embedding call.

```bash
curl -s "http://127.0.0.1:8000/debug/retrieve?q=How%20many%20days%20of%20leave"
```

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/debug/retrieve?q=How many days of leave" | ConvertTo-Json -Depth 5
```

### Ask with RAG

`POST /ask` now embeds the question, retrieves the `top_k` nearest chunks (default 5), and answers
**only** from them, citing `document_id`s inline (`[handbook]`) and in `answer.citations`. If the chunks
don't contain the answer, it replies `I don't know based on the provided documents.` with
`sources_needed: true`. The prompt template is `GROUNDING_SYSTEM_PROMPT` in `main.py`.

`tokens_used` and `cost_usd` include the question embedding, and `retrieved_chunk_ids` lists what was retrieved.
Send `"use_rag": false` for the original Session 1 behaviour and response shape.

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ask" -ContentType "application/json" `
    -Body '{"question": "How many days of paid leave do employees get?", "top_k": 5}' | ConvertTo-Json -Depth 5
```

```json
{
  "answer": {
    "answer": "Employees get 25 days of paid leave per year [handbook].",
    "confidence": 1.0,
    "sources_needed": false,
    "citations": ["handbook"]
  },
  "tokens_used": 547,
  "model": "gpt-4o",
  "latency_ms": 1900,
  "cost_usd": 0.0015,
  "retrieved_chunk_ids": ["handbook#0"]
}
```

## Deploy on Render

- Root directory: `ai-engineering-bootcamp-v2/week-1`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment: set `OPENAI_API_KEY` and `PINECONE_API_KEY` (plus any optional `PINECONE_*` overrides) in the service's Environment tab. `.env` is not deployed.
- After deploy, open `https://<your-service>.onrender.com/health/pinecone` to confirm Pinecone is reachable.
  Don't use it as Render's health check path: a Pinecone outage would then block deploys of `/ask` too.

## Smoke-test all stages

Requires a valid `OPENAI_API_KEY`:

```bash
python test_all_stages.py
```

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined) + /health/pinecone
├── rag.py               # Session 2 Pinecone vector store layer
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
