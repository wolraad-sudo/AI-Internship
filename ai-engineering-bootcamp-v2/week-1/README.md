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

### RAG UI (Streamlit)

`rag_ui.py` is a thin UI over the API: an **Ask** tab (shows the answer, cited `document_id`s or a
yellow refusal box, retrieved chunk IDs, tokens, cost and latency) and an **Ingest** tab (paste text +
`document_id`). It only calls `/ask`, `/ingest` and `/health/pinecone` — no RAG logic runs in Streamlit.

Point it at the live service with `RAG_API_URL` (or change the sidebar field; the default is the local server):

```powershell
$env:RAG_API_URL = "https://ai-eng-wolraad.onrender.com"; python -m streamlit run rag_ui.py
```

### Golden-set eval

`golden_eval.py` asks 10 questions with known answers from the Northwind pack (8 answerable, 2 that must be refused)
and scores each one:

- **Retrieval hit**: the expected document's chunk containing the key sentence is in the top 5 (`/debug/retrieve`).
- **Faithful**: a separate judge model (gpt-4o-mini) confirms every claim in the answer is supported by the retrieved chunks.
- **Correct**: the answer contains the expected facts, or for refusal questions is the exact refusal text.

```powershell
python golden_eval.py --base-url https://ai-eng-wolraad.onrender.com --out golden_results.md
```

#### Results - `https://ai-eng-wolraad.onrender.com` (top_k=5)

| # | Question | Expected | Service answer | Retrieval hit (top-5) | Faithful | Correct |
|---|---|---|---|---|---|---|
| 1 | How many days per week can employees work remotely? | Up to 3 days per week | Employees may work remotely up to three days per week [POL-101]. | ✅ | ✅ | ✅ |
| 2 | How many days of annual leave do employees get? | 28 days plus public holidays | Employees get 28 days of annual leave plus public holidays [POL-101]. | ✅ | ✅ | ✅ |
| 3 | What is the mileage rate for business travel? | 45p per mile for travel over 50 miles | The mileage rate for business travel is 45 pence per mile [POL-114]. | ✅ | ✅ | ✅ |
| 4 | How long do I have to submit an expense claim? | Within 30 days, with a receipt | You must submit an expense claim within 30 days of the transaction. Claims older than 30 days need director approval and will only be paid in the next cycle if approved. Claims older than 90 days are denied, except in cases of documented system outages [POL-114]. | ✅ | ✅ | ✅ |
| 5 | How quickly must a lost laptop be reported? | Within 1 hour, to the security desk | A lost laptop must be reported within one hour to the security desk, as stated in Section 3 [POL-207]. | ✅ | ✅ | ✅ |
| 6 | What are the password requirements? | At least 14 characters, rotated every 90 days | Passwords must be at least 14 characters long and rotated every 90 days. Multi-factor authentication is mandatory for all systems. It is preferred to use a company-approved password manager and unique passwords for each system. Reusing Northwind passwords on personal sites, sharing passwords in Slack, email, or tickets is not allowed. If a shared break-glass account is needed, it must be vaulted, logged, and owned by IT [POL-207]. | ✅ | ✅ | ✅ |
| 7 | What is the WB-9 payload limit? | 25 kg | The payload limit for the WB-9 warehouse robot is 25 kg [SPEC-WB9]. | ✅ | ✅ | ✅ |
| 8 | How long does the WB-9 battery last and how long does it take to recharge? | 8 hours; 90 minute recharge | The WB-9 robot's battery lasts for 8 hours on a single charge and takes 90 minutes to recharge [SPEC-WB9]. | ✅ | ✅ | ✅ |
| 9 | What is the parental leave policy? | Refuse (not in docs) | I don't know based on the provided documents. | N/A | ✅ | ✅ |
| 10 | Do employees get stock options? | Refuse (not in docs) | I don't know based on the provided documents. | N/A | ✅ | ✅ |

**Summary:** retrieval hit 8/8 · faithful 10/10 · correct 10/10 (including 2 refusal questions).

### Add-on: reranking

Every Northwind policy has the same header fields (Owner, Approver, Next review), so questions about them make
header chunks from several documents compete, and vector similarity alone can rank the right one low.
With reranking on (the default), `/ask` and `/debug/retrieve` fetch `RERANK_CANDIDATES` (20) chunks by vector
similarity, re-order them with Pinecone's hosted `bge-reranker-v2-m3` (same `PINECONE_API_KEY`), and keep the top `top_k`.
Send `"rerank": false` (or `?rerank=false`) to compare. If the reranker fails (e.g. the free plan's monthly
quota), retrieval falls back to vector order and the response shows `"reranked": false`.

```powershell
python rerank_eval.py --base-url https://ai-eng-wolraad.onrender.com --out rerank_results.md
```

#### Results - `http://127.0.0.1:8010` (top_k=5)

| Question | Expected doc | Rank without rerank | Rank with rerank |
|---|---|---|---|
| How many days per week can employees work remotely? | POL-101 | #1 | #1 |
| How many days of annual leave do employees get? | POL-101 | #1 | #1 |
| What is the mileage rate for business travel? | POL-114 | #1 | #1 |
| How long do I have to submit an expense claim? | POL-114 | #1 | #2 ⬇️ |
| How quickly must a lost laptop be reported? | POL-207 | #1 | #1 |
| What are the password requirements? | POL-207 | #1 | #1 |
| What is the WB-9 payload limit? | SPEC-WB9 | #1 | #1 |
| How long does the WB-9 battery last and how long does it take to recharge? | SPEC-WB9 | #1 | #1 |
| Who owns the facilities policy? | POL-118 | #5 | #1 ⬆️ |
| When is the security policy next reviewed? | POL-207 | #2 | #1 ⬆️ |
| Who approves the expenses policy? | POL-114 | #3 | #1 ⬆️ |
| What version is the employee handbook? | POL-101 | #1 | #1 |

| Metric | Without rerank | With rerank |
|---|---|---|
| hit@1 | 0.75 | 0.92 |
| hit@3 | 0.92 | 1.00 |
| hit@5 | 1.00 | 1.00 |
| mrr | 0.84 | 0.96 |
| avg /debug/retrieve latency | 477 ms | 700 ms |

**Answer check** - "Who owns the facilities policy?" with top_k=3:

- rerank off: ❌ "I don't know based on the provided documents." (chunks: POL-101#21, POL-207#11, POL-114#0)
- rerank on : ✅ "The facilities policy is owned by the Head of Workplace [POL-118]." (chunks: POL-118#0, POL-207#1, POL-114#0)

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
