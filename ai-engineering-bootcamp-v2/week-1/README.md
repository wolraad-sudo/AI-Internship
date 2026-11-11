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

### CDD policy as a RAG source (KYC architecture, Session 2)

The CDD Periodic Review Policy follows `KYC_Refresh_Agent_Architecture.docx`, Session 2:

| Doc rule | How it's done here |
|---|---|
| 1. Chunk by rule | `ingest_policy_rules.py`: one document per rule (`document_id` = rule ID), text prefixed `CDD-6 Proof of residential address › CDD-6.3 Periodic review`; the CDD-4.1 table stays whole |
| 2. Metadata | `rule_id`, `topic` (e.g. `address`), `applies_to`, `review_context` (`periodic` / `event-driven` / `new person` / `all`), `references` (rule IDs in the text plus hand-checked links such as CDD-6.3 → CDD-6.2), `policy_version` (`3.2, effective 2025-01-01`) |
| 3. Hybrid retrieval | Rule IDs in a question ("get CDD-5.3") are fetched exactly and ranked first; the rest is embedding search (reranked), narrowed by `filters`; then `references` are followed one hop and CDD-1.3 / CDD-1.4 are always added. Linked rules come back in `expanded_chunk_ids`, on top of `top_k` |
| 4. Numbers extracted once | `extract_rule_params.py` writes `cdd_rule_params.json` (e.g. CDD-6.3 → `max_age_months: 12, refresh_max_age_months: 3, must_match_registers: true`); check it by hand, set `"reviewed": true`, re-ingest. Stored per chunk as `parameters` + `parameters_reviewed` |
| 5. Checkable citations | `/ask` rejects any answer citing a document not retrieved in that run (or a non-refusal with no citations): one retry, then HTTP 502 |
| 6. Own evals | `cdd_golden_set.py`: 30 questions with the rule each should retrieve, scored by `golden_eval.py --set cdd` |

```powershell
python extract_rule_params.py "C:\path\to\CDD_Periodic_Review_Policy_v3.2.md"     # then review cdd_rule_params.json
python ingest_policy_rules.py "C:\path\to\CDD_Periodic_Review_Policy_v3.2.md" --base-url https://ai-eng-wolraad.onrender.com
```

```powershell
# Filters: topic, review_context, applies_to, rule_id, document_id, policy_version
Invoke-RestMethod -Method Post -Uri "https://ai-eng-wolraad.onrender.com/ask" -ContentType "application/json" `
    -Body '{"question": "How old can a proof of address be?", "filters": {"topic": "address", "review_context": "periodic"}}' | ConvertTo-Json -Depth 5
```

### Golden-set eval: CDD Periodic Review Policy

`cdd_golden_set.py` holds 30 questions on the synthetic CDD Periodic Review Policy v3.2 (ingested rule by rule with
`ingest_policy_rules.py`, so `document_id` is the rule ID): 28 answerable, covering 24 rules, and 2 that must be refused.
Every expected answer and evidence sentence was checked against the policy text. Scoring is the same as above.

```powershell
python golden_eval.py --set cdd --base-url https://ai-eng-wolraad.onrender.com --out cdd_golden_results.md
```

#### Results - `https://ai-eng-wolraad.onrender.com` (top_k=5)

| # | Question | Expected | Service answer | Retrieval hit (top-5) | Faithful | Correct |
|---|---|---|---|---|---|---|
| 1 | What shareholding makes a shareholder a relevant person? | 10% or more of the issued shares | A shareholder becomes a relevant person if they hold 10% or more of the issued shares, either as an individual or a corporate shareholder [CDD-1.2]. | ✅ | ✅ | ✅ |
| 2 | What risk ratings can a client entity have? | Low, Medium or High | A client entity can have one of three risk ratings: Low, Medium, or High [CDD-2.1]. | ✅ | ✅ | ✅ |
| 3 | How many corporate layers between the entity and its ultimate beneficial owners force a High rating? | More than two | More than two corporate layers between the entity and its ultimate beneficial owners force a High rating [CDD-2.3]. | ✅ | ✅ | ✅ |
| 4 | How often must High and Medium risk clients be reviewed? | Every 12 months | High and Medium risk clients must be reviewed every 12 months, measured from the last review date [CDD-3.1]. | ✅ | ✅ | ✅ |
| 5 | How often must Low risk clients be reviewed? | Every 24 months | Low risk clients must be reviewed every 24 months [CDD-3.1]. | ✅ | ✅ | ✅ |
| 6 | Within how many days must an event-driven review be completed? | Within 30 days of the Firm becoming aware of the event | An event-driven review must be completed within 30 days of the Firm becoming aware of the event [CDD-3.3]. | ✅ | ✅ | ✅ |
| 7 | Does the appointment of a new director trigger an event-driven review? | Yes | Yes, the appointment of a new director triggers an event-driven review, which must be completed within 30 days of the Firm becoming aware of the event [CDD-3.3]. | ✅ | ✅ | ✅ |
| 8 | Is someone with exactly 25% a beneficial owner? | Yes - the 25% threshold is inclusive | Yes, someone with exactly 25% is considered a beneficial owner [CDD-4.2]. | ✅ | ✅ | ✅ |
| 9 | If a person owns 100% of a company that holds 45% of the client, what is their indirect ownership? | 45% | Their indirect ownership is 45%. Indirect ownership is calculated by multiplying percentages through each layer, so 100% of a company that holds 45% results in 45% indirect ownership [CDD-4.3]. | ✅ | ✅ | ✅ |
| 10 | Is a driving licence acceptable as the only identity document? | No - passport or national ID card with photo | No, a driving licence is not accepted as the sole identity document. [CDD-5.1] | ✅ | ✅ | ✅ |
| 11 | If an identity document expires soon after the review date, when must a renewed document be requested? | If it expires within 90 days after the review date | If an identity document expires within 90 days after the review date, the renewed document must be requested as an outstanding item [CDD-5.2]. | ✅ | ✅ | ✅ |
| 12 | How recent must the certification on a copy of an identity document be? | Dated within 3 months before the Firm receives the copy | The certification on a copy of an identity document must be dated within 3 months before the Firm receives the copy [CDD-5.3]. | ✅ | ✅ | ✅ |
| 13 | Who can certify a copy of an identity document? | Attorney-at-law, notary public, chartered accountant, bank officer, or officer of a regulated CSP | A copy of an identity document can be certified by an attorney-at-law, notary public, chartered accountant, bank officer, or an officer of a regulated corporate services provider. The certification must include the certifier's name, capacity, signature, and date, and must be dated within 3 months before the Firm receives the copy. [CDD-5.3] | ✅ | ✅ | ✅ |
| 14 | Is a mobile phone bill acceptable proof of address? | No - utility bills must be electricity, water or landline | No, a mobile phone bill is not acceptable proof of address. [CDD-6.1] | ✅ | ✅ | ✅ |
| 15 | Is a PO Box acceptable as a residential address? | No | No, a PO Box is not acceptable as a residential address. The proof of address document must show the person's name and residential address, and a PO Box is specifically stated as not acceptable [CDD-6.1]. | ✅ | ✅ | ✅ |
| 16 | How recent must proof of address be for a new relevant person? | Dated within 3 months before receipt | For a new relevant person, the proof of address must be dated within 3 months before receipt [CDD-6.2]. | ✅ | ✅ | ✅ |
| 17 | At a periodic review, how old can an existing proof of address be and still be accepted? | Within 12 months of the review date, if the address matches the registers | At a periodic review, an existing proof of address is acceptable if it is dated within 12 months of the review date and the address on it matches the registers [CDD-6.3]. | ✅ | ✅ | ✅ |
| 18 | What documents must the Firm hold for a corporate shareholder of 10% or more? | Certificate of incorporation, register of directors, register of members, certificate of good standing | For a corporate shareholder of 10% or more, the Firm must hold the following documents: the certificate of incorporation, the register of directors, the register of members, and a certificate of good standing (or equivalent) [CDD-7.1]. | ✅ | ✅ | ✅ |
| 19 | How recent must a certificate of good standing be? | Within 6 months of the review date | A certificate of good standing must be dated within 6 months of the review date [CDD-7.2]. | ✅ | ✅ | ✅ |
| 20 | How recent must a corporate shareholder's register of directors and register of members be? | Within 12 months of the review date | The corporate shareholder's register of directors and register of members must be dated within 12 months of the review date [CDD-7.3]. | ✅ | ✅ | ✅ |
| 21 | What is needed when a person's name differs because of marriage? | A certified copy of the linking document, e.g. a marriage certificate | When a person's name differs because of marriage, a certified copy of the linking document, such as a marriage certificate, is required. The certification must follow the guidelines in CDD-5.3 [CDD-8.2]. | ✅ | ✅ | ✅ |
| 22 | When must a new director be screened? | Before their appointment | A new director must be screened before their appointment as a director. If the appointment occurs without prior screening, it is considered a breach and must be reported to the MLRO [CDD-9.2]. | ✅ | ✅ | ✅ |
| 23 | Within how many days must the beneficial ownership register be updated after a change? | Within 30 days of the Firm becoming aware of the change | The beneficial ownership register must be updated within 30 days of the Firm becoming aware of a change in beneficial ownership [CDD-10.1]. | ✅ | ✅ | ✅ |
| 24 | Above what acquisition amount must source of funds be corroborated with documentary evidence? | A single acquisition over US$500,000 (or a High-rated client) | The source of funds must be corroborated with documentary evidence if a single acquisition exceeds US$500,000 or if the client is rated as High [CDD-10.3]. | ✅ | ✅ | ✅ |
| 25 | What is the standard response period for a consolidated client request? | 30 days | The standard response period for a consolidated client request is 30 days [CDD-11.1]. | ✅ | ✅ | ✅ |
| 26 | What happens to an item outstanding for more than 90 days after it was first requested? | Escalated to the MLRO, who decides whether to restrict services | An item outstanding for more than 90 days after it was first requested must be escalated to the MLRO, who decides whether to restrict services [CDD-11.2]. | ✅ | ✅ | ✅ |
| 27 | What statuses can an item take in a review record? | OK, Missing, Expired, Stale | An item in a review record can take one of four statuses: OK (meets the rule), Missing (not on file), Expired (past its expiry date), or Stale (older than the rule allows) [CDD-12.2]. | ✅ | ✅ | ✅ |
| 28 | Who must approve a review prepared with automated assistance? | A Compliance Analyst | A review prepared with automated assistance must be approved by a Compliance Analyst [CDD-12.4]. | ✅ | ✅ | ✅ |
| 29 | What is the Firm's policy on onboarding cryptocurrency exchange clients? | Refuse (not in docs) | I don't know based on the provided documents. | N/A | ✅ | ✅ |
| 30 | What is the name of the Firm's current MLRO? | Refuse (not in docs) | I don't know based on the provided documents. | N/A | ✅ | ✅ |

**Summary:** retrieval hit 28/28 · faithful 30/30 · correct 30/30 (including 2 refusal questions).

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

## Session 3: ADK review agent

This is a single Google ADK agent, `cdd_agent/`, that drafts a CDD periodic review for one case and then stops for analyst approval. It is built on the patterns in `ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py`, minus the router: the job is a single pipeline, so there is nothing to route.

- **Job:** given a case ID, the agent:
  1. loads `cases/<id>/00_file_index.json` and the prior review
  2. looks up the rules it needs
  3. gives every item a status of OK, Missing, Expired or Stale (CDD-12.2)
  4. saves `cases/<id>/review_record_draft.json` with status `PENDING_ANALYST_APPROVAL` (CDD-12.4)
- **Tools** (`cdd_agent/tools.py`): `read_file_index`, `get_prior_review`, `search_policy`, `save_review_record`.
  - **`search_policy` is the real tool.** It reuses the Session 2 retrieval in `rag.py`, the same path as `POST /ask`: an OpenAI embedding, then Pinecone (exact rule IDs plus vector search), then the reranker, then linked rules. It returns each rule's text and parameters.
  - **Failures come back as data.** An empty query, an OpenAI or Pinecone failure (e.g. a 401 for a bad key) or no matches returns `{"error": ...}`, which the model sees as an observation, so the run doesn't crash. A requested rule that isn't in the index is listed in `missing_rule_ids`.
  - **It needs** `OPENAI_API_KEY` and `PINECONE_API_KEY` (already in `.env`), and the CDD policy ingested with `ingest_policy_rules.py`.
- **Step limit:** `MAX_LLM_CALLS = 12` in `cdd_agent/agent.py`, enforced in two places:
  - a `before_model_callback` that ends the run with a message (this also applies in `adk web`)
  - `RunConfig(max_llm_calls=...)` as a backstop
- **Model:** `gemini-3.5-flash` by default; override it with `CDD_AGENT_MODEL` in `.env`. The sample's `gemini-2.5-flash` returns 404 for new API keys.
- **Env:** `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI=FALSE` in this folder's `.env`. Install ADK with `python -m pip install google-adk`.

Run it in a terminal; it prints a THINK / ACT / OBSERVE trace:

```bash
python run_cdd_agent.py CASE-2026-0147
```

Or use ADK's browser UI and pick `cdd_agent` from the list (the `adk` command isn't on PATH with system Python):

```bash
python -m google.adk.cli web .
```

`cases/CASE-2026-0147/` is a made-up sample case. `EXPECTED.md` in that folder is its answer key.

`cdd_agent_run_log.txt` is one successful run (2026-11-10, `gemini-3.5-flash`). It took 7 model turns and 4 `search_policy` calls to Pinecone, and its 10 item statuses match `EXPECTED.md`. The log maps ADK events to labels like this:

| Label | ADK event |
|---|---|
| THINK | a thought part, or plan text before a tool call |
| ACT | a `function_call` part |
| OBSERVE | a `function_response` part |
| ANSWER | text in the final response |

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
├── cdd_agent/           # Session 3 ADK agent (agent.py = root_agent, tools.py)
├── run_cdd_agent.py     # Runs the agent on one case with a Think/Act/Observe trace
├── cdd_agent_run_log.txt  # One successful agent run
├── cases/               # Sample case files for the agent
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
