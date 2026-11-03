## RAG proof - `https://ai-eng-wolraad.onrender.com`

The service embeds each question, retrieves the 5 closest chunks from Pinecone, and answers only from those chunks. It cites the `document_id` of every chunk it uses and returns the retrieved chunk IDs. If the chunks don't contain the answer, it refuses instead of guessing.

### 1. Question answered from the docs

```bash
curl -s -X POST https://ai-eng-wolraad.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days per week can employees work remotely?"}'
```

```json
{
  "answer": {
    "answer": "Employees may work remotely up to three days per week. [POL-101]",
    "confidence": 1.0,
    "sources_needed": false,
    "citations": [
      "POL-101"
    ]
  },
  "tokens_used": 1179,
  "model": "gpt-4o",
  "latency_ms": 1401,
  "cost_usd": 0.0032,
  "retrieved_chunk_ids": [
    "POL-101#5",
    "POL-101#6",
    "POL-101#4",
    "POL-101#7",
    "POL-101#3"
  ]
}
```

**Result:** answered from the docs, citing `POL-101`; retrieved chunks `POL-101#5, POL-101#6, POL-101#4, POL-101#7, POL-101#3`.

### 2. Question not in the docs (refusal)

```bash
curl -s -X POST https://ai-eng-wolraad.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the parental leave policy?"}'
```

```json
{
  "answer": {
    "answer": "I don't know based on the provided documents.",
    "confidence": 0.0,
    "sources_needed": true,
    "citations": []
  },
  "tokens_used": 1106,
  "model": "gpt-4o",
  "latency_ms": 1016,
  "cost_usd": 0.002935,
  "retrieved_chunk_ids": [
    "POL-101#7",
    "POL-101#9",
    "POL-101#8",
    "POL-101#17",
    "POL-101#1"
  ]
}
```

**Result:** the docs don't cover parental leave, so the service refused: no citations and `sources_needed: true`, even though it still retrieved the closest leave-related chunks.

