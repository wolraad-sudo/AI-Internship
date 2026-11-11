"""Tools for the CDD review agent.

ADK turns each plain function into a tool. The model only sees the function name, the type hints and
the docstring, so the docstrings are written for the model. (Same pattern as demo1_routing.py.)

search_policy is the real tool: it reuses the Session 2 retrieval in rag.py (OpenAI embedding ->
Pinecone exact rule fetch + vector search -> Pinecone reranker -> linked rules), the same path as POST /ask.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

WEEK1_DIR = Path(__file__).resolve().parent.parent
CASES_DIR = WEEK1_DIR / "cases"
if str(WEEK1_DIR) not in sys.path:  # So `import rag` works under `adk web` as well as run_cdd_agent.py.
    sys.path.insert(0, str(WEEK1_DIR))

MAX_TOP_K = 20  # Same cap as POST /ask.

CASE_ID_PATTERN = re.compile(r"^CASE-\d{4}-\d{4}$")
RULE_ID_PATTERN = re.compile(r"\bCDD-\d+\.\d+\b")
ALLOWED_STATUSES = {"OK", "Missing", "Expired", "Stale"}  # CDD-12.2
REVIEW_STATUS = "PENDING_ANALYST_APPROVAL"  # CDD-12.4: the agent never approves.


def _case_dir(case_id: str) -> Path:
    """Resolve a case folder, rejecting anything that isn't a plain case ID (no path tricks)."""
    if not CASE_ID_PATTERN.fullmatch(case_id or ""):
        raise ValueError(f"Invalid case_id {case_id!r}; expected e.g. CASE-2026-0147")
    folder = CASES_DIR / case_id
    if not folder.is_dir():
        raise FileNotFoundError(f"No case folder for {case_id}")
    return folder


def read_file_index(case_id: str) -> dict:
    """Load the case's 00_file_index.json: client, risk rating, review date, relevant persons and the
    files on record (document type, issue/expiry/certification dates). Call this first."""
    try:
        return json.loads((_case_dir(case_id) / "00_file_index.json").read_text(encoding="utf-8"))
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}


def get_prior_review(case_id: str) -> dict:
    """Load the previous review record for the case: item statuses and outstanding items with the date
    each was first requested. Returns an error if there is no prior review."""
    try:
        folder = _case_dir(case_id)
        index = json.loads((folder / "00_file_index.json").read_text(encoding="utf-8"))
        name = index.get("prior_review_file")
        if not name:
            return {"error": f"No prior review listed for {case_id}"}
        return json.loads((folder / Path(name).name).read_text(encoding="utf-8"))
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}


def search_policy(query: str, top_k: int = 8) -> dict:
    """Search the CDD Periodic Review Policy (v3.2) stored in the Pinecone vector store and return the
    matching rules' exact text and parameters. Use it before judging any item, so age limits and outcomes
    come from the policy, not memory. Rule IDs in the query (e.g. "CDD-5.2 CDD-6.3") are fetched exactly;
    other words are searched by meaning (e.g. "proof of address age limit"). Linked rules and definitions
    are added automatically. If a rule appears in missing_rule_ids, it is not in the index."""
    query = (query or "").strip()
    if not query:
        return {"error": "query is empty; pass rule IDs or a short description of what you need"}
    requested = list(dict.fromkeys(RULE_ID_PATTERN.findall(query)))
    top_k = min(max(int(top_k or 8), len(requested), 1), MAX_TOP_K)

    try:
        import rag  # Imported here so a missing key or package comes back as an observation, not a crash.

        vectors, tokens = rag.embed([query])
        primary, expanded, reranked = rag.retrieve(query, vectors[0], top_k=top_k, rerank=True)
    except Exception as exc:  # OpenAI or Pinecone down, key missing, index not created...
        return {"error": f"Policy search failed: {type(exc).__name__}: {str(exc)[:300]}"}

    rules = []
    for chunk in primary + expanded:
        meta = chunk["metadata"]
        rules.append({
            "rule_id": meta.get("rule_id") or chunk["document_id"],
            "title": meta.get("rule_title"),
            "text": chunk["text"],
            "parameters": meta.get("parameters"),
            "references": meta.get("references", []),
            "found_via": chunk["via"],
        })
    found = {r["rule_id"] for r in rules}
    if not rules:
        return {"error": f"No policy rules matched {query!r}. Is the policy ingested into Pinecone?"}
    return {
        "source": f"pinecone:{rag.PINECONE_INDEX_NAME}/{rag.PINECONE_NAMESPACE}",
        "reranked": reranked,
        "embedding_tokens": tokens,
        "rules": rules,
        "missing_rule_ids": [rid for rid in requested if rid not in found],
    }


def save_review_record(case_id: str, record: dict) -> dict:
    """Save the drafted review record for analyst approval. `record` must contain `items` (a list, each
    with person_id, requirement, rule, status in OK/Missing/Expired/Stale, and reason), plus `risk_rating`
    and `summary`. The status is always set to PENDING_ANALYST_APPROVAL. Call this once, at the end."""
    try:
        folder = _case_dir(case_id)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}

    items = record.get("items")
    if not isinstance(items, list) or not items:
        return {"error": "record.items must be a non-empty list"}
    bad = [i for i, item in enumerate(items) if not isinstance(item, dict) or item.get("status") not in ALLOWED_STATUSES]
    if bad:
        return {"error": f"items {bad} have a status outside {sorted(ALLOWED_STATUSES)}; fix and save again"}

    saved = {
        **record,
        "case_id": case_id,
        "status": REVIEW_STATUS,  # Overrides anything the model wrote.
        "prepared_by": "cdd_review_agent (automated assistance)",
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = folder / "review_record_draft.json"
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return {"saved": str(path.relative_to(WEEK1_DIR)), "status": REVIEW_STATUS, "items": len(items)}
