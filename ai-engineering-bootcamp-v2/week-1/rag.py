"""Session 2 — Pinecone vector store layer: chunk, embed, upsert, query, health check.

Run `python rag.py` to print the Pinecone health check without starting the server.
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Same .env as main.py, so this module also works when run on its own.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Fixed, not env-configurable: ingest and query must use the same model,
# and the index dimension is tied to it.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Per-1K-token USD for embeddings (OpenAI list price: $0.02 / 1M tokens).
EMBEDDING_PRICE_PER_1K = 0.00002

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "session2-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Chunking — ~800 chars keeps a paragraph or two per chunk; overlap keeps boundary sentences findable.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be >= 0 and smaller than CHUNK_SIZE ({CHUNK_SIZE})")

# Batch size for embedding and upsert calls — well under both APIs' request limits.
BATCH_SIZE = 100

# Reranking — fetch a wider candidate pool by vector similarity, then let a cross-encoder
# (hosted by Pinecone, same API key) re-order it by reading question and chunk together.
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

# Hybrid retrieval for rule-based policies (KYC architecture doc, Session 2 step 3).
# Rule IDs in a question ("get CDD-5.3") are fetched directly instead of searched for.
RULE_ID_PATTERN = re.compile(r"\b[A-Z]{2,}-\d+\.\d+\b")
# Definitions every answer about a policy needs, added whenever that policy's rules are retrieved.
POLICY_DEFINITIONS = {"CDD": ["CDD-1.3", "CDD-1.4"]}  # Review date; month calculations.
# Metadata fields /ask and /debug/retrieve may filter on.
FILTERABLE_FIELDS = {"topic", "review_context", "applies_to", "rule_id", "document_id", "policy_version"}

openai_client = OpenAI()  # Reads OPENAI_API_KEY from the environment.


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    """Create the Pinecone client on first use, so the app still starts without a key."""

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=api_key)


@lru_cache(maxsize=1)
def get_index():
    """Return the index handle, creating a serverless index the first time it is needed."""

    pc = get_pinecone()
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed texts in batches; returns vectors (same order) plus tokens used for cost tracking."""

    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), BATCH_SIZE):
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts[start : start + BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in response.data)
        tokens += response.usage.total_tokens
    return vectors, tokens


def embedding_cost_usd(tokens: int) -> float:
    """Embeddings are billed on input tokens only."""

    return tokens / 1000 * EMBEDDING_PRICE_PER_1K


def chunk_text(text: str) -> list[str]:
    """Split on paragraphs, then lines, then words — so chunks break at natural boundaries."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def delete_stale_chunks(document_id: str, keep: int) -> int:
    """
    After re-ingesting, drop chunks numbered `keep` and above — left over when the new text is shorter.
    Runs after the upsert, so the document is never missing from the index mid-update.
    """

    index = get_index()
    deleted = 0
    for page in index.list(prefix=f"{document_id}#", namespace=PINECONE_NAMESPACE):
        stale = [
            item.id
            for item in page.vectors
            if item.id.rsplit("#", 1)[1].isdigit() and int(item.id.rsplit("#", 1)[1]) >= keep
        ]
        if stale:
            index.delete(ids=stale, namespace=PINECONE_NAMESPACE)
            deleted += len(stale)
    return deleted


def upsert_chunks(
    document_id: str,
    chunks: list[str],
    vectors: list[list[float]],
    source: str,
    extra_metadata: dict | None = None,
) -> int:
    """Store chunks under ids like `handbook#0`, `handbook#1` so they can be found and replaced per document."""

    records = [
        {
            "id": f"{document_id}#{i}",
            "values": vector,
            "metadata": {
                **(extra_metadata or {}),
                # Core fields last, so caller metadata can't overwrite them.
                "text": chunk,
                "document_id": document_id,
                "chunk_index": i,
                "source": source,
            },
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    get_index().upsert(
        vectors=records, namespace=PINECONE_NAMESPACE, batch_size=BATCH_SIZE, show_progress=False
    )
    return len(records)


def to_chunk(chunk_id: str, metadata: dict | None, score: float | None, via: str) -> dict:
    """One retrieved chunk. `via` says how it was found: search, rule id, reference, definition."""

    metadata = dict(metadata or {})
    chunk = {
        "id": chunk_id,
        "score": round(score, 4) if score is not None else None,  # Cosine similarity: higher is closer.
        "via": via,
        "document_id": metadata.pop("document_id", ""),
        "chunk_index": int(metadata.pop("chunk_index", -1)),
        "source": metadata.pop("source", ""),
        "text": metadata.pop("text", ""),
    }
    if "parameters" in metadata:  # Stored as JSON text because Pinecone metadata can't nest objects.
        metadata["parameters"] = json.loads(metadata["parameters"])
    chunk["metadata"] = metadata  # rule_id, topic, applies_to, references, ... for rule-based documents.
    return chunk


def build_filter(filters: dict[str, str] | None) -> dict | None:
    """{"topic": "address"} -> Pinecone filter. $in matches both plain and list fields (applies_to)."""

    if not filters:
        return None
    return {field: {"$in": [value]} for field, value in filters.items()}


def query_chunks(vector: list[float], top_k: int = 4, filters: dict[str, str] | None = None) -> list[dict]:
    """Nearest chunks for an already-embedded question, optionally narrowed by metadata."""

    result = get_index().query(
        vector=vector,
        top_k=top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
        filter=build_filter(filters),
    )
    return [to_chunk(match.id, match.metadata, match.score, "search") for match in result.matches]


def fetch_documents(document_ids: list[str], via: str) -> list[dict]:
    """Every chunk of the given documents (rule IDs for a policy), in the order asked for."""

    index = get_index()
    chunk_ids = [
        item.id
        for document_id in document_ids
        for page in index.list(prefix=f"{document_id}#", namespace=PINECONE_NAMESPACE)
        for item in page.vectors
    ]
    if not chunk_ids:
        return []
    vectors = index.fetch(ids=chunk_ids, namespace=PINECONE_NAMESPACE).vectors
    return [to_chunk(cid, vectors[cid].metadata, None, via) for cid in chunk_ids if cid in vectors]


def rerank_chunks(question: str, matches: list[dict], top_n: int) -> tuple[list[dict], bool]:
    """
    Re-order vector matches with the reranker and keep the best top_n.
    Returns (chunks, reranked). On any reranker failure (e.g. quota) it falls back to vector order,
    so retrieval degrades instead of breaking /ask.
    """

    if top_n <= 0:
        return [], False
    if len(matches) <= 1:
        return matches[:top_n], False
    try:
        result = get_pinecone().inference.rerank(
            model=RERANK_MODEL,
            query=question,
            documents=[{"id": m["id"], "text": m["text"]} for m in matches],
            top_n=top_n,
            return_documents=False,
        )
    except Exception:
        return matches[:top_n], False

    reranked = []
    for row in result.data:
        chunk = dict(matches[row.index])
        chunk["rerank_score"] = round(row.score, 4)
        reranked.append(chunk)
    return reranked, True


def expand_linked_rules(chunks: list[dict]) -> list[dict]:
    """
    Follow each retrieved rule's `references` one hop (a hit on CDD-8.2 also brings CDD-5.3) and add the
    policy's definitions (CDD-1.3, CDD-1.4). Returns only chunks not already retrieved.
    """

    have = {c["document_id"] for c in chunks}
    wanted: list[tuple[str, str]] = []  # (rule ID, via)
    for chunk in chunks:
        for ref in chunk["metadata"].get("references", []):
            if RULE_ID_PATTERN.fullmatch(ref):  # Section refs like "CDD-5" would pull in whole sections.
                wanted.append((ref, f"reference from {chunk['document_id']}"))
    prefixes = {c["metadata"]["rule_id"].split("-")[0] for c in chunks if c["metadata"].get("rule_id")}
    for prefix in sorted(prefixes):
        wanted += [(rule_id, "definition") for rule_id in POLICY_DEFINITIONS.get(prefix, [])]

    expanded: list[dict] = []
    for rule_id, via in wanted:
        if rule_id in have:
            continue
        have.add(rule_id)
        expanded += fetch_documents([rule_id], via)
    return expanded


def retrieve(
    question: str,
    vector: list[float],
    top_k: int,
    rerank: bool,
    filters: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict], bool]:
    """
    Hybrid retrieval. Returns (top_k chunks, linked chunks added on top, whether reranking ran).
    1. Rule IDs named in the question are fetched exactly and ranked first.
    2. The rest of the top_k comes from vector search (filtered by metadata), optionally reranked.
    3. Linked rules and definitions are added beyond top_k, so the top-k hit rate still measures search.
    """

    exact = fetch_documents(list(dict.fromkeys(RULE_ID_PATTERN.findall(question))), "rule id")
    exact_ids = {c["id"] for c in exact}

    if rerank:
        candidates = query_chunks(vector, top_k=max(top_k, RERANK_CANDIDATES), filters=filters)
        candidates = [c for c in candidates if c["id"] not in exact_ids]
        searched, reranked = rerank_chunks(question, candidates, top_n=max(top_k - len(exact), 0))
    else:
        candidates = query_chunks(vector, top_k=top_k + len(exact), filters=filters)
        searched, reranked = [c for c in candidates if c["id"] not in exact_ids], False

    primary = (exact + searched)[:top_k]
    return primary, expand_linked_rules(primary), reranked


def pinecone_health() -> dict:
    """
    Read-only check that Pinecone is reachable and the index matches our embedding model.
    Never creates the index and never returns the API key.
    """

    status: dict = {
        "ok": False,
        "api_key_set": bool(os.getenv("PINECONE_API_KEY")),
        "index_name": PINECONE_INDEX_NAME,
        "namespace": PINECONE_NAMESPACE,
        "embedding_model": EMBEDDING_MODEL,
        "expected_dimension": EMBEDDING_DIMENSION,
    }
    if not status["api_key_set"]:
        status["error"] = "PINECONE_API_KEY is not set"
        return status

    try:
        pc = get_pinecone()
        status["index_exists"] = pc.has_index(PINECONE_INDEX_NAME)
        if not status["index_exists"]:
            # Reachable and authenticated; the index is created on first ingest.
            status["ok"] = True
            status["note"] = "Pinecone reachable; index will be created on first ingest"
            return status

        stats = pc.Index(PINECONE_INDEX_NAME).describe_index_stats()
        namespace_summary = stats.namespaces.get(PINECONE_NAMESPACE)
        status["index_dimension"] = stats.dimension
        status["total_vector_count"] = stats.total_vector_count
        status["namespace_vector_count"] = namespace_summary.vector_count if namespace_summary else 0
        status["dimension_matches"] = stats.dimension == EMBEDDING_DIMENSION
        status["ok"] = status["dimension_matches"]
        if not status["dimension_matches"]:
            status["error"] = (
                f"Index dimension {stats.dimension} does not match {EMBEDDING_MODEL} "
                f"({EMBEDDING_DIMENSION}); use a different PINECONE_INDEX_NAME"
            )
    except Exception as exc:  # Health checks report failures instead of raising.
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


if __name__ == "__main__":
    import json

    print(json.dumps(pinecone_health(), indent=2))
