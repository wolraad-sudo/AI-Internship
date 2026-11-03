"""Session 2 — minimal Streamlit UI over the live RAG API (/ingest and /ask).

The API is the source of truth: this page only sends requests and displays responses.

Run:
    $env:RAG_API_URL = "https://ai-eng-wolraad.onrender.com"; python -m streamlit run rag_ui.py
Without RAG_API_URL it defaults to the local server; the sidebar can override either.
"""

import json
import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
# Render's free tier sleeps when idle; the first request can take ~1 minute to wake it.
TIMEOUT_SECONDS = 120


def call_api(method: str, base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    """One HTTP call; returns (status, json) and turns network failures into a readable error."""

    try:
        response = httpx.request(method, f"{base_url.rstrip('/')}{path}", json=payload, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return 0, {"detail": f"Could not reach {base_url}: {exc}"}
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"detail": response.text[:500] or "Empty response"}


def show_error(status: int, data: dict) -> None:
    label = f"HTTP {status}" if status else "Network error"
    st.error(f"{label}: {data.get('detail', data)}")


st.set_page_config(page_title="Northwind RAG", page_icon="📚", layout="centered")
st.title("Northwind RAG")
st.caption("Ingest documents and ask grounded questions. All retrieval and generation happen in the API.")

with st.sidebar:
    st.header("API")
    base_url = st.text_input("API base URL", DEFAULT_API_URL, key="base_url", help="Set RAG_API_URL to change the default.")
    if st.button("Check Pinecone health", key="health"):
        with st.spinner("Calling /health/pinecone (may take a minute if Render is waking up)..."):
            status, data = call_api("GET", base_url, "/health/pinecone")
        if status == 200 and data.get("ok"):
            st.success(f"OK: {data.get('namespace_vector_count', 0)} chunks in `{data.get('index_name')}`")
        else:
            show_error(status, data)

ask_tab, ingest_tab = st.tabs(["Ask", "Ingest"])

with ask_tab:
    with st.form("ask"):
        question = st.text_input("Question", key="question", placeholder="How many days per week can employees work remotely?")
        col_k, col_model = st.columns(2)
        top_k = col_k.slider("Chunks to retrieve (top_k)", 1, 20, 5, key="top_k")
        model = col_model.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], key="model")
        asked = st.form_submit_button("Ask", type="primary", key="ask_btn")

    if asked:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving and answering..."):
                status, data = call_api(
                    "POST", base_url, "/ask", {"question": question, "top_k": top_k, "model": model}
                )
            if status != 200:
                show_error(status, data)
            else:
                answer = data["answer"]
                citations = answer.get("citations") or []
                if answer.get("sources_needed") and not citations:
                    st.warning(f"**Refused: not in the documents.**\n\n{answer['answer']}")
                else:
                    st.success(answer["answer"])
                    st.markdown("**Cited documents:** " + " ".join(f"`{doc_id}`" for doc_id in citations))

                if "retrieved_chunk_ids" in data:
                    st.markdown(
                        "**Retrieved chunks:** "
                        + (" ".join(f"`{chunk_id}`" for chunk_id in data["retrieved_chunk_ids"]) or "none")
                    )
                else:
                    st.info("This API's /ask returned no retrieved_chunk_ids — it may not be the RAG version.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Confidence", f"{answer.get('confidence', 0):.2f}")
                m2.metric("Tokens", data.get("tokens_used", 0))
                m3.metric("Cost (USD)", f"${data.get('cost_usd', 0):.6f}")
                m4.metric("Latency", f"{data.get('latency_ms', 0)} ms")

                with st.expander("Full JSON response"):
                    st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")

with ingest_tab:
    with st.form("ingest"):
        document_id = st.text_input("document_id", key="document_id", placeholder="POL-101", help="Re-using an id replaces that document.")
        source = st.text_input("Source (optional)", key="source", placeholder="handbook.txt")
        text = st.text_area("Text", height=250, key="text", placeholder="Paste the document text here...")
        ingested = st.form_submit_button("Ingest", type="primary", key="ingest_btn")

    if ingested:
        payload = {"document_id": document_id, "text": text}
        if source.strip():
            payload["source"] = source.strip()
        # Empty fields still go to the API so its own 400 validation is what you see.
        with st.spinner("Chunking, embedding and upserting..."):
            status, data = call_api("POST", base_url, "/ingest", payload)
        if status != 200:
            show_error(status, data)
        else:
            count = data["chunks_indexed"]
            st.success(f"{data['status'].capitalize()} `{data['document_id']}`: {count} chunk{'' if count == 1 else 's'}")
            st.caption(f"Embedding tokens: {data.get('tokens_used', 0)} · cost ${data.get('cost_usd', 0):.6f}")
            with st.expander("Full JSON response"):
                st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
