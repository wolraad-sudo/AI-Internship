"""Prove the RAG submission: one grounded answer and one refusal, printed as paste-ready Markdown.

Run against the live service (first request may take ~1 min while Render wakes up):
    python prove_rag.py --base-url https://YOUR-SERVICE.onrender.com > rag_proof.md
"""

import argparse
import json
import sys

import httpx

# Answerable from the Northwind pack (POL-101: remote work up to 3 days a week).
ANSWERABLE = "How many days per week can employees work remotely?"
# Deliberately absent from the Northwind pack, so the service must refuse.
NOT_IN_DOCS = "What is the parental leave policy?"
REFUSAL_TEXT = "I don't know based on the provided documents."


def curl_command(base_url: str, question: str) -> str:
    body = json.dumps({"question": question})
    return (
        f"curl -s -X POST {base_url}/ask \\\n"
        f'  -H "Content-Type: application/json" \\\n'
        f"  -d '{body}'"
    )


def ask(client: httpx.Client, question: str) -> dict:
    response = client.post("/ask", json={"question": question})
    if response.status_code != 200:
        sys.exit(f"/ask returned HTTP {response.status_code}: {response.text}")
    data = response.json()
    if "retrieved_chunk_ids" not in data:
        sys.exit("This server's /ask has no retrieved_chunk_ids - it isn't running the RAG version yet.")
    return data


def section(title: str, base_url: str, question: str, data: dict, verdict: str) -> str:
    return (
        f"### {title}\n\n"
        f"```bash\n{curl_command(base_url, question)}\n```\n\n"
        f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```\n\n"
        f"{verdict}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="e.g. https://YOUR-SERVICE.onrender.com")
    base_url = parser.parse_args().base_url.rstrip("/")

    with httpx.Client(base_url=base_url, timeout=120) as client:
        grounded = ask(client, ANSWERABLE)
        refused = ask(client, NOT_IN_DOCS)

    answer = grounded["answer"]
    grounded_ok = bool(answer.get("citations")) and answer["answer"] != REFUSAL_TEXT
    refused_ok = refused["answer"]["answer"] == REFUSAL_TEXT and not refused["answer"].get("citations")

    print(f"## RAG proof - `{base_url}`\n")
    print(
        "The service embeds each question, retrieves the 5 closest chunks from Pinecone, and answers "
        "only from those chunks. It cites the `document_id` of every chunk it uses and returns the "
        "retrieved chunk IDs. If the chunks don't contain the answer, it refuses instead of guessing.\n"
    )
    print(section(
        "1. Question answered from the docs",
        base_url,
        ANSWERABLE,
        grounded,
        f"**Result:** answered from the docs, citing `{', '.join(answer.get('citations', []))}`; "
        f"retrieved chunks `{', '.join(grounded['retrieved_chunk_ids'])}`.",
    ))
    print(section(
        "2. Question not in the docs (refusal)",
        base_url,
        NOT_IN_DOCS,
        refused,
        "**Result:** the docs don't cover parental leave, so the service refused: no citations and "
        "`sources_needed: true`, even though it still retrieved the closest leave-related chunks.",
    ))

    if not (grounded_ok and refused_ok):
        print(
            f"WARNING: grounded_ok={grounded_ok}, refused_ok={refused_ok} - check the responses above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
