"""Ingest a folder of .txt/.md documents through POST /ingest, then report the vector store size.

Each file's document_id comes from its "Document ID: XXX" header line (e.g. POL-101), falling back
to the filename stem, so re-running the script replaces documents instead of duplicating them.

Run (server must be up):
    python ingest_docs.py                                   # defaults to Downloads\\northwind-sample-docs
    python ingest_docs.py C:\\path\\to\\docs --base-url http://127.0.0.1:8000
"""

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

DEFAULT_FOLDER = Path.home() / "Downloads" / "northwind-sample-docs"
DOC_ID_PATTERN = re.compile(r"^Document ID:\s*(\S+)", re.MULTILINE)
SKIP_FILES = {"readme.md"}  # The pack's README describes the docs; it isn't one.


def stable_document_id(path: Path, text: str) -> str:
    """Prefer the document's own ID; otherwise a slug of the filename."""

    match = DOC_ID_PATTERN.search(text[:2000])
    if match:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    files = sorted(
        p for p in args.folder.glob("*") if p.suffix.lower() in {".txt", ".md"} and p.name.lower() not in SKIP_FILES
    )
    if not files:
        print(f"No .txt or .md files found in {args.folder}")
        return 1

    total_chunks = 0
    total_cost = 0.0
    failures = 0
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        for path in files:
            text = path.read_text(encoding="utf-8")
            document_id = stable_document_id(path, text)
            title = text.strip().splitlines()[0][:200]
            response = client.post(
                "/ingest",
                json={"document_id": document_id, "source": path.name, "text": text, "metadata": {"title": title}},
            )
            if response.status_code != 200:
                failures += 1
                print(f"FAILED  {path.name:<28} {document_id:<10} HTTP {response.status_code}: {response.text}")
                continue
            data = response.json()
            total_chunks += data["chunks_indexed"]
            total_cost += data["cost_usd"]
            print(f"{data['status']:<7} {path.name:<28} {document_id:<10} {data['chunks_indexed']:>3} chunks")

        print(f"\nIngested {total_chunks} chunks from {len(files) - failures}/{len(files)} files (${total_cost:.6f})")

        # Pinecone serverless takes a few seconds to reflect new vectors in its stats.
        health: dict = {}
        for _ in range(15):
            health = client.get("/health/pinecone").json()
            if health.get("namespace_vector_count", 0) >= total_chunks:
                break
            time.sleep(2)
        print(
            f"Vector store '{health.get('index_name')}' / namespace '{health.get('namespace')}': "
            f"{health.get('namespace_vector_count')} chunks (index total {health.get('total_vector_count')})"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
