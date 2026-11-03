"""Before/after test for reranking: where does the right chunk land with rerank off vs on?

Uses the API only (GET /debug/retrieve, POST /ask), so it tests exactly what the service does.

Run:
    python rerank_eval.py --base-url https://ai-eng-wolraad.onrender.com --out rerank_results.md
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

from golden_eval import GOLDEN, normalise

TOP_K = 5

# Every Northwind policy has the same header fields, so these questions make header chunks from
# several documents compete — the case vector similarity alone ranks poorly.
HEADER_PROBES = [
    {"question": "Who owns the facilities policy?", "doc": "POL-118", "evidence": "Owner: Head of Workplace"},
    {"question": "When is the security policy next reviewed?", "doc": "POL-207", "evidence": "Next review: 5 July 2026"},
    {"question": "Who approves the expenses policy?", "doc": "POL-114", "evidence": "Approver: CFO"},
    {"question": "What version is the employee handbook?", "doc": "POL-101", "evidence": "Version: 3.2"},
]
CASES = [g for g in GOLDEN if g["doc"]] + HEADER_PROBES

# Answer-level check with a tight top_k, where ranking decides whether the evidence reaches the model.
ANSWER_CHECK = {"question": "Who owns the facilities policy?", "top_k": 3, "expect": "head of workplace"}


def evidence_rank(chunks: list[dict], doc: str, evidence: str) -> int | None:
    wanted = normalise(evidence)
    return next(
        (i for i, c in enumerate(chunks, start=1) if c["document_id"] == doc and wanted in normalise(c["text"])),
        None,
    )


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--out", type=Path, help="Also write the Markdown to this file (UTF-8)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ranks: dict[bool, list[int | None]] = {False: [], True: []}
    latency: dict[bool, list[float]] = {False: [], True: []}
    rows = []
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120) as api:
        api.get("/health/pinecone")  # Wake Render before timing anything.
        for case in CASES:
            row = [case["question"], case["doc"]]
            for rerank in (False, True):
                start = time.perf_counter()
                data = api.get(
                    "/debug/retrieve", params={"q": case["question"], "top_k": TOP_K, "rerank": rerank}
                ).raise_for_status().json()
                latency[rerank].append((time.perf_counter() - start) * 1000)
                if rerank and not data.get("reranked"):
                    sys.exit("Reranking did not run - check the deployed code and Pinecone rerank quota.")
                rank = evidence_rank(data["matches"], case["doc"], case["evidence"])
                ranks[rerank].append(rank)
                row.append(rank)
            rows.append(row)

        answers = {}
        for rerank in (False, True):
            data = api.post(
                "/ask",
                json={"question": ANSWER_CHECK["question"], "top_k": ANSWER_CHECK["top_k"], "rerank": rerank},
            ).raise_for_status().json()
            answers[rerank] = data

    def fmt(rank: int | None) -> str:
        return f"#{rank}" if rank else "not in top 5"

    before, after = summarise(ranks[False]), summarise(ranks[True])
    lines = [f"### Reranking before/after - `{args.base_url}` (top_k={TOP_K})\n"]
    lines.append("| Question | Expected doc | Rank without rerank | Rank with rerank |")
    lines.append("|---|---|---|---|")
    for question, doc, rank_off, rank_on in rows:
        changed = " ⬆️" if (rank_on or 99) < (rank_off or 99) else (" ⬇️" if (rank_on or 99) > (rank_off or 99) else "")
        lines.append(f"| {question} | {doc} | {fmt(rank_off)} | {fmt(rank_on)}{changed} |")

    lines.append("\n| Metric | Without rerank | With rerank |")
    lines.append("|---|---|---|")
    for metric in ("hit@1", "hit@3", "hit@5", "mrr"):
        lines.append(f"| {metric} | {before[metric]:.2f} | {after[metric]:.2f} |")
    lines.append(
        f"| avg /debug/retrieve latency | {sum(latency[False]) / len(latency[False]):.0f} ms | "
        f"{sum(latency[True]) / len(latency[True]):.0f} ms |"
    )

    lines.append(f"\n**Answer check** - \"{ANSWER_CHECK['question']}\" with top_k={ANSWER_CHECK['top_k']}:\n")
    for rerank in (False, True):
        data = answers[rerank]
        answer = data["answer"]["answer"]
        ok = ANSWER_CHECK["expect"] in answer.lower()
        lines.append(
            f"- rerank {'on ' if rerank else 'off'}: {'✅' if ok else '❌'} \"{answer}\" "
            f"(chunks: {', '.join(data['retrieved_chunk_ids'])})"
        )

    markdown = "\n".join(lines) + "\n"
    print(markdown, end="")
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")
    return 0 if after["mrr"] >= before["mrr"] else 1


if __name__ == "__main__":
    sys.exit(main())
