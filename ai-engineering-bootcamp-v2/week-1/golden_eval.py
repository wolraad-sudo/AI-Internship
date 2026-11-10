"""Golden-set eval for the RAG service: retrieval hit, faithfulness and correctness per question.

Calls the API (it stays the source of truth) and prints a Markdown table for the README.

Run:
    python golden_eval.py --base-url https://ai-eng-wolraad.onrender.com --out golden_results.md
    python golden_eval.py --set cdd --base-url https://ai-eng-wolraad.onrender.com --out cdd_golden_results.md

Checks:
- Retrieval hit: a chunk from the expected document containing the key evidence sentence is in the top 5
  (GET /debug/retrieve, same embedding and top_k as /ask).
- Faithfulness: an LLM judge (gpt-4o-mini, run locally) confirms every claim in the answer is supported
  by the chunks /ask retrieved. A refusal makes no claims, so it is faithful.
- Correctness: the answer contains the expected facts (regex), or for refusal questions is the exact
  refusal text.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

REFUSAL_TEXT = "I don't know based on the provided documents."
TOP_K = 5
JUDGE_MODEL = "gpt-4o-mini"

# Facts verified against the Northwind sample pack. `evidence` is text the right chunk must contain;
# `must_match` are regexes the answer must all match to count as correct.
GOLDEN = [
    {
        "question": "How many days per week can employees work remotely?",
        "expected": "Up to 3 days per week",
        "doc": "POL-101",
        "evidence": "Employees may work remotely up to three days per week",
        "must_match": [r"\b(3|three)\b"],
    },
    {
        "question": "How many days of annual leave do employees get?",
        "expected": "28 days plus public holidays",
        "doc": "POL-101",
        "evidence": "Annual leave is 28 days plus public holidays",
        "must_match": [r"\b28\b"],
    },
    {
        "question": "What is the mileage rate for business travel?",
        "expected": "45p per mile for travel over 50 miles",
        "doc": "POL-114",
        "evidence": "Travel over 50 miles may be claimed at 45 pence per mile",
        "must_match": [r"\b45\s*(p|pence)\b"],
    },
    {
        "question": "How long do I have to submit an expense claim?",
        "expected": "Within 30 days, with a receipt",
        "doc": "POL-114",
        "evidence": "All claims must be submitted within 30 days with a receipt",
        "must_match": [r"\b(30|thirty)\s*days\b"],
    },
    {
        "question": "How quickly must a lost laptop be reported?",
        "expected": "Within 1 hour, to the security desk",
        "doc": "POL-207",
        "evidence": "Lost devices must be reported within one hour",
        "must_match": [r"\b(1|one)\s*hour\b"],
    },
    {
        "question": "What are the password requirements?",
        "expected": "At least 14 characters, rotated every 90 days",
        "doc": "POL-207",
        "evidence": "Passwords must be at least 14 characters and rotated every 90 days",
        "must_match": [r"\b14\b", r"\b90\b"],
    },
    {
        "question": "What is the WB-9 payload limit?",
        "expected": "25 kg",
        "doc": "SPEC-WB9",
        "evidence": "payload capacity of 25 kg",
        "must_match": [r"\b25\s*kg\b"],
    },
    {
        "question": "How long does the WB-9 battery last and how long does it take to recharge?",
        "expected": "8 hours; 90 minute recharge",
        "doc": "SPEC-WB9",
        "evidence": "Battery life is 8 hours on a single charge with a 90 minute recharge",
        "must_match": [r"\b(8|eight)[\s-]*hours?\b", r"\b90\b"],
    },
    {
        "question": "What is the parental leave policy?",
        "expected": "Refuse (not in docs)",
        "doc": None,
    },
    {
        "question": "Do employees get stock options?",
        "expected": "Refuse (not in docs)",
        "doc": None,
    },
]


class Verdict(BaseModel):
    faithful: bool
    reason: str


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def judge_faithfulness(judge: OpenAI, answer: str, chunks: list[dict]) -> Verdict:
    """Ask a separate model whether every claim in the answer is supported by the retrieved chunks."""

    context = "\n\n".join(f"[{c['document_id']} / {c['id']}]\n{c['text']}" for c in chunks)
    completion = judge.chat.completions.parse(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You check RAG answers for faithfulness. faithful=true only if every factual claim in "
                    "the answer is directly supported by the context. Ignore citation formatting. "
                    "Give a one-sentence reason."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"},
        ],
        response_format=Verdict,
    )
    return completion.choices[0].message.parsed


def mark(ok: bool | None) -> str:
    return "N/A" if ok is None else ("✅" if ok else "❌")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.getenv("RAG_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--out", type=Path, help="Also write the Markdown to this file (UTF-8)")
    parser.add_argument(
        "--set", choices=["northwind", "cdd"], default="northwind",
        help="northwind: Northwind sample pack; cdd: CDD Periodic Review Policy v3.2",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which has no ✅.

    judge = OpenAI()
    rows, notes = [], []
    with httpx.Client(base_url=base_url, timeout=120) as api:
        if args.set == "cdd":
            from cdd_golden_set import CDD_GOLDEN as golden
        else:
            golden = GOLDEN
        for n, item in enumerate(golden, start=1):
            question = item["question"]
            should_refuse = item["doc"] is None

            retrieved = api.get("/debug/retrieve", params={"q": question, "top_k": TOP_K}).raise_for_status().json()
            chunks = retrieved["matches"]
            response = api.post("/ask", json={"question": question, "top_k": TOP_K}).raise_for_status().json()
            answer = response["answer"]["answer"]
            citations = response["answer"].get("citations", [])
            refused = answer.strip() == REFUSAL_TEXT

            # /ask also returns linked rules and definitions added on top of the top_k; compare search results only.
            expanded_ids = set(response.get("expanded_chunk_ids") or [])
            searched_ids = [cid for cid in response["retrieved_chunk_ids"] if cid not in expanded_ids]
            if [c["id"] for c in chunks] != searched_ids:
                notes.append(f"Q{n}: /debug/retrieve and /ask retrieved different chunks")

            if should_refuse:
                hit = None
                faithful = refused  # A refusal makes no claims; any other answer here is unsupported.
                correct = refused
                if not refused:
                    notes.append(f"Q{n}: expected a refusal but got: {answer}")
            else:
                evidence = normalise(item["evidence"])
                hit = any(c["document_id"] == item["doc"] and evidence in normalise(c["text"]) for c in chunks)
                if refused:
                    faithful = True
                    notes.append(f"Q{n}: refused although the answer is in {item['doc']}")
                else:
                    # Judge against everything the model saw, including linked rules and definitions.
                    verdict = judge_faithfulness(judge, answer, chunks + retrieved.get("expanded", []))
                    faithful = verdict.faithful
                    if not faithful:
                        notes.append(f"Q{n}: judge says unfaithful - {verdict.reason}")
                correct = not refused and all(re.search(p, answer, re.IGNORECASE) for p in item["must_match"])

            got = answer + (f" [{', '.join(citations)}]" if citations and "[" not in answer else "")
            got = " ".join(got.split())  # Keep multi-line answers on one table row.
            rows.append((n, question, item["expected"], got.replace("|", "\\|"), hit, faithful, correct))

    answerable = [r for r in rows if r[4] is not None]
    lines = []
    lines.append(f"### Golden-set eval ({args.set}) - `{base_url}` (top_k={TOP_K})\n")
    lines.append("| # | Question | Expected | Service answer | Retrieval hit (top-5) | Faithful | Correct |")
    lines.append("|---|---|---|---|---|---|---|")
    for n, question, expected, got, hit, faithful, correct in rows:
        lines.append(f"| {n} | {question} | {expected} | {got} | {mark(hit)} | {mark(faithful)} | {mark(correct)} |")

    lines.append(
        f"\n**Summary:** retrieval hit {sum(bool(r[4]) for r in answerable)}/{len(answerable)} · "
        f"faithful {sum(r[5] for r in rows)}/{len(rows)} · correct {sum(r[6] for r in rows)}/{len(rows)} "
        f"(including {len(rows) - len(answerable)} refusal questions)."
    )
    if notes:
        lines.append("\n**Notes:**\n" + "\n".join(f"- {note}" for note in notes))

    markdown = "\n".join(lines) + "\n"
    print(markdown, end="")
    if args.out:
        args.out.write_text(markdown, encoding="utf-8")

    return 0 if all(r[6] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
