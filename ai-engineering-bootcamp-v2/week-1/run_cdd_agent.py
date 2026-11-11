"""Run the CDD review agent on one case and print its Think / Act / Observe trace.

Run (from this folder):
    python run_cdd_agent.py                  # default sample case CASE-2026-0147
    python run_cdd_agent.py CASE-2026-0147

Runner + InMemorySessionService + run_async is the same pattern as ask() in
ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py. The loop lives in
cdd_agent/runner.py (shared with POST /agent); see its docstring for the event-to-label mapping.
"""

import asyncio
import sys

from cdd_agent.agent import MAX_LLM_CALLS
from cdd_agent.runner import describe_error, run_agent


def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-2026-0147"
    try:
        run = asyncio.run(run_agent(f"New case file: {case_id}. Prepare the review.", on_line=print))
    except Exception as exc:  # e.g. Gemini 503 after retries
        print(f"\n--- STOPPED: {describe_error(exc)}", file=sys.stderr)
        print("Nothing was saved unless an OBSERVE line above shows save_review_record -> saved.", file=sys.stderr)
        return 1
    print(f"\n--- FINAL after {run.model_turns} model turns (limit {MAX_LLM_CALLS}) ---\n{run.answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
