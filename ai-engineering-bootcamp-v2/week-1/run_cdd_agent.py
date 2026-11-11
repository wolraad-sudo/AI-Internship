"""Run the CDD review agent on one case and print its Think / Act / Observe trace.

Run (from this folder):
    python run_cdd_agent.py                  # default sample case CASE-2026-0147
    python run_cdd_agent.py CASE-2026-0147

Runner + InMemorySessionService + run_async is the same pattern as ask() in
ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py; this version logs every event
instead of returning only the final text.

How ADK events map to the log:
    THINK   a model "thought" part (Gemini thought summary), or plan text before a tool call
    ACT     a function_call part: the model chose a tool and its arguments
    OBSERVE a function_response part: what the tool returned (errors included), fed back to the model
    ANSWER  text in the final response (event.is_final_response())
"""

import asyncio
import json
import sys

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types

from cdd_agent.agent import MAX_LLM_CALLS, root_agent

APP_NAME = "cdd_review"
USER_ID = "analyst1"


def short(value, limit: int = 300) -> str:
    """One-line preview of a tool argument or result."""
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


async def review(case_id: str) -> str:
    service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id=USER_ID)
    message = types.Content(role="user", parts=[types.Part(text=f"New case file: {case_id}. Prepare the review.")])

    # Backstop: ADK itself raises LlmCallsLimitExceededError past this, even if the callback were removed.
    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS + 2)

    step, final = 0, "(no final response)"
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=message, run_config=run_config
    ):
        if not (event.content and event.content.parts):
            continue
        # A model turn is any event that isn't just tool results coming back.
        if not all(p.function_response for p in event.content.parts):
            step += 1
        for part in event.content.parts:
            if part.thought and part.text:
                print(f"\n[{step:02}] THINK   {short(part.text, 600)}")
            elif part.function_call:
                print(f"[{step:02}] ACT     {part.function_call.name}({short(part.function_call.args)})")
            elif part.function_response:
                print(f"[{step:02}] OBSERVE {part.function_response.name} -> {short(part.function_response.response)}")
            elif part.text:  # Visible text: the final answer, or the model narrating its plan mid-run.
                label = "ANSWER " if event.is_final_response() else "THINK  "
                print(f"[{step:02}] {label} {short(part.text, 600)}")
        if event.is_final_response():
            final = "".join(p.text or "" for p in event.content.parts if not p.thought)

    print(f"\n--- FINAL after {step} model turns (limit {MAX_LLM_CALLS}) ---\n{final}")
    return final


def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-2026-0147"
    try:
        asyncio.run(review(case_id))
    except Exception as exc:  # Model API errors surface wrapped in ADK's node errors; show the root cause.
        root = exc
        while (root.__cause__ or root.__context__) and not isinstance(root, genai_errors.APIError):
            root = root.__cause__ or root.__context__
        print(f"\n--- STOPPED: {type(root).__name__}: {str(root)[:300]}", file=sys.stderr)
        print("Nothing was saved unless an OBSERVE line above shows save_review_record -> saved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
