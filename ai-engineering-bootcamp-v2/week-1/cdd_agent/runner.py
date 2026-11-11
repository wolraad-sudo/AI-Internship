"""Run the CDD review agent once and collect what happened. Shared by run_cdd_agent.py and POST /agent.

How ADK events map to Think / Act / Observe:
    THINK   a model "thought" part (Gemini thought summary), or plan text before a tool call
    ACT     a function_call part: the model chose a tool and its arguments
    OBSERVE a function_response part: what the tool returned (errors included), fed back to the model
    ANSWER  text in the final response (event.is_final_response())
"""

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types

from .agent import MAX_LLM_CALLS, MODEL, root_agent

APP_NAME = "cdd_review"
USER_ID = "analyst1"

# Env var names whose values must never leave the process (in a log line or an API response).
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)


def redact(text: str) -> str:
    """Replace any secret env value that slipped into text (e.g. echoed in an API error) with ***."""
    for name, value in os.environ.items():
        if SECRET_NAME.search(name) and value and len(value) >= 8:
            text = text.replace(value, "***")
    return text


def short(value, limit: int = 300) -> str:
    """One-line, redacted preview of a tool argument or result."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = redact(" ".join(text.split()))
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


def describe_error(exc: BaseException) -> str:
    """ADK wraps model API errors (404, 503...) in its own node errors; report the root cause, redacted."""
    root = exc
    while (root.__cause__ or root.__context__) and not isinstance(root, genai_errors.APIError):
        root = root.__cause__ or root.__context__
    return redact(f"{type(root).__name__}: {str(root)[:300]}")


@dataclass
class AgentRun:
    answer: str = "(no final response)"
    model: str = MODEL
    model_turns: int = 0
    steps: list[dict] = field(default_factory=list)  # One per tool call: tool, args, observation.


async def run_agent(
    message: str,
    on_line: Callable[[str], None] | None = None,
    preview_chars: int = 300,
) -> AgentRun:
    """Send one message to the agent and run it to its final answer (or the step limit).
    on_line, if given, receives each THINK / ACT / OBSERVE / ANSWER log line as it happens."""
    log = on_line or (lambda _line: None)
    service = InMemorySessionService()  # Fresh session per run: no memory between requests.
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id=USER_ID)
    content = types.Content(role="user", parts=[types.Part(text=message)])
    # Backstop: ADK raises LlmCallsLimitExceededError past this, even if the agent's callback were removed.
    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS + 2)

    run = AgentRun()
    pending: dict[str, dict] = {}  # function_call id -> step waiting for its observation
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=content, run_config=run_config
    ):
        if not (event.content and event.content.parts):
            continue
        if not all(p.function_response for p in event.content.parts):  # Tool results alone aren't a turn.
            run.model_turns += 1
        n = run.model_turns
        for part in event.content.parts:
            if part.thought and part.text:
                log(f"\n[{n:02}] THINK   {short(part.text, 600)}")
            elif part.function_call:
                call = part.function_call
                step = {"step": len(run.steps) + 1, "tool": call.name, "args": short(call.args, preview_chars),
                        "observation": None}
                run.steps.append(step)
                pending[call.id or call.name] = step
                log(f"[{n:02}] ACT     {call.name}({short(call.args)})")
            elif part.function_response:
                resp = part.function_response
                observation = short(resp.response, preview_chars)
                step = pending.pop(resp.id or resp.name, None)
                if step is not None:
                    step["observation"] = observation
                log(f"[{n:02}] OBSERVE {resp.name} -> {short(resp.response)}")
            elif part.text:
                label = "ANSWER " if event.is_final_response() else "THINK  "
                log(f"[{n:02}] {label} {short(part.text, 600)}")
        if event.is_final_response():
            run.answer = redact("".join(p.text or "" for p in event.content.parts if not p.thought))
    return run
