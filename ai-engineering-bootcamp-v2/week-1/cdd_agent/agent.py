"""CDD periodic-review agent (Session 3).

One root agent, no router: the job is a single pipeline, so there is nothing to route.
Patterns copied from ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py
(Agent(name, model, instruction, tools) and plain functions as tools).

`adk web` / `adk run` look for a module-level `root_agent` in this file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .tools import get_prior_review, read_file_index, save_review_record, search_policy

# Same .env as the FastAPI service (GOOGLE_API_KEY, GOOGLE_GENAI_USE_VERTEXAI=FALSE).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# The sample uses gemini-2.5-flash, which Google no longer serves to new API keys (404). On 2026-11-10
# gemini-3.8/3.7-flash returned 503 "high demand" while 3.5-flash answered. Override with CDD_AGENT_MODEL in .env.
MODEL = os.getenv("CDD_AGENT_MODEL", "gemini-3.5-flash")

# Step limit: each model call is one Think -> Act turn. 4 tools plus a few retries fit well inside 12.
MAX_LLM_CALLS = 12
STEP_LIMIT_MESSAGE = (
    f"Stopped: step limit of {MAX_LLM_CALLS} model calls reached before the review was saved. "
    "Nothing was approved. Re-run, or finish the review by hand."
)

INSTRUCTION = """You prepare a draft CDD periodic review for one case. You never approve it.

GOAL
Given a case ID, produce a review record that gives every required item a status of OK, Missing,
Expired or Stale (CDD-12.2), save it, and stop for Compliance Analyst approval (CDD-12.4).

STEPS
1. Call read_file_index(case_id). Use its review_date as "today" for every age calculation.
2. Call get_prior_review(case_id) to find items that were already outstanding and when they were first requested.
3. Before acting, write a short plan: the persons, which documents each needs, and the rule IDs involved.
4. Call search_policy with all the rule IDs you need in one query (e.g. "CDD-2.3 CDD-5.2 CDD-5.3 CDD-6.2
   CDD-6.3 CDD-7.2 CDD-7.3 CDD-8.2 CDD-9.2 CDD-11.2 CDD-12.2 CDD-12.3"). Judge every item by the rule TEXT
   it returns (e.g. a document that is still valid but expires soon is not Expired). Search again only
   for rules listed in missing_rule_ids or that you still need.
5. Build the items: one per person per requirement, with person_id, requirement, rule, status, reason
   (the dates you compared), and file_id when a file exists.
6. Call save_review_record(case_id, record) once, with items, risk_rating (re-rate if a rule forces it,
   and say which), outstanding_items, proposed_outcome (one of the CDD-12.3 outcomes, quoted from the
   policy text), and a two-sentence summary.

CONSTRAINTS
- Use only facts returned by the tools. Never invent documents, dates or people.
- If a tool returns an error, say so and stop; do not guess the missing data.
- Statuses are exactly OK, Missing, Expired or Stale.
- Never call the review approved or complete. Its status is PENDING_ANALYST_APPROVAL.

DONE
You are done when save_review_record has returned "saved". Then reply with: the saved path, a count of
items by status, any risk re-rating, the proposed outcome, and "Awaiting Compliance Analyst approval (CDD-12.4)." Then stop."""


def enforce_step_limit(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    """Runs before every model call. Counts calls for this run and ends the run once the limit is hit.
    Works in `adk web` too, where we don't control RunConfig."""
    calls = callback_context.state.get("temp:llm_calls", 0) + 1
    callback_context.state["temp:llm_calls"] = calls
    if calls > MAX_LLM_CALLS:
        # Returning a response skips the model call; a text-only reply ends the agent loop.
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=STEP_LIMIT_MESSAGE)]))
    return None


root_agent = Agent(
    name="cdd_review_agent",
    # Gemini wrapper only to add retries: 429/503 "high demand" errors are common and usually pass in seconds.
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2, max_delay=30, http_status_codes=[429, 500, 503]),
    ),
    description="Drafts a CDD periodic review record for one case and stops for analyst approval.",
    instruction=INSTRUCTION,
    tools=[read_file_index, get_prior_review, search_policy, save_review_record],
    # include_thoughts=True makes Gemini return thought summaries, which run_cdd_agent.py logs as THINK.
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=True)),
    before_model_callback=enforce_step_limit,
    generate_content_config=types.GenerateContentConfig(temperature=0),
)
