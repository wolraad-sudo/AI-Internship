"""Extract each rule's test parameters once, at indexing time, for a human to check (architecture doc, step 4).

An LLM turns every rule that contains a test into named parameters, e.g.
CDD-6.3 -> {"max_age_months": 12, "refresh_max_age_months": 3, "must_match_registers": true}.
The result is written to cdd_rule_params.json with "reviewed": false. Check every value against the policy,
fix anything wrong, then set "reviewed": true (and reviewed_by / reviewed_on) before re-running
ingest_policy_rules.py, which stores the parameters beside each chunk.

Run:
    python extract_rule_params.py "C:\\path\\to\\CDD_Periodic_Review_Policy_v3.2.md"
"""

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from ingest_policy_rules import DEFAULT_PARAMS_FILE, front_matter, parse_rules

load_dotenv(Path(__file__).resolve().parent / ".env")

EXTRACTION_MODEL = "gpt-4o"

INSTRUCTIONS = """You extract the testable parameters from one rule of a compliance policy, so code can apply the
rule without re-reading it.

- A parameter is a number, threshold, time limit or yes/no condition the rule tests, e.g. "dated within 12 months"
  -> max_age_months: 12; "25% or more" -> min_ownership_pct: 25; "must match the registers" -> must_match_registers: true.
- Name parameters in snake_case and end numeric names with their unit: _months, _days, _hours, _pct, _usd, _count.
- Use only values stated in the rule. Do not infer values from other rules or general knowledge.
- If the rule has no testable value (a definition, a list of documents, a description), set has_test to false
  and return no parameters.

Example: "At a periodic review, an existing proof of address is acceptable if it is dated within 12 months of the
review date and the address on it matches the registers. Otherwise a new proof of address, dated within 3 months,
must be requested." -> max_age_months: 12, must_match_registers: true, refresh_max_age_months: 3."""


class RuleParameter(BaseModel):
    name: str
    value: float | bool | str


class RuleParameters(BaseModel):
    has_test: bool
    parameters: list[RuleParameter]


def clean(value: float | bool | str) -> float | int | bool | str:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def numbers_missing_from_text(params: dict, text: str) -> list[str]:
    """Code check before the human check: every number should appear in the rule text."""

    digits = text.replace(",", "")
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "twelve": 12}
    found = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", digits)}
    found |= {float(v) for w, v in words.items() if re.search(rf"\b{w}\b", text.lower())}
    return [
        f"{name}={value}"
        for name, value in params.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) not in found
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_PARAMS_FILE)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    markdown = args.policy.read_text(encoding="utf-8")
    rules = parse_rules(markdown)
    client = OpenAI()

    extracted: dict[str, dict] = {}
    flags: dict[str, list[str]] = {}
    for rule in rules:
        completion = client.chat.completions.parse(
            model=EXTRACTION_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"{rule['rule_id']} {rule['rule_title']}\n\n{rule['body']}"},
            ],
            response_format=RuleParameters,
        )
        result = completion.choices[0].message.parsed
        if not result.has_test or not result.parameters:
            print(f"{rule['rule_id']:<9} -")
            continue
        params = {p.name: clean(p.value) for p in result.parameters}
        extracted[rule["rule_id"]] = params
        missing = numbers_missing_from_text(params, rule["body"])
        if missing:
            flags[rule["rule_id"]] = missing
        print(f"{rule['rule_id']:<9} {json.dumps(params)}" + (f"   CHECK: {', '.join(missing)} not in rule text" if missing else ""))

    output = {
        "policy": front_matter(markdown, "title"),
        "policy_version": front_matter(markdown, "version"),
        "policy_effective": front_matter(markdown, "effective"),
        "extracted_by": EXTRACTION_MODEL,
        "reviewed": False,
        "reviewed_by": "",
        "reviewed_on": "",
        "flags_for_review": flags,
        "rules": extracted,
    }
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {len(extracted)} rules with parameters to {args.out} ({len(flags)} flagged for a closer look).")
    print('Check every value against the policy, then set "reviewed": true before re-ingesting.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
