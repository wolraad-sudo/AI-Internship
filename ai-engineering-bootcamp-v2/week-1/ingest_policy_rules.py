"""Ingest a rule-numbered policy (e.g. CDD_Periodic_Review_Policy_v3.2.md) rule by rule through POST /ingest.

Follows KYC_Refresh_Agent_Architecture.docx, Session 2 ("the policy as a RAG source"):
1. Chunk by rule, not by size: one document per rule, document_id = rule ID, text prefixed with its section
   heading ("CDD-6 Proof of residential address › CDD-6.3 Periodic review"). Tables (CDD-4.1) stay whole.
2. Metadata on every chunk:
   - rule_id         exact lookup and citation
   - topic           short section topic for filtering ("address")
   - applies_to      party roles the rule covers (keyword match, else the policy's CDD-4.1 role table)
   - review_context  periodic / event-driven / new person / all
   - references      rule IDs to pull in one hop: IDs the text mentions plus LINKED_REFERENCES
   - policy_version  "3.2, effective 2025-01-01" (plus policy_effective for filtering)
4. Numbers extracted once at indexing time: `parameters` (JSON) from the hand-checked rule-parameters file
   made by extract_rule_params.py, with `parameters_reviewed` recording whether a human has checked it.

Run (server must be up):
    python ingest_policy_rules.py "C:\\path\\to\\CDD_Periodic_Review_Policy_v3.2.md" --base-url https://ai-eng-wolraad.onrender.com
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

SECTION_PATTERN = re.compile(r"^## ([A-Z]+-\d+) (.+)$")
RULE_PATTERN = re.compile(r"^\*\*([A-Z]+-\d+\.\d+) (.+?)\.\*\*\s*(.*)$")
RULE_REF_PATTERN = re.compile(r"\b[A-Z]+-\d+(?:\.\d+)?\b")
DEFAULT_PARAMS_FILE = Path(__file__).resolve().parent / "cdd_rule_params.json"

# Short topic per section, for metadata filtering (the architecture doc's example: CDD-6.3 -> "address").
SECTION_TOPICS = {
    "CDD-1": "definitions",
    "CDD-2": "risk rating",
    "CDD-3": "review frequency",
    "CDD-4": "verification scope",
    "CDD-5": "identity",
    "CDD-6": "address",
    "CDD-7": "corporate structure",
    "CDD-8": "data consistency",
    "CDD-9": "screening",
    "CDD-10": "beneficial ownership",
    "CDD-11": "outstanding items",
    "CDD-12": "review record",
}

# Links the text doesn't spell out as rule IDs, checked by hand against the policy.
LINKED_REFERENCES = {
    "CDD-6.3": ["CDD-6.2"],  # A refreshed proof of address must meet the 3-month rule (architecture doc example).
    "CDD-2.3": ["CDD-11.2"],  # "Outstanding for more than 180 days" is the ageing rule in CDD-11.2.
    "CDD-3.4": ["CDD-3.3"],  # Unactioned triggers are the trigger events listed in CDD-3.3.
}

# Roles a rule can apply to, matched in the rule text (lowercase). Order sets the output order.
ROLE_KEYWORDS = [
    ("director", r"(?<!register of )\bdirectors?\b"),
    ("beneficial owner", r"\bbeneficial owners?\b"),
    ("corporate shareholder", r"\bcorporate shareholders?\b"),
    ("shareholder", r"(?<!corporate )\bshareholders?\b"),
    ("relevant person", r"\brelevant persons?\b"),
    ("client entity", r"\bclient(?: entit(?:y|ies))?\b"),
    ("MLRO", r"\bmlro\b"),
    ("compliance analyst", r"\bcompliance analyst\b"),
]
CONTEXT_KEYWORDS = [
    ("periodic", r"\bperiodic review"),
    ("event-driven", r"\bevent-driven\b|\btrigger events?\b"),
    ("new person", r"\bnew relevant person|\bbefore\b.*\bappointment\b"),
]


def front_matter(text: str, key: str) -> str | None:
    match = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_rules(markdown: str) -> list[dict]:
    """Split the policy into rules, keeping each rule's section and full text (including lists and tables)."""

    rules: list[dict] = []
    section_id = section_title = None
    current: dict | None = None
    for line in markdown.splitlines():
        if (section := SECTION_PATTERN.match(line)):
            section_id, section_title = section.groups()
            current = None
            continue
        if line.strip() == "---":
            current = None
            continue
        if (rule := RULE_PATTERN.match(line)) and section_id:
            rule_id, rule_title, first_line = rule.groups()
            current = {
                "rule_id": rule_id,
                "section_id": section_id,
                "section_title": section_title,
                "rule_title": rule_title,
                "lines": [first_line] if first_line else [],
            }
            rules.append(current)
            continue
        if current is not None:
            current["lines"].append(line)

    for rule in rules:
        rule["body"] = "\n".join(rule.pop("lines")).strip()
    return rules


def roles_in(text: str) -> list[str]:
    return [role for role, pattern in ROLE_KEYWORDS if re.search(pattern, text.lower())]


def roles_by_rule_reference(rules: list[dict]) -> dict[str, list[str]]:
    """
    Read the policy's own role table (e.g. CDD-4.1: "| Director | Identity (CDD-5), address (CDD-6) ... |")
    into {rule or section ID: roles}, used for rules whose text names no role.
    """

    mapping: dict[str, list[str]] = {}
    for rule in rules:
        for line in rule["body"].splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= {"-"}:
                continue
            for ref in RULE_REF_PATTERN.findall(cells[1]):
                for role in roles_in(cells[0]):
                    mapping.setdefault(ref, [])
                    if role not in mapping[ref]:
                        mapping[ref].append(role)
    return mapping


def rule_sort_key(rule_id: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", rule_id)]


def chunk_text(rule: dict) -> str:
    """The architecture doc's prefix: section heading › rule, so both are embedded with the rule text."""

    return f"{rule['section_id']} {rule['section_title']} › {rule['rule_id']} {rule['rule_title']}\n\n{rule['body']}"


def derive_metadata(
    rule: dict,
    version: str,
    effective: str | None,
    table_roles: dict[str, list[str]],
    params: dict | None,
    params_reviewed: bool,
) -> dict:
    body = rule["body"].lower()
    # Roles named in the rule itself; otherwise the policy's role table for this rule or its section.
    applies_to = (
        roles_in(body)
        or table_roles.get(rule["rule_id"])
        or table_roles.get(rule["section_id"])
        or ["client entity"]
    )
    titled_body = f"{rule['rule_title']}. {body}".lower()  # Titles like "Periodic review frequency" count too.
    contexts = [name for name, pattern in CONTEXT_KEYWORDS if re.search(pattern, titled_body)] or ["all"]
    mentioned = {ref for ref in RULE_REF_PATTERN.findall(rule["body"]) if ref != rule["rule_id"]}
    references = sorted(mentioned | set(LINKED_REFERENCES.get(rule["rule_id"], [])), key=rule_sort_key)

    metadata = {
        "rule_id": rule["rule_id"],
        "rule_title": rule["rule_title"],
        "topic": SECTION_TOPICS.get(rule["section_id"], rule["section_title"].lower()),
        "applies_to": applies_to,
        "review_context": contexts,
        "policy_version": f"{version}, effective {effective}" if effective else version,
    }
    if effective:
        metadata["policy_effective"] = effective
    if references:  # Omit rather than store an empty list.
        metadata["references"] = references
    if params:  # Pinecone metadata can't nest objects, so parameters are stored as JSON text.
        metadata["parameters"] = json.dumps(params, sort_keys=True)
        metadata["parameters_reviewed"] = params_reviewed
    return metadata


def load_params(path: Path, version: str) -> tuple[dict, bool]:
    if not path.exists():
        print(f"No parameters file at {path} - run extract_rule_params.py first to store rule parameters.")
        return {}, False
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("policy_version")) != version:
        sys.exit(f"{path.name} is for policy version {data.get('policy_version')}, not {version} - re-extract it.")
    return data.get("rules", {}), bool(data.get("reviewed"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_FILE, help="Rule parameters JSON")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    markdown = args.policy.read_text(encoding="utf-8")
    version = front_matter(markdown, "version") or "unknown"
    effective = front_matter(markdown, "effective")
    rules = parse_rules(markdown)
    if not rules:
        print(f"No rules found in {args.policy}")
        return 1
    table_roles = roles_by_rule_reference(rules)
    all_params, params_reviewed = load_params(args.params, version)
    if all_params and not params_reviewed:
        print(f"Note: {args.params.name} is not marked reviewed - parameters are stored with parameters_reviewed=false.")

    total_chunks = failures = 0
    total_cost = 0.0
    last_section = None
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120) as client:
        before = client.get("/health/pinecone").json().get("namespace_vector_count", 0)  # Also wakes Render.
        for rule in rules:
            if rule["section_id"] != last_section:
                last_section = rule["section_id"]
                print(f"\n{rule['section_id']} {rule['section_title']}")
            metadata = derive_metadata(
                rule, version, effective, table_roles, all_params.get(rule["rule_id"]), params_reviewed
            )
            response = client.post(
                "/ingest",
                json={
                    "document_id": rule["rule_id"],
                    "source": args.policy.name,
                    "text": chunk_text(rule),
                    "metadata": metadata,
                },
            )
            if response.status_code != 200:
                failures += 1
                print(f"  FAILED {rule['rule_id']:<9} HTTP {response.status_code}: {response.text}")
                continue
            data = response.json()
            total_chunks += data["chunks_indexed"]
            total_cost += data["cost_usd"]
            if data["chunks_indexed"] != 1:
                print(f"  WARNING {rule['rule_id']} was split into {data['chunks_indexed']} chunks - raise CHUNK_SIZE")
            print(
                f"  {rule['rule_id']:<9} {rule['rule_title']:<38} {data['chunks_indexed']} chunk(s)  "
                f"topic={metadata['topic']} | applies_to={', '.join(metadata['applies_to'])} | "
                f"context={', '.join(metadata['review_context'])} | refs={', '.join(metadata.get('references', [])) or '-'}"
                f" | params={metadata.get('parameters', '-')}"
            )

        print(f"\nIngested {total_chunks} chunks from {len(rules) - failures}/{len(rules)} rules (${total_cost:.6f})")

        # Pinecone serverless takes a few seconds to reflect new vectors in its stats.
        health: dict = {}
        for _ in range(15):
            health = client.get("/health/pinecone").json()
            if health.get("namespace_vector_count", 0) >= before:
                break
            time.sleep(2)
        print(
            f"Vector store '{health.get('index_name')}' / namespace '{health.get('namespace')}': "
            f"{health.get('namespace_vector_count')} chunks in total (was {before} before this run)"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
