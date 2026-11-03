### Reranking before/after - `http://127.0.0.1:8010` (top_k=5)

| Question | Expected doc | Rank without rerank | Rank with rerank |
|---|---|---|---|
| How many days per week can employees work remotely? | POL-101 | #1 | #1 |
| How many days of annual leave do employees get? | POL-101 | #1 | #1 |
| What is the mileage rate for business travel? | POL-114 | #1 | #1 |
| How long do I have to submit an expense claim? | POL-114 | #1 | #2 ⬇️ |
| How quickly must a lost laptop be reported? | POL-207 | #1 | #1 |
| What are the password requirements? | POL-207 | #1 | #1 |
| What is the WB-9 payload limit? | SPEC-WB9 | #1 | #1 |
| How long does the WB-9 battery last and how long does it take to recharge? | SPEC-WB9 | #1 | #1 |
| Who owns the facilities policy? | POL-118 | #5 | #1 ⬆️ |
| When is the security policy next reviewed? | POL-207 | #2 | #1 ⬆️ |
| Who approves the expenses policy? | POL-114 | #3 | #1 ⬆️ |
| What version is the employee handbook? | POL-101 | #1 | #1 |

| Metric | Without rerank | With rerank |
|---|---|---|
| hit@1 | 0.75 | 0.92 |
| hit@3 | 0.92 | 1.00 |
| hit@5 | 1.00 | 1.00 |
| mrr | 0.84 | 0.96 |
| avg /debug/retrieve latency | 477 ms | 700 ms |

**Answer check** - "Who owns the facilities policy?" with top_k=3:

- rerank off: ❌ "I don't know based on the provided documents." (chunks: POL-101#21, POL-207#11, POL-114#0)
- rerank on : ✅ "The facilities policy is owned by the Head of Workplace [POL-118]." (chunks: POL-118#0, POL-207#1, POL-114#0)
