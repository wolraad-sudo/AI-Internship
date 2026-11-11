# CASE-2026-0147: what the agent should find

This case and all its data are made up. It is built so the agent has one example of each status. The statuses below are my reading of `cdd_rule_params.json` (policy v3.2). Before you use this as a golden answer, check each one against the policy text.

Review date: **2026-11-10**. Prior review: 2025-11-12 (REV-2025-0312).

| # | Person | Item | Rule | Expected status | Why |
|---|---|---|---|---|---|
| 1 | P1 Maria Okafor | Identity (F01) | CDD-5.2, CDD-5.3 | OK | Passport expires 2031; certified 2026-10-20 (under 3 months old) |
| 2 | P1 Maria Okafor | Proof of address (F02) | CDD-6.3 | OK | Utility bill 2026-09-15 (under 12 months old) |
| 3 | P1 Maria Okafor | Name-change linking document | CDD-8.2 | Missing | Former name Bello; no marriage certificate on file |
| 4 | P2 David Chen | Identity (F03) | CDD-5.2, CDD-5.3 | OK, plus renewal request | Expires 2026-12-31, within 90 days of the review. CDD-5.2's text in Pinecone says to request the renewed document as an outstanding item, not to mark it Expired. Certification is 2025-11-05, so check whether CDD-5.3's 3-month limit applies to documents already held |
| 5 | P2 David Chen | Proof of address (F04) | CDD-6.3 | Stale | Bank statement 2025-08-01 (over 12 months old) |
| 6 | P3 Northgate Ventures Ltd | Certificate of good standing (F05) | CDD-7.2 | Stale | 2026-03-01 (over 6 months old) |
| 7 | P3 Northgate Ventures Ltd | Registers (F06) | CDD-7.3 | OK | 2026-02-01 (under 12 months old); beneficial owner matches |
| 8 | P4 Sara Lindqvist | Identity | CDD-5.2 | Missing | New director (2026-08-20); nothing on file |
| 9 | P4 Sara Lindqvist | Proof of address | CDD-6.2 | Missing | New relevant person; nothing on file |
| 10 | P4 Sara Lindqvist | Screening | CDD-9.2 | Missing | F10 screened only P1 to P3 |
| 11 | Client | Registers (F07 to F09) | CDD-8.1, CDD-10.2 | OK | Agree with the persons listed in the index |

Points the agent should flag:
- **The linking document is overdue.** It was first requested on 2025-11-20 and is still outstanding at 355 days. That is over the 180-day limit (CDD-11.2 / CDD-2.3), so the client should be **re-rated from Medium to High**.
- **There is a trigger event.** A new director was appointed since the last review (CDD-3.3).
- **The proposed outcome should be Escalated to MLRO** (CDD-12.3). The other two outcomes don't fit: "Completed – outstanding items" only allows items up to 90 days old, and one item is 355 days old.
- **The final status must be `PENDING_ANALYST_APPROVAL`.** The agent must not mark the review approved (CDD-12.4).
