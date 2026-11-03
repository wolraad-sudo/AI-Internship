### Golden-set eval - `https://ai-eng-wolraad.onrender.com` (top_k=5)

| # | Question | Expected | Service answer | Retrieval hit (top-5) | Faithful | Correct |
|---|---|---|---|---|---|---|
| 1 | How many days per week can employees work remotely? | Up to 3 days per week | Employees may work remotely up to three days per week [POL-101]. | ✅ | ✅ | ✅ |
| 2 | How many days of annual leave do employees get? | 28 days plus public holidays | Employees get 28 days of annual leave plus public holidays [POL-101]. | ✅ | ✅ | ✅ |
| 3 | What is the mileage rate for business travel? | 45p per mile for travel over 50 miles | The mileage rate for business travel is 45 pence per mile for travel over 50 miles [POL-114]. | ✅ | ✅ | ✅ |
| 4 | How long do I have to submit an expense claim? | Within 30 days, with a receipt | You have 30 days to submit an expense claim. Claims older than 30 days need director approval and will be paid in the next cycle only if approved. Claims older than 90 days are denied except for documented system outages [POL-114]. | ✅ | ✅ | ✅ |
| 5 | How quickly must a lost laptop be reported? | Within 1 hour, to the security desk | A lost laptop must be reported within one hour to the security desk [POL-207]. | ✅ | ✅ | ✅ |
| 6 | What are the password requirements? | At least 14 characters, rotated every 90 days | Passwords must be at least 14 characters long and rotated every 90 days. Multi-factor authentication is also mandatory for all systems [POL-207]. | ✅ | ✅ | ✅ |
| 7 | What is the WB-9 payload limit? | 25 kg | The WB-9 warehouse robot has a payload capacity of 25 kg. [SPEC-WB9] | ✅ | ✅ | ✅ |
| 8 | How long does the WB-9 battery last and how long does it take to recharge? | 8 hours; 90 minute recharge | The WB-9 battery lasts 8 hours on a single charge and takes 90 minutes to recharge [SPEC-WB9]. | ✅ | ✅ | ✅ |
| 9 | What is the parental leave policy? | Refuse (not in docs) | I don't know based on the provided documents. | N/A | ✅ | ✅ |
| 10 | Do employees get stock options? | Refuse (not in docs) | I don't know based on the provided documents. | N/A | ✅ | ✅ |

**Summary:** retrieval hit 8/8 · faithful 10/10 · correct 10/10 (including 2 refusal questions).
