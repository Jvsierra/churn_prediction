# Business Rules — Churn Prediction

## Risk segments
| Segment | Score range | Action |
|---|---|---|
| High | >= 0.60 | Proactive call + personalized retention offer |
| Medium | 0.30–0.59 | Email or push with discount or upsell |
| Low | < 0.30 | No action |

## Business assumptions
- Save rate: 30% of contacted customers are successfully retained
- Cost per action: $15 per retention action (call, discount, etc.)
- Default threshold: 0.5 (can be lowered to increase recall)

## Business impact (holdout simulation)
- Baseline churn rate: 26.56%
- Post-retention churn rate (Random Forest): 20.45%
- MRR protected per cycle: ~$6,500
- ROI: 5.71x

## When to lower the threshold
If the cost of missing a churner (lost LTV) is much higher than the cost
of a false positive (unnecessary retention action), lower the threshold to
0.3 or 0.4 to increase recall at the cost of precision.

## Key churn drivers (from EDA)
1. Month-to-month contract — 43% churn rate vs 3% for 2-year contracts
2. Short tenure (< 12 months) — highest risk window
3. No premium services (security, backup, tech support) — no switching cost
4. Fiber optic internet — more competitive market
5. Electronic check payment — correlates with higher churn
6. Senior citizens — 2x higher churn rate than non-seniors