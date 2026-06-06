# Table 1 (reproduction): end-of-training return and training cost-rate

Final-10%-of-training averages; mean +/- std over seeds. Cost-rate is x1e2 (cumulative-cost / steps). Lower cost is better.

| Task | Algorithm | Variant | Seeds | Return (mean +/- std) | Cost (mean +/- std) | Cost-rate x1e2 |
|---|---|---|---|---|---|---|
| PointGoal1 | PPOLag | Base | 3 | 13.44 +/- 1.39 | 49.4 +/- 3.4 | 6.23 +/- 0.06 |
| PointGoal1 | PPOLag | SR | 3 | 12.22 +/- 0.64 | 53.7 +/- 15.0 | 6.32 +/- 0.50 |
| PointGoal1 | TD3Lag | Base | 3 | 4.65 +/- 1.42 | 38.9 +/- 10.3 | 5.15 +/- 0.23 |
| PointGoal1 | TD3Lag | SR | 3 | 2.25 +/- 0.23 | 47.7 +/- 24.5 | 5.08 +/- 0.26 |
| PointGoal1 | SACLag | Base | 3 | 0.49 +/- 0.16 | 45.6 +/- 7.8 | 4.83 +/- 0.19 |
| PointGoal1 | SACLag | SR | 3 | 0.36 +/- 0.10 | 46.5 +/- 7.4 | 4.88 +/- 0.40 |
| PointButton1 | PPOLag | Base | 3 | 6.39 +/- 2.67 | 100.2 +/- 15.1 | 9.81 +/- 0.62 |
| PointButton1 | PPOLag | SR | 3 | 5.28 +/- 0.98 | 87.6 +/- 7.6 | 8.98 +/- 0.22 |
| PointButton1 | TD3Lag | Base | 3 | 0.05 +/- 0.06 | 69.5 +/- 13.3 | 10.26 +/- 0.17 |
| PointButton1 | TD3Lag | SR | 3 | -0.76 +/- 0.94 | 42.8 +/- 8.3 | 10.29 +/- 0.30 |
| PointButton1 | SACLag | Base | 3 | 0.38 +/- 0.06 | 66.6 +/- 13.5 | 5.67 +/- 0.25 |
| PointButton1 | SACLag | SR | 3 | 0.47 +/- 0.05 | 59.4 +/- 5.0 | 5.10 +/- 0.23 |

## SRPL effect (SR - Base), per task/algorithm

| Task | Algorithm | dReturn | dCost | dCost-rate |
|---|---|---|---|---|
| PointGoal1 | PPOLag | -1.22 | +4.3 | +0.09 |
| PointGoal1 | TD3Lag | -2.41 | +8.8 | -0.07 |
| PointGoal1 | SACLag | -0.13 | +0.9 | +0.06 |
| PointButton1 | PPOLag | -1.11 | -12.6 | -0.83 |
| PointButton1 | TD3Lag | -0.81 | -26.7 | +0.03 |
| PointButton1 | SACLag | +0.09 | -7.2 | -0.57 |
