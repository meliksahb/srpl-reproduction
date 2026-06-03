# SRPL Reproduction — Implementation Plan

**Course:** CENG 502 — Advanced Deep Learning
**Paper:** Safety Representations for Safer Policy Learning (SRPL), Mani et al., ICLR 2025 ([arXiv:2502.20341](https://arxiv.org/abs/2502.20341))
**Goal:** Reproduce the core SRPL results on Safety-Gymnasium from scratch (no public code exists for SRPL), evaluated via a clean, gradeable GitHub repo.

---

## 1. Scope (locked)

| ID | Result | What we produce |
|----|--------|-----------------|
| A | Figure 4 (partial) — main training curves | Episodic return & episodic cost vs. steps, for 2 envs × 3 algos × {base, SR} × 5 seeds = **60 runs** |
| B | Table 1 (corresponding rows) | End-of-training avg return + cost-rate for the 6 conditions, paper's format |
| C | Figure 1 — Island Navigation toy | DQN vs. DQN+GT-oracle: return curve + Q-value distribution plot (motivational) |
| D | Figure 6 — transfer | S2C trained on PointButton1, frozen, applied to PointGoal1 with PPO-Lag; vs. random-init S2C and a no-S2C baseline, all on unified-obs; 5 seeds, **~15 runs** |

**Environments:** `SafetyPointGoal1-v0`, `SafetyPointButton1-v0`
**Algorithms:** PPO-Lag (on-policy), TD3-Lag, SAC-Lag (off-policy)
**Seeds:** 5 per condition
**Total:** 60 (main) + ~15 (transfer) + ~10 (Island Nav) runs

### Reproduction-target framing (state this in the README)
PPO-Lag / TD3-Lag / SAC-Lag are **not** in the paper's Table 1 (its baselines are CPO, TRPO-PID, SauteRL, CRPO, CSC, CVPO). The paper's central claim is that SRPL is **algorithm-agnostic**. Our reproduction target is therefore the **SRPL effect** — the *relative* improvement of each SR-variant over its own base (higher return, lower/equal cost-rate) — benchmarked against the direction and magnitude of improvement the paper reports (e.g., SR-CPO lifts PointGoal1 return 6.25→11.63, ≈ +86%). This is a legitimate and arguably stronger test than cell-matching, and it is exactly what the paper claims should hold.

---

## 2. What we reuse vs. what we build

### Reuse (do NOT reimplement — standard practice, documented in README)
- **OmniSafe** (`PKU-Alignment/omnisafe`): base algorithms `PPOLag`, `TD3Lag`, `SACLag`; Lagrange-multiplier handling; vectorized-env training loop; logging; parallelism. The SRPL authors themselves used OmniSafe for their on-policy baselines.
- **Safety-Gymnasium** (`PKU-Alignment/safety-gymnasium`): the `SafetyPointGoal1-v0` / `SafetyPointButton1-v0` environments and the separate `cost` channel.
- **PyTorch / MuJoCo**: dependencies pulled in by the above.

### Build from scratch (this is the SRPL contribution — no public code exists)
1. **S2C model** — MLP → softmax over K bins (`srpl/s2c_model.py`)
2. **Steps-to-cost labeling** — δ computation from episode cost signals (`srpl/labeling.py`) ← highest-risk component
3. **S2C buffer** — FIFO buffer for on-policy; replay-buffer hooks for off-policy (`srpl/s2c_buffer.py`)
4. **State augmentation** — detached concat of S2C output to obs (`srpl/augmentation.py`)
5. **SRPL-wrapped algorithms** — `SRPPOLag`, `SRTD3Lag`, `SRSACLag` subclassing OmniSafe (`srpl/algorithms/`)
6. **Island Navigation** — gridworld (4 layouts) + DQN + DQN-GT, fully from scratch (`island_navigation/`)
7. **Transfer harness** — unified-observation wrapper + frozen/random-S2C logic (`scripts/transfer.py`)
8. **Results tooling** — log parsing → Table 1 + Figure 4/1/transfer plots (`scripts/aggregate_results.py`)

---

## 3. Software stack & environment

| Component | Choice / Notes |
|-----------|----------------|
| OS | Ubuntu 22.04 (the A4000 box) |
| Python | 3.10 or 3.11 — match OmniSafe's current requirement; verify in their `pyproject.toml` |
| Package mgr | `conda` for the env, `pip` inside it |
| Core deps | `omnisafe`, `safety-gymnasium`, `torch`, `numpy` |
| Headless MuJoCo | `export MUJOCO_GL=egl` (preferred on GPU box) or `osmesa` fallback; `apt-get install libosmesa6-dev python3-opengl` if needed |
| Logging | TensorBoard (always on) + optional Weights & Biases |
| Plots/tables | `matplotlib`, `pandas`, `seaborn` |
| Tests | `pytest` |

**Pin everything** in `requirements.txt` / `environment.yml` with exact versions. Safety-Gymnasium and OmniSafe both move; an unpinned env is the most likely cause of a "works on my machine" failure when the professor clones the repo.

---

## 4. SRPL method specification (implementation-exact)

### 4.1 S2C model
- MLP, **2 hidden layers × 64 units** (Appendix A.3.2: "same architecture as the policy"). **Verify OmniSafe's default policy net size in Stage 0** — the paper ties S2C to the policy architecture, and OmniSafe's *off-policy* (TD3/SAC) defaults may be larger than [64,64]. Match S2C to whatever the base policy actually uses.
- Input: **raw** state `s` (original obs, *before* augmentation — S2C never takes its own output as input).
- Output: softmax over **K = H_s / bin_size** bins. For Safety-Gym: `H_s = 80`, `bin_size = 4` → **K = 20**.
- Loss: NLL / cross-entropy against one-hot bin label (Eq. 3).

### 4.2 Steps-to-cost labeling (the critical piece)
Safety-Gym episodes **do not terminate on cost** — they are fixed-length (1000 steps by default) and *truncate*, accumulating many cost events along the way (reaching a goal respawns a new goal within the same episode). So each state's label is **distance to the NEXT cost event from that state**, not to the first/last failure. The code below operates on whatever episode length `T` it's given, so it's robust to the exact max-steps value.

> **Vectorized-env caveat:** with N parallel envs (which we need for speed), the cost stream must be segmented **per env index** and labeled at **each env's own episode boundary**. Flattening cost across envs, or labeling at the wrong boundary, silently corrupts every label. Maintain one in-progress trajectory buffer per env; label and flush when that env returns `truncated`/`terminated`.

```python
def label_trajectory(costs, H_s=80, bin_size=4):
    """costs[t] > 0 means step t incurred a cost. Returns bin index per state."""
    T, K = len(costs), H_s // bin_size
    cost_steps = [t for t in range(T) if costs[t] > 0]
    labels, j = np.empty(T, dtype=np.int64), 0
    for t in range(T):
        while j < len(cost_steps) and cost_steps[j] <= t:
            j += 1
        delta = (cost_steps[j] - t) if j < len(cost_steps) else H_s   # >= 1
        delta = min(delta, H_s)
        labels[t] = min((delta - 1) // bin_size, K - 1)
    return labels
```

Edge cases (each silently corrupts S2C if wrong — unit-test all):
- δ ∈ {1,…,H_s}; δ=1 means the *next* state is unsafe.
- "Safe through horizon" (δ=H_s) and "cost at exactly step 80" both fall in the last bin (index 19) — intended.
- Because H_s=80 ≪ 1000, most states get δ=H_s → last bin. The S2C output is expected to skew "safe"; do **not** "fix" this.

### 4.3 State augmentation + gradient isolation
```python
with torch.no_grad():                 # gradients must NOT flow into S2C from the policy loss
    safety_repr = s2c_model(obs)      # [B, K] softmax
augmented_obs = torch.cat([obs, safety_repr], dim=-1)
```
S2C is trained **only** by its own NLL loss (Fig 2 = two separate gradient paths). Missing `detach`/`no_grad` is a classic silent bug.

### 4.4 Buffer + update-frequency asymmetry (on-policy vs off-policy)
| | On-policy (PPO-Lag) | Off-policy (TD3/SAC-Lag) |
|---|---|---|
| S2C data | **separate** FIFO buffer of recent rollouts | **reuse** agent replay buffer |
| S2C update_freq | 100 (frequent — stable, policy frozen during rollout) | 20000 (rare — target-network-like) |
| S2C lr | 1e-6 or 1e-5 | 1e-3 |
| S2C batch | 5000 | 512 |

Principle to preserve: on-policy → update S2C often; off-policy → update S2C rarely (or the augmented-state distribution shifts every step and destabilizes training). `update_freq` is the **most sensitive knob** — be ready to tune it.

> **`update_freq` units are ambiguous in the paper.** "100" (on-policy) and "20000" (off-policy) are not stated in env-steps vs gradient-steps. For on-policy the policy updates ~once per 20000-step epoch, so "every 100" must be sub-epoch. Read the exact semantics off OmniSafe's training loop in Stage 2 and treat these as starting points, not gospel.

> **Off-policy labeling is necessarily delayed.** A state's δ can only be computed once the next cost (or episode end) is observed. So the off-policy flow is: buffer an episode's transitions *without* δ → at episode end, backfill δ via `label_trajectory` → commit labeled transitions to the replay buffer. The S2C then trains on sampled batches that already carry δ.

> **Augmentation-staleness design decision (paper underspecifies).** Two options: (i) store **raw** obs in the replay buffer and augment on-sample with the *current* S2C; (ii) store the **augmented** obs (frozen at collection time). Because the S2C is held nearly fixed across the long `update_freq` window (the paper's "target-network-like" analogy), the two are almost equivalent within a window. Default to (i) — store raw, augment with current S2C — and document the choice.

### 4.5 Training horizons (two regimes — do not conflate)
- **PPO-Lag (on-policy):** 10M steps (Fig 4 regime).
- **TD3-Lag / SAC-Lag (off-policy):** 2M steps (Fig 5 regime — off-policy is more sample-efficient).

### 4.6 Metrics
- **Return:** mean episodic return over eval episodes at end of training. **Eval protocol (pin it):** deterministic policy (mean action, no exploration noise / no sampling), averaged over the last N≈10 evaluation rollouts, matching the paper's "end of training."
- **Cost-Rate (×1e2):** `cumulative_cost_over_all_training / total_env_steps × 100`. Cumulative from step 0 — **not** episodic cost at convergence. (OmniSafe does not log this form directly; accumulate it ourselves from per-step costs and total step count.)
- **Cost threshold** β = 10 for both Safety-Gym tasks.

---

## 5. Integration strategy (the main technical risk)

**Primary approach — inject SRPL into OmniSafe** while reusing its tested base algorithms. The Stage-2 spike tries these injection mechanisms in order of expected cleanliness:
1. **ObservationWrapper + training callback.** A Gymnasium wrapper holds a reference to the live S2C and augments obs (frozen-within-window); a callback collects episodes, labels them, and steps the S2C optimizer at `update_freq`. Cleanest *if* OmniSafe exposes a callback hook at the right granularity.
2. **Subclass the algorithm** (`SRPPOLag(PPOLag)`, etc.), overriding (a) transition storage → record `(s, cost)` + compute δ at episode end, (b) the obs handed to actor/critic → detached-concat S2C output, (c) the update step → train S2C. Uses OmniSafe's algorithm registry.
3. **Custom outer loop** built from OmniSafe components (actor-critic, buffer, Lagrange, update fns). Most control, most code — last resort.

**Why reuse base algorithms at all:** re-implementing correct Lagrangian PPO/TD3/SAC is unnecessary bug surface; the course requirement is to implement *SRPL* (no public code), not base safe-RL algorithms. Repo clarity comes from a clean, documented `srpl/` module + clean configs/scripts/README.

**Risk & go/no-go:** OmniSafe's override points may be awkward. Stage 2 validates one short SR-PPO-Lag run *before* the full grid. **Fallback:** if all three mechanisms above are too invasive/unreadable, drop to a single-file CleanRL-style Lagrangian PPO + SRPL (full control, more readable, but we then re-implement and re-validate the base) — decided only if the spike fails.

---

## 6. Repository structure

```
srpl-reproduction/
├── README.md                  # overview, setup, how-to-run, RESULTS, discussion, citation
├── PLAN.md                    # this document
├── requirements.txt           # exact pins
├── environment.yml            # conda env
├── LICENSE
├── .gitignore                 # ignores experiments/ raw outputs, wandb/, __pycache__
│
├── srpl/                      # OUR CONTRIBUTION (no public code exists)
│   ├── __init__.py
│   ├── s2c_model.py           # S2C network + NLL loss
│   ├── labeling.py            # steps-to-cost δ labeling
│   ├── s2c_buffer.py          # FIFO (on-policy) + replay hooks (off-policy)
│   ├── augmentation.py        # detached state augmentation
│   └── algorithms/
│       ├── __init__.py        # register with OmniSafe
│       ├── sr_ppo_lag.py
│       ├── sr_td3_lag.py
│       └── sr_sac_lag.py
│
├── island_navigation/         # Figure 1 (standalone)
│   ├── env.py                 # gridworld, 4 layouts (Fig 11)
│   ├── dqn.py                 # DQN + DQN-GT (oracle Manhattan distance)
│   └── run_island.py
│
├── configs/                   # one yaml per (algo); env/seed via CLI
│   ├── ppo_lag.yaml
│   ├── td3_lag.yaml
│   └── sac_lag.yaml
│
├── scripts/
│   ├── train.py               # single-run entry: --algo --env --seed --sr
│   ├── run_all.sh             # full grid launcher (parallel seeds)
│   ├── transfer.py            # Figure 6 transfer experiment
│   └── aggregate_results.py   # logs → Table 1 (md/csv) + Figure 4/1/transfer plots
│
├── tests/
│   ├── test_labeling.py       # δ-labeling unit tests (highest priority)
│   ├── test_augmentation.py   # detach / shape checks
│   └── test_s2c_model.py      # forward + loss smoke test
│
├── experiments/               # (gitignored) raw run outputs / checkpoints
├── results/                   # (committed) final plots + tables
│   ├── figure4_pointgoal1.png
│   ├── figure4_pointbutton1.png
│   ├── table1.md
│   ├── figure1_island.png
│   └── figure_transfer.png
└── notebooks/
    └── analysis.ipynb         # optional exploration
```

---

## 7. Experiment matrix (precise)

### Main grid (A + B) — 60 runs
```
envs   = [SafetyPointGoal1-v0, SafetyPointButton1-v0]
algos  = [PPOLag (10M), TD3Lag (2M), SACLag (2M)]
cond   = [base, SR]
seeds  = [0,1,2,3,4]
```
Each SR run logs: episodic return, episodic cost, cumulative cost (for cost-rate), S2C NLL.

### Transfer (D) — ~15 runs (Figure 6)
All transfer runs use the **unified-observation wrapper** (Appendix A.4), since default PointGoal1 and PointButton1 have different obs dims and a transferred S2C must ingest both. This means the transfer plot needs its own unified-obs baseline — the default-obs PPO-Lag base from the main grid is *not* comparable.
- Source S2C: train on **unified-obs PointButton1** (1–2 runs; can piggyback a SR-PPO-Lag run with the wrapper).
- Target: **unified-obs PointGoal1** with PPO-Lag, 5 seeds each of:
  - **frozen transferred S2C** (the SRPL claim),
  - **frozen random-init S2C** (clean ablation — isolates transferred knowledge vs random features of the same shape),
  - **plain PPO-Lag, no S2C** (unified-obs) — the paper-faithful no-safety reference.

### Island Navigation (C) — ~10 runs (cheap)
- DQN vs DQN+GT-oracle, a few seeds; plus the Q-value-distribution snapshot for the Fig-1 col-2 plot.

---

## 8. Compute plan (the binding resource is CPU, not GPU)

MuJoCo/Safety-Gym is **CPU-bound**: the simulator steps on CPU; the GPU only runs a tiny [64,64] MLP forward/backward and sits >90% idle. Wall-clock is set by **CPU cores + vectorized-env count**, not GPU class.

- **Use the A4000 box** if it has a strong multi-core CPU (8+ physical cores). Run multiple seeds as **separate parallel processes**, each with a few vectorized envs; the GPU holds all of them at once (<1 GB each).
- **Paid Colab buys little here** — an A100 next to a weak CPU is slower than the A4000 next to a strong CPU. Only worth it if the local CPU is weak or you want to fan seeds across machines for wall-clock.
- **Rough budget:** on-policy 10M runs dominate (~3–6 h each); off-policy 2M runs ~1.5–3 h; transfer ~20–30 h; Island Nav ~5 h. Total ≈ **180–250 CPU-hours**, ~3–5 days with 2–3 parallel seeds. If compute gets tight, **cut a second off-policy algorithm first** (SAC- *or* TD3-Lag), since it re-tests the same claim.

---

## 9. Execution stages (de-risk before scaling — do NOT launch 70 jobs early)

| Stage | Goal | Exit criterion |
|-------|------|----------------|
| 0 | Env setup + smoke test | Vanilla `PPOLag` on PointGoal1 for ~100k steps trains and logs return/cost |
| 1 | Labeling correctness | `pytest tests/test_labeling.py` green on hand-built cost arrays |
| 2 | **SRPL integration spike** | SR-PPO-Lag, 1 seed, short run: S2C NLL ↓, augmented agent trains, cost-rate plausible |
| 3 | On-policy grid | PPO-Lag base+SR, 2 envs, 5 seeds (10M) → Fig-4 curves + Table-1 rows |
| 4 | Off-policy grid | TD3-Lag / SAC-Lag base+SR, 2 envs, 5 seeds (2M); tune `update_freq` if unstable |
| 5 | Transfer (Fig 6) | Unified-obs wrapper works; frozen-transfer beats random-init S2C |
| 6 | Island Nav (Fig 1) | DQN+GT explores more / higher return than plain DQN; Q-value plot reproduced |
| 7 | Aggregate + write-up | All plots/tables in `results/`; README results section complete |

Stages 1 and 6 are independent and can run in parallel with others.

---

## 10. Risks & pitfalls (ranked by likelihood of silently wasting time)

1. **Wrong δ-labeling** under multiple-costs-per-episode (§4.2) → unit-test first.
2. **Vectorized-env label corruption** (§4.2): with N parallel envs, the cost stream must be segmented per env index and labeled at each env's own episode boundary. Naive flattening silently corrupts every label → one trajectory buffer per env, flush on that env's done.
3. **S2C gradient leaking into policy** (missing `detach`) → assert no grad on S2C params after a policy step.
4. **Off-policy instability** from too-frequent S2C updates (§4.4) → start `update_freq=20000`, watch for return collapse.
5. **Off-policy delayed labeling** (§4.4): label at episode end and backfill before committing to the replay buffer; don't commit unlabeled transitions.
6. **Cost-rate mis-defined** as episodic-cost-at-convergence instead of cumulative/steps (§4.6) → track running cost sum from step 0.
7. **Transfer obs-dim mismatch** (Fig 6): default PointGoal1 and PointButton1 have different obs dims, so a S2C trained on one can't ingest the other. Build the unified aggregated-LiDAR wrapper (Appendix A.4) — or pad/truncate as a documented fallback — and use it for **all** transfer runs including the baseline; plan it from day 1 of Stage 5.
8. **Env-version drift**: paper used original Safety-Gym (Ray 2019, mujoco-py); we use Safety-Gymnasium (gymnasium, modern MuJoCo). Rewards/costs are *intended* to match but minor numerical differences are expected — document as a known discrepancy rather than chasing exact parity.
9. **Unpinned deps** → exact version pins in `requirements.txt`.
10. **OmniSafe injection awkwardness** → the Stage-2 spike exists precisely to catch this before the full grid.

---

## 11. README plan (graded artifact)

The README must let the professor (a) understand the project, (b) clone-and-run, (c) see results without running anything.

1. **Title + one-paragraph overview** (paper, what SRPL is, what we reproduced).
2. **Scope table** (A/B/C/D) + the reproduction-target framing (§1) — including the explicit note that PPO/TD3/SAC-Lag aren't in Table 1.
3. **Setup**: exact conda/pip commands, MuJoCo env var, verified Python version.
4. **How to run**: single run (`scripts/train.py --algo PPOLag --env SafetyPointGoal1-v0 --seed 0 --sr`), full grid (`run_all.sh`), transfer, Island Nav.
5. **Repo structure** (the tree above, annotated; clearly marks `srpl/` as our contribution vs OmniSafe-provided base algos).
6. **RESULTS** (the centerpiece):
   - Figure 4 plots (PointGoal1, PointButton1) — base vs SR curves.
   - Table 1 (markdown) — our 6 conditions, return + cost-rate, with the paper's relative-improvement direction for comparison.
   - Figure 1 (Island Navigation) — return curve + Q-value distribution.
   - Transfer (Figure 6) — frozen vs random S2C.
7. **Discussion**: which SRPL effects replicated, magnitudes vs paper, observed discrepancies, env-version caveat, off-policy `update_freq` sensitivity.
8. **Reproducibility**: seeds, hardware, wall-clock, commit hash.
9. **Citation** (paper) + **acknowledgement** (OmniSafe/Safety-Gymnasium as base infra).

---

## 12. Immediate next step

Start at **Stage 1**: implement `srpl/labeling.py` + `tests/test_labeling.py` (hand-built cost arrays covering: no-cost episode, single cost, multiple costs, cost at step 0, cost beyond horizon). This is the highest-risk, lowest-cost component and everything downstream depends on it being correct. Then Stage 0 (env smoke test) in parallel, then the Stage-2 integration spike.
