---

## 6. How to reproduce

### Setup

```bash
conda create -n srpl python=3.10 -y && conda activate srpl
pip install safety-gymnasium omnisafe
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -e .
pytest -q                                                            # 49 tests pass
```

> **Note.** Training is CPU-bound (MuJoCo physics + tiny networks); a GPU is not
> required. On this workstation each run is capped to **2 torch threads** and
> runs **4 in parallel** — small networks train *faster* with few threads, and
> 4x2 threads saturates the 8 physical cores without oversubscription.

### Main experiment (Figure 4 + Table 1)

```bash
# 36 runs: PPO-Lag (1M) + TD3-Lag/SAC-Lag (500K), 2 envs x base/SR x 3 seeds.
# Resumable: re-running skips completed runs (markers in experiments/full/_markers).
nohup bash scripts/run_all.sh > experiments_run.log 2>&1 &

# When done (ls experiments/full/_markers | wc -l == 36):
python scripts/aggregate_results.py --base ./experiments/full
# -> experiments/full/_analysis/{fig4_*.png, table1.md, results.csv}
```

### Island Navigation (Figure 1)

```bash
PYTHONPATH=. python island_navigation/dqn.py --layout pond --episodes 400 --seeds 10 \
    --out experiments/full/_analysis
# -> experiments/full/_analysis/fig1_pond.png
```

### Transfer (Figure 6)

```bash
# Phase 1 (source): train + save the PointButton1 S2C (also run by run_transfer.sh).
PYTHONPATH=. python scripts/train_source_s2c.py --out experiments/transfer/s2c_button1.pt

# Phase 2 (target): 9 runs = {frozen-transfer, frozen-random, base} x 3 seeds on PointGoal1.
nohup bash scripts/run_transfer.sh > transfer_run.log 2>&1 &

# When done (ls experiments/transfer/_markers | wc -l == 9):
python scripts/plot_transfer.py --base ./experiments/transfer
# -> experiments/transfer/_analysis/fig6_transfer.png
```

---

## 7. Metric definitions

- **Episodic return / cost:** the per-episode `Metrics/EpRet` / `Metrics/EpCost`
  OmniSafe logs each epoch (both on- and off-policy).
- **Cost-rate (×1e2):** cumulative cost over *all* training divided by total
  steps, ×100 — a "how unsafe was the agent *while learning*" metric (computed
  over all epochs, not just the end).
- **End-of-training return/cost:** mean over the final 10% of epochs.

> Note: cost-rate (cumulative window) and end-of-training cost (final-10% window)
> measure different windows and can therefore differ in sign on the SR−Base
> delta; both are reported and labeled distinctly.

---

## 8. Limitations

- **n = 3 seeds** → no statistical-significance claims; means can be swung by a
  single seed (visible in some PointGoal1 cells).
- **Reduced horizon** → results are pre-convergence; return-side gains and the
  PointGoal1 effect are not separable at this budget.
- **Different algorithms than the paper** → effect-level, not absolute-number,
  reproduction.
- **Transfer uses zero-padding (not semantic unified-LiDAR) and a random-policy
  source S2C** → a functional but approximate version of the paper's Appendix-A.4
  setup.

Despite these, the reproduction (a) implements the SRPL method faithfully and
verifiably, (b) reproduces the **safety-during-learning** effect on the harder
task across all three algorithms, and (c) reproduces the Island Navigation
motivating result cleanly.

---

## 9. Acknowledgements & citation

Built on [OmniSafe](https://github.com/PKU-Alignment/omnisafe) (Ji et al., 2024)
and [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).
Island Navigation follows Leike et al. (2017).

```bibtex
@inproceedings{mani2025srpl,
  title     = {Safety Representations for Safer Policy Learning},
  author    = {Mani, Kaustubh and Mai, Vincent and Gauthier, Charlie and
               Chen, Annie and Nashed, Samer and Paull, Liam},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

*Reproduction by Melikşah (CENG 502). 3 seeds; reduced horizon; PPO-Lag /
TD3-Lag / SAC-Lag. See Section 3 for all deviations from the paper.*
