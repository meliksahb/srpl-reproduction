"""Exhaustive unit tests for srpl.labeling.

The steps-to-cost labeling is the highest-risk component of the SRPL
reproduction: it feeds the S2C model, whose output is concatenated into the
policy's observation, so a labeling bug degrades training silently rather than
crashing. These tests pin the convention documented in srpl/labeling.py.

Run:  pytest -q tests/test_labeling.py
"""

import numpy as np
import pytest

from srpl.labeling import (
    steps_to_cost,
    _steps_to_cost_reference,
    delta_to_bin,
    label_trajectory,
    num_bins,
    PerEnvEpisodeAccumulator,
    DEFAULT_SAFETY_HORIZON,
    DEFAULT_BIN_SIZE,
)


# --------------------------------------------------------------------------- #
# num_bins / validation
# --------------------------------------------------------------------------- #
def test_num_bins_safety_gym_default():
    # Paper: H_s = 80, bin_size = 4 -> K = 20 for PointGoal1 / PointButton1.
    assert num_bins(80, 4) == 20
    assert num_bins() == 20  # defaults


@pytest.mark.parametrize("H_s,bin_size", [(0, 4), (80, 0), (4, 80), (-1, 1)])
def test_validation_rejects_bad_params(H_s, bin_size):
    with pytest.raises(ValueError):
        num_bins(H_s, bin_size)


def test_steps_to_cost_rejects_2d():
    with pytest.raises(ValueError):
        steps_to_cost(np.zeros((3, 3)))


# --------------------------------------------------------------------------- #
# steps_to_cost: explicit hand-computed cases
# --------------------------------------------------------------------------- #
def test_no_cost_all_horizon():
    costs = [0, 0, 0, 0, 0]
    delta = steps_to_cost(costs, safety_horizon=80)
    assert np.array_equal(delta, np.full(5, 80))


def test_single_cost_at_end():
    # cost at index 4; states before it count down, the cost state itself -> H_s.
    costs = [0, 0, 0, 0, 1]
    delta = steps_to_cost(costs, safety_horizon=80)
    assert np.array_equal(delta, np.array([4, 3, 2, 1, 80]))


def test_single_cost_at_start_has_no_future_cost():
    # Documented quirk: an unsafe state with nothing after it is labeled H_s,
    # and so is everything else (no future cost anywhere).
    costs = [1, 0, 0, 0, 0]
    delta = steps_to_cost(costs, safety_horizon=80)
    assert np.array_equal(delta, np.full(5, 80))


def test_multiple_costs_uses_nearest_future():
    # costs at indices 2 and 5.
    costs = [0, 0, 1, 0, 0, 1, 0]
    delta = steps_to_cost(costs, safety_horizon=80)
    #  idx0->2 (d2), idx1->2 (d1), idx2->5 (d3), idx3->5 (d2),
    #  idx4->5 (d1), idx5->none (80), idx6->none (80)
    assert np.array_equal(delta, np.array([2, 1, 3, 2, 1, 80, 80]))


def test_consecutive_cost_cluster():
    # In-hazard states (except the last) point to the next adjacent cost (d=1).
    costs = [0, 1, 1, 1, 0]
    delta = steps_to_cost(costs, safety_horizon=80)
    assert np.array_equal(delta, np.array([1, 1, 1, 80, 80]))


def test_clipping_to_horizon():
    # Far-away cost: states more than H_s steps before it are capped at H_s.
    H_s = 4
    costs = [0, 0, 0, 0, 0, 0, 1]  # cost at index 6
    delta = steps_to_cost(costs, safety_horizon=H_s)
    #  idx6->none->4, idx5->1, idx4->2, idx3->3, idx2->4(cap), idx1->4(cap), idx0->4(cap)
    assert np.array_equal(delta, np.array([4, 4, 4, 3, 2, 1, 4]))


def test_delta_range_is_one_to_horizon():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = int(rng.integers(1, 200))
        costs = (rng.random(T) < 0.1).astype(float)
        delta = steps_to_cost(costs, safety_horizon=80)
        assert delta.min() >= 1
        assert delta.max() <= 80


def test_empty_trajectory():
    delta = steps_to_cost([], safety_horizon=80)
    assert delta.shape == (0,)
    labels = label_trajectory([], 80, 4)
    assert labels.shape == (0,)


def test_dtype_is_int64():
    delta = steps_to_cost([0, 1, 0], 80)
    assert delta.dtype == np.int64
    labels = label_trajectory([0, 1, 0], 80, 4)
    assert labels.dtype == np.int64


# --------------------------------------------------------------------------- #
# Property test: vectorized == reference oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cost_prob", [0.0, 0.02, 0.1, 0.5, 1.0])
def test_vectorized_matches_reference(cost_prob):
    rng = np.random.default_rng(42)
    for _ in range(200):
        T = int(rng.integers(0, 300))
        H_s = int(rng.integers(1, 120))
        costs = (rng.random(T) < cost_prob).astype(float)
        fast = steps_to_cost(costs, safety_horizon=H_s)
        slow = _steps_to_cost_reference(costs, safety_horizon=H_s)
        assert np.array_equal(fast, slow), (
            f"mismatch: T={T}, H_s={H_s}, p={cost_prob}\n"
            f"costs={costs.astype(int).tolist()}\nfast={fast.tolist()}\n"
            f"slow={slow.tolist()}"
        )


def test_nonbinary_costs_treated_as_unsafe_when_positive():
    # Costs need not be 0/1; any value > 0 is an unsafe event.
    costs = [0.0, 0.3, 0.0, 2.5, 0.0]  # unsafe at indices 1 and 3
    delta = steps_to_cost(costs, safety_horizon=80)
    #  idx0->1 (d1), idx1->3 (d2, next cost STRICTLY after 1), idx2->3 (d1),
    #  idx3->none (80), idx4->none (80)
    assert np.array_equal(delta, np.array([1, 2, 1, 80, 80]))


# --------------------------------------------------------------------------- #
# delta_to_bin
# --------------------------------------------------------------------------- #
def test_delta_to_bin_boundaries_default():
    # H_s=80, bin_size=4 -> K=20. Bin b covers delta in [4b+1, 4b+4].
    H_s, bs = 80, 4
    assert delta_to_bin(1, H_s, bs) == 0
    assert delta_to_bin(4, H_s, bs) == 0
    assert delta_to_bin(5, H_s, bs) == 1
    assert delta_to_bin(8, H_s, bs) == 1
    assert delta_to_bin(80, H_s, bs) == 19   # last bin
    # No-cost label (delta = H_s) lands in the last bin.
    assert delta_to_bin(H_s, H_s, bs) == 19


def test_delta_to_bin_clips_above_horizon():
    # Defensive: values >= H_s collapse into the final bin.
    assert delta_to_bin(999, 80, 4) == 19


def test_delta_to_bin_vectorized():
    H_s, bs = 80, 4
    delta = np.array([1, 4, 5, 8, 9, 80])
    expected = np.array([0, 0, 1, 1, 2, 19])
    assert np.array_equal(delta_to_bin(delta, H_s, bs), expected)


def test_bins_within_range_random():
    rng = np.random.default_rng(7)
    for _ in range(100):
        T = int(rng.integers(1, 200))
        H_s = int(rng.integers(2, 120))
        # pick a divisor-ish bin size
        bs = int(rng.integers(1, max(2, H_s // 2)))
        K = H_s // bs
        costs = (rng.random(T) < 0.1).astype(float)
        labels = label_trajectory(costs, H_s, bs)
        assert labels.min() >= 0
        assert labels.max() <= K - 1


def test_bin_size_one_is_identity_shift():
    # With bin_size=1, bin == delta-1.
    costs = [0, 0, 1, 0, 0]
    delta = steps_to_cost(costs, 80)
    labels = label_trajectory(costs, 80, 1)
    assert np.array_equal(labels, delta - 1)


# --------------------------------------------------------------------------- #
# label_trajectory end-to-end
# --------------------------------------------------------------------------- #
def test_label_trajectory_matches_manual():
    costs = [0, 0, 1, 0, 0, 1, 0]
    # deltas = [2,1,3,2,1,80,80]; with bin_size=4 all small deltas -> bin 0.
    labels = label_trajectory(costs, 80, 4)
    assert np.array_equal(labels, np.array([0, 0, 0, 0, 0, 19, 19]))


# --------------------------------------------------------------------------- #
# PerEnvEpisodeAccumulator
# --------------------------------------------------------------------------- #
def _obs_vec(env_idx, step, dim=3):
    """Deterministic dummy observation so we can assert ordering/segmentation."""
    return np.full(dim, env_idx * 1000 + step, dtype=np.float64)


def test_accumulator_single_env_flush_on_done():
    acc = PerEnvEpisodeAccumulator(num_envs=1, safety_horizon=80, bin_size=4)
    costs = [0, 0, 1, 0]  # delta = [2,1,80,80] -> bins [0,0,19,19]
    out = None
    for t, c in enumerate(costs):
        done = (t == len(costs) - 1)
        out = acc.add(0, _obs_vec(0, t), c, done)
    assert out is not None
    ep_obs, ep_labels = out
    assert ep_obs.shape == (4, 3)
    assert np.array_equal(ep_labels, np.array([0, 0, 19, 19]))
    # Buffer cleared after flush.
    assert acc.pending_lengths() == [0]


def test_accumulator_segments_envs_independently():
    # Two envs with DIFFERENT episode lengths and cost patterns, stepped together.
    # This is the exact scenario where a naive flatten would corrupt labels.
    acc = PerEnvEpisodeAccumulator(num_envs=2, safety_horizon=80, bin_size=4)

    # env0 episode A: length 3, cost at local idx 1 -> delta [1,80,80]
    # env1 episode A: length 5, cost at local idx 3 -> delta [3,2,1,80,80]
    env0_costs = [0, 1, 0]
    env1_costs = [0, 0, 0, 1, 0]

    # Build per-step batches up to the max length; envs end at different steps.
    # We model independent resets: env0 ends at global step 2, env1 at step 4.
    finished_all = []
    # Step through 5 global steps; after env0 finishes it starts a fresh episode.
    # To keep the test deterministic we only feed env0 its 3 steps then idle it
    # by starting a new (length-1, no-cost) episode that we don't assert on.
    # Simpler: drive each env on its own timeline via add().
    for t, c in enumerate(env0_costs):
        out = acc.add(0, _obs_vec(0, t), c, done=(t == len(env0_costs) - 1))
        if out is not None:
            finished_all.append(("env0", out))
    for t, c in enumerate(env1_costs):
        out = acc.add(1, _obs_vec(1, t), c, done=(t == len(env1_costs) - 1))
        if out is not None:
            finished_all.append(("env1", out))

    assert len(finished_all) == 2
    by_env = dict(finished_all)

    env0_obs, env0_labels = by_env["env0"]
    assert env0_obs.shape == (3, 3)
    # delta [1,80,80] -> bins [0,19,19]
    assert np.array_equal(env0_labels, np.array([0, 19, 19]))
    # Observations are env0's, in order.
    assert np.array_equal(env0_obs[:, 0], np.array([0, 1, 2]))

    env1_obs, env1_labels = by_env["env1"]
    assert env1_obs.shape == (5, 3)
    # delta [3,2,1,80,80] -> bins [0,0,0,19,19]
    assert np.array_equal(env1_labels, np.array([0, 0, 0, 19, 19]))
    assert np.array_equal(env1_obs[:, 0], np.array([1000, 1001, 1002, 1003, 1004]))


def test_accumulator_add_batch_returns_finished_in_env_order():
    acc = PerEnvEpisodeAccumulator(num_envs=3, safety_horizon=80, bin_size=4)
    dim = 2
    # Step 0: nobody done.
    obs = np.stack([_obs_vec(i, 0, dim) for i in range(3)])
    fin = acc.add_batch(obs, cost_batch=[0, 0, 0], done_batch=[False, False, False])
    assert fin == []
    # Step 1: env0 and env2 finish (env1 keeps going).
    obs = np.stack([_obs_vec(i, 1, dim) for i in range(3)])
    fin = acc.add_batch(obs, cost_batch=[1, 0, 0], done_batch=[True, False, True])
    assert len(fin) == 2
    (obs0, lab0), (obs2, lab2) = fin
    # env0: costs [0,1] -> delta [1,80] -> bins [0,19]
    assert np.array_equal(lab0, np.array([0, 19]))
    # env2: costs [0,0] -> delta [80,80] -> bins [19,19]
    assert np.array_equal(lab2, np.array([19, 19]))
    # env1 still has 2 pending steps.
    assert acc.pending_lengths() == [0, 2, 0]


def test_accumulator_validates_shapes():
    acc = PerEnvEpisodeAccumulator(num_envs=2)
    with pytest.raises(ValueError):
        acc.add_batch(np.zeros((3, 4)), [0, 0, 0], [False, False, False])
    with pytest.raises(IndexError):
        acc.add(5, np.zeros(4), 0.0, False)


def test_accumulator_reset_clears_partial():
    acc = PerEnvEpisodeAccumulator(num_envs=2)
    acc.add(0, np.zeros(3), 0.0, done=False)
    acc.add(1, np.zeros(3), 1.0, done=False)
    assert acc.pending_lengths() == [1, 1]
    acc.reset()
    assert acc.pending_lengths() == [0, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
