"""Steps-to-cost (delta) labeling for the SRPL S2C model.

Reference: Mani et al., "Safety Representations for Safer Policy Learning",
ICLR 2025 (arXiv:2502.20341), Section 3.3 and Appendix A.3.2.

------------------------------------------------------------------------------
What the S2C model predicts
------------------------------------------------------------------------------
For a state ``s``, the S2C model outputs a categorical distribution over
"steps-to-cost": the number of environment steps until the agent next enters a
cost-inducing (unsafe) state. This module produces the *training labels* for
that model from observed trajectories.

------------------------------------------------------------------------------
Labeling convention (FIXED — read before changing anything)
------------------------------------------------------------------------------
Given a 1-D array ``costs`` for a single episode, where ``costs[t] > 0`` means
the environment returned a positive cost at step ``t`` (i.e. step ``t`` is an
"unsafe event"):

    delta[t] = (distance to the next unsafe event at an index STRICTLY > t),
               clipped to the safety horizon H_s;
               H_s if there is no unsafe event after t.

Consequences of "strictly after", all intentional and matching the paper:

  * delta is always in {1, ..., H_s}.  delta = 1 means an unsafe event is
    exactly one step away (the riskiest non-cost state); see Fig. 3 in the
    paper where the risky state B peaks at delta = 1.
  * "No unsafe event encountered" => delta = H_s  (Section 3.3: the distance for
    all states is set to H_s to indicate safety within the horizon).
  * An unsafe state with NO later unsafe event (e.g. an isolated or terminal
    cost) is labeled by the distance to the *next* cost, or H_s if none. The
    S2C predicts FUTURE proximity, so a terminal cost state is "safe within
    horizon" w.r.t. what comes after it. In Safety-Gym, hazards are usually
    multi-step regions, so states *inside* a hazard cluster get small delta
    (good signal); only the last consecutive cost step looks farther ahead.

Because H_s << episode length (H_s = 80 vs. ~1000 steps in Safety-Gym), most
states are labeled H_s -> the last bin. This skew is expected; do not "correct"
it.

------------------------------------------------------------------------------
Binning
------------------------------------------------------------------------------
To reduce the output dimensionality from H_s to K, delta in {1, ..., H_s} is
mapped to one of K = H_s // bin_size bins:

    bin = clip((delta - 1) // bin_size, 0, K - 1)

For Safety-Gym (PointGoal1 / PointButton1) the paper uses H_s = 80, bin_size = 4
=> K = 20 (Appendix A.3.2). ``bin_size`` is the number of steps per bin, NOT the
number of bins; smaller bin_size -> more bins -> finer resolution.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Defaults for the Safety-Gym tasks used in this reproduction (Appendix A.3.2).
DEFAULT_SAFETY_HORIZON: int = 80
DEFAULT_BIN_SIZE: int = 4


def num_bins(safety_horizon: int = DEFAULT_SAFETY_HORIZON,
             bin_size: int = DEFAULT_BIN_SIZE) -> int:
    """Return K, the number of S2C output bins (= safety_horizon // bin_size)."""
    _validate_horizon_bin(safety_horizon, bin_size)
    return safety_horizon // bin_size


def _validate_horizon_bin(safety_horizon: int, bin_size: int) -> None:
    if safety_horizon < 1:
        raise ValueError(f"safety_horizon must be >= 1, got {safety_horizon}")
    if bin_size < 1:
        raise ValueError(f"bin_size must be >= 1, got {bin_size}")
    if bin_size > safety_horizon:
        raise ValueError(
            f"bin_size ({bin_size}) must be <= safety_horizon ({safety_horizon})"
        )


def steps_to_cost(costs: ArrayLike,
                  safety_horizon: int = DEFAULT_SAFETY_HORIZON) -> NDArray[np.int64]:
    """Compute raw steps-to-cost (delta) for one episode.

    Vectorized O(T log C) implementation (C = number of cost events). See module
    docstring for the exact convention.

    Args:
        costs: 1-D array-like of per-step costs for ONE episode. ``costs[t] > 0``
            marks step ``t`` as an unsafe event.
        safety_horizon: H_s. delta is clipped to this value; states with no
            future unsafe event within the horizon are labeled H_s.

    Returns:
        int64 array of shape (T,) with delta[t] in {1, ..., H_s}.
    """
    if safety_horizon < 1:
        raise ValueError(f"safety_horizon must be >= 1, got {safety_horizon}")

    costs_arr = np.asarray(costs)
    if costs_arr.ndim != 1:
        raise ValueError(f"costs must be 1-D, got shape {costs_arr.shape}")
    T = costs_arr.shape[0]
    H_s = int(safety_horizon)

    if T == 0:
        return np.empty(0, dtype=np.int64)

    # Indices of unsafe events.
    cost_idx = np.flatnonzero(costs_arr > 0)

    if cost_idx.size == 0:
        # No unsafe event anywhere -> every state is safe within the horizon.
        return np.full(T, H_s, dtype=np.int64)

    t = np.arange(T)
    # pos = index into cost_idx of the first cost event STRICTLY after t.
    pos = np.searchsorted(cost_idx, t, side="right")
    has_next = pos < cost_idx.size

    # Guard the gather against out-of-range pos for states with no later cost.
    safe_pos = np.where(has_next, pos, 0)
    next_cost_index = cost_idx[safe_pos]
    delta_if_next = np.minimum(next_cost_index - t, H_s)

    delta = np.where(has_next, delta_if_next, H_s).astype(np.int64)
    return delta


def _steps_to_cost_reference(costs: ArrayLike,
                             safety_horizon: int = DEFAULT_SAFETY_HORIZON
                             ) -> NDArray[np.int64]:
    """Slow, obviously-correct backward-recurrence implementation.

    Kept only as a correctness oracle for the vectorized ``steps_to_cost`` (see
    the property test). Not used in the training pipeline.

    Recurrence (going backward over t):
        delta[T-1] = H_s
        delta[t]   = 1                         if costs[t+1] > 0
                   = min(delta[t+1] + 1, H_s)   otherwise
    """
    costs_arr = np.asarray(costs)
    T = costs_arr.shape[0]
    H_s = int(safety_horizon)
    delta = np.full(T, H_s, dtype=np.int64)
    for t in range(T - 2, -1, -1):
        if costs_arr[t + 1] > 0:
            delta[t] = 1
        else:
            delta[t] = min(int(delta[t + 1]) + 1, H_s)
    return delta


def delta_to_bin(delta: ArrayLike,
                 safety_horizon: int = DEFAULT_SAFETY_HORIZON,
                 bin_size: int = DEFAULT_BIN_SIZE) -> NDArray[np.int64]:
    """Map raw delta in {1, ..., H_s} to a bin index in {0, ..., K-1}.

        bin = clip((delta - 1) // bin_size, 0, K - 1),  K = H_s // bin_size
    """
    _validate_horizon_bin(safety_horizon, bin_size)
    K = safety_horizon // bin_size
    delta_arr = np.asarray(delta, dtype=np.int64)
    bins = (delta_arr - 1) // bin_size
    return np.clip(bins, 0, K - 1).astype(np.int64)


def label_trajectory(costs: ArrayLike,
                     safety_horizon: int = DEFAULT_SAFETY_HORIZON,
                     bin_size: int = DEFAULT_BIN_SIZE) -> NDArray[np.int64]:
    """Full pipeline: per-episode costs -> S2C bin labels.

    Composition of :func:`steps_to_cost` and :func:`delta_to_bin`.

    Args:
        costs: 1-D per-step costs for ONE episode (``costs[t] > 0`` == unsafe).
        safety_horizon: H_s (default 80).
        bin_size: steps per bin (default 4) => K = H_s // bin_size bins.

    Returns:
        int64 array (T,) of bin labels, each in {0, ..., K-1}, suitable as
        cross-entropy targets for the S2C model.
    """
    delta = steps_to_cost(costs, safety_horizon)
    return delta_to_bin(delta, safety_horizon, bin_size)


class PerEnvEpisodeAccumulator:
    """Segment trajectories per parallel env and emit labeled data at episode end.

    With N vectorized environments the per-step cost stream is interleaved across
    envs, and episodes end at different times. Labeling requires *complete*
    per-episode cost sequences, so a naive flatten across envs would corrupt
    every label. This accumulator keeps one in-progress buffer per env and, when
    an env's episode terminates/truncates, returns that episode's observations
    together with their S2C bin labels.

    Typical use inside a rollout loop::

        acc = PerEnvEpisodeAccumulator(num_envs, H_s, bin_size)
        for obs, cost, done in rollout:                  # batched over envs
            finished = acc.add_batch(obs, cost, done)
            for ep_obs, ep_labels in finished:
                s2c_buffer.add(ep_obs, ep_labels)

    Notes:
        * ``obs`` stored here should be the *raw* observation (the S2C input),
          not the safety-augmented observation.
        * Labels for a finished episode depend only on that episode's costs, so
          partial episodes still in progress are never (mis)labeled.
    """

    def __init__(self,
                 num_envs: int,
                 safety_horizon: int = DEFAULT_SAFETY_HORIZON,
                 bin_size: int = DEFAULT_BIN_SIZE) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        _validate_horizon_bin(safety_horizon, bin_size)
        self.num_envs = int(num_envs)
        self.safety_horizon = int(safety_horizon)
        self.bin_size = int(bin_size)
        self._obs: List[List[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._costs: List[List[float]] = [[] for _ in range(self.num_envs)]

    def add(self, env_idx: int, obs: ArrayLike, cost: float, done: bool
            ) -> Tuple[NDArray, NDArray[np.int64]] | None:
        """Add one (obs, cost) step for ``env_idx``; flush if the episode ended.

        Returns ``(episode_obs, episode_labels)`` when ``done`` is True, else None.
        """
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        self._obs[env_idx].append(np.asarray(obs))
        self._costs[env_idx].append(float(cost))
        if done:
            return self._flush(env_idx)
        return None

    def add_batch(self,
                  obs_batch: ArrayLike,
                  cost_batch: ArrayLike,
                  done_batch: ArrayLike
                  ) -> List[Tuple[NDArray, NDArray[np.int64]]]:
        """Add one batched step across all envs; return all episodes that ended.

        Args:
            obs_batch:  (num_envs, obs_dim) raw observations.
            cost_batch: (num_envs,) per-env costs.
            done_batch: (num_envs,) per-env episode-end flags (terminated OR
                        truncated).

        Returns:
            List of ``(episode_obs, episode_labels)`` for every env whose episode
            ended on this step (possibly empty), in ascending env-index order.
        """
        obs_batch = np.asarray(obs_batch)
        cost_batch = np.asarray(cost_batch)
        done_batch = np.asarray(done_batch)
        if obs_batch.shape[0] != self.num_envs:
            raise ValueError(
                f"obs_batch first dim {obs_batch.shape[0]} != num_envs "
                f"{self.num_envs}"
            )
        if cost_batch.shape[0] != self.num_envs or done_batch.shape[0] != self.num_envs:
            raise ValueError("cost_batch / done_batch must have length num_envs")

        finished: List[Tuple[NDArray, NDArray[np.int64]]] = []
        for env_idx in range(self.num_envs):
            out = self.add(env_idx,
                           obs_batch[env_idx],
                           cost_batch[env_idx],
                           bool(done_batch[env_idx]))
            if out is not None:
                finished.append(out)
        return finished

    def _flush(self, env_idx: int) -> Tuple[NDArray, NDArray[np.int64]]:
        """Label and clear the completed episode for ``env_idx``."""
        costs = np.asarray(self._costs[env_idx], dtype=np.float64)
        labels = label_trajectory(costs, self.safety_horizon, self.bin_size)
        obs = np.stack(self._obs[env_idx], axis=0) if self._obs[env_idx] \
            else np.empty((0,))
        self._obs[env_idx] = []
        self._costs[env_idx] = []
        return obs, labels

    def pending_lengths(self) -> List[int]:
        """Number of buffered (un-flushed) steps per env. Useful for tests/debug."""
        return [len(c) for c in self._costs]

    def reset(self) -> None:
        """Drop all buffered partial episodes (e.g. on a hard env reset)."""
        self._obs = [[] for _ in range(self.num_envs)]
        self._costs = [[] for _ in range(self.num_envs)]
