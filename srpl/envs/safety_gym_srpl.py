"""SRPL environment wrapper for Safety-Gymnasium, integrated with OmniSafe.

This is the injection point that lets us reproduce SRPL *without modifying
OmniSafe's algorithms*. We subclass OmniSafe's own ``SafetyGymnasiumEnv`` and
change only what SRPL requires:

  1. The environment owns the S2C model, an episode accumulator (for
     steps-to-cost labeling), a FIFO buffer of ``(raw_obs, bin_label)`` training
     pairs, and an optimizer.
  2. The advertised ``observation_space`` becomes the AUGMENTED space
     (raw_dim + K), e.g. 60 + 20 = 80 for SafetyPointGoal1-v0.
  3. ``reset`` / ``step`` return the augmented observation
     ``s' = concat(s, S2C(s).detach())`` and, on the SRPL path, record each
     transition's ``(raw_obs, cost)``; at episode end the episode is labeled and
     the S2C is trained every ``s2c_update_freq`` environment steps.
  4. A single ``use_sr`` flag selects SR vs. baseline. In baseline mode the
     safety slot is filled with a constant so that BASE and SR see an
     identically shaped 80-dim observation — the only difference is whether the
     extra coordinates carry real safety information. This keeps the comparison
     clean (same network input size, same parameter count downstream).

Because OmniSafe then sees a perfectly ordinary CMDP whose observation happens
to be 80-dimensional, ``PPOLag`` / ``TD3Lag`` / ``SACLag`` train on it with zero
changes to OmniSafe internals.

Design notes / correctness:
  * OmniSafe uses ``autoreset=True`` + AutoReset wrapper for num_envs=1: on the
    step that ends an episode, ``step`` already returns the NEXT episode's first
    observation, while the true terminal observation is in
    ``info['final_observation']``. The cost returned on that step still belongs
    to the ending episode, so we record ``(raw_obs, cost)`` for the CURRENT
    transition BEFORE flushing, then flush on done. ``PerEnvEpisodeAccumulator``
    handles the per-episode segmentation.
  * The S2C output fed into the observation is always detached
    (:meth:`S2CModel.safety_representation`), so policy/critic gradients never
    reach the S2C. The S2C is trained solely by its own NLL loss here.
  * On-policy default hyperparameters (Appendix A.3.2): H_s=80, bin_size=4
    (K=20), S2C MLP [64,64], lr 1e-5, batch 5000, update_freq 100. These are
    overridable via env kwargs so the same class serves the off-policy regime
    (lr 1e-3, batch 512, update_freq 20000) by passing different values.

This module currently supports num_envs == 1 (sufficient for the reproduction;
OmniSafe's SafetyGymnasiumEnv vectorization path is not used here because the
per-episode S2C labeling is simplest and least bug-prone with a single stream).
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch
from gymnasium.spaces import Box

from omnisafe.envs.core import CMDP, env_register
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv

from srpl.labeling import (
    DEFAULT_SAFETY_HORIZON,
    DEFAULT_BIN_SIZE,
    label_trajectory,
)
from srpl.s2c_model import S2CModel


# Map each SRPL env id to the base Safety-Gymnasium id it wraps.
# We expose NEW ids so OmniSafe routes them to THIS class (not the base wrapper).
# Two id families per task encode the SR-vs-baseline choice WITHOUT relying on
# OmniSafe passing custom env kwargs through its config: '...SRPL-v0' turns
# augmentation + S2C training ON, '...Base-v0' turns it OFF (constant-filled
# safety slot, identical observation shape). use_sr can still be overridden via
# an explicit kwarg if needed.
_SR_IDS: dict[str, str] = {
    "SafetyPointGoal1SRPL-v0": "SafetyPointGoal1-v0",
    "SafetyPointButton1SRPL-v0": "SafetyPointButton1-v0",
}
_BASE_IDS: dict[str, str] = {
    "SafetyPointGoal1Base-v0": "SafetyPointGoal1-v0",
    "SafetyPointButton1Base-v0": "SafetyPointButton1-v0",
}
_SRPL_TO_BASE: dict[str, str] = {**_SR_IDS, **_BASE_IDS}


# --------------------------------------------------------------------------- #
# S2C configuration hook
# --------------------------------------------------------------------------- #
# OmniSafe forwards `env_cfgs` to the env constructor ONLY for on-policy
# algorithms (PPOLag lists `env_cfgs: {}`; the off-policy YAMLs do not, and
# OmniSafe's recursive_check_config rejects unknown keys with a KeyError). To
# pass S2C hyperparameters to the wrapper uniformly across on- AND off-policy
# algorithms, the training script sets this module-level override BEFORE
# constructing the OmniSafe Agent; the wrapper's __init__ reads it as the
# default for any S2C kwarg not explicitly provided.
#
# Precedence in __init__:  explicit kwarg  >  module override  >  hard default.
_S2C_CONFIG_OVERRIDE: dict[str, Any] = {}


def set_s2c_config(**kwargs: Any) -> None:
    """Set S2C hyperparameters for SafetyGymSRPLEnv instances built afterwards.

    Recognized keys: ``s2c_lr``, ``s2c_batch_size``, ``s2c_update_freq``,
    ``s2c_hidden_sizes``, ``s2c_activation``, ``safety_horizon``, ``bin_size``,
    ``s2c_buffer_capacity``, ``s2c_warmup``, ``baseline_fill``, and the transfer
    keys ``frozen_s2c`` (bool), ``load_s2c_path`` (str), ``s2c_input_dim`` (int).

    Call with no arguments (or ``clear_s2c_config()``) to reset to hard defaults.
    """
    _S2C_CONFIG_OVERRIDE.clear()
    _S2C_CONFIG_OVERRIDE.update(kwargs)


def clear_s2c_config() -> None:
    """Clear any S2C override so subsequent envs use the hard defaults."""
    _S2C_CONFIG_OVERRIDE.clear()


def _cfg(kwargs: dict, key: str, hard_default: Any) -> Any:
    """Resolve a config value: explicit kwarg > module override > hard default."""
    if key in kwargs:
        return kwargs.pop(key)
    if key in _S2C_CONFIG_OVERRIDE:
        return _S2C_CONFIG_OVERRIDE[key]
    return hard_default


@env_register
class SafetyGymSRPLEnv(SafetyGymnasiumEnv):
    """Safety-Gymnasium + SRPL state augmentation, as an OmniSafe CMDP.

    Args:
        env_id: one of the keys in ``_SRPL_TO_BASE`` (e.g.
            ``'SafetyPointGoal1SRPL-v0'``).
        num_envs: must be 1 (only the single-stream path is implemented).
        device: torch device for the S2C model and tensors.

    Keyword Args (all optional; defaults match the paper's on-policy setting):
        use_sr (bool): if True, augment with the learned S2C output and train the
            S2C online; if False (baseline), fill the safety slot with
            ``baseline_fill`` and never train the S2C. Default True.
        safety_horizon (int): H_s. Default 80.
        bin_size (int): steps per bin. Default 4 (=> K=20).
        s2c_hidden_sizes (tuple): S2C MLP hidden widths. Default (64, 64).
        s2c_lr (float): S2C optimizer learning rate. Default 1e-5 (on-policy).
        s2c_batch_size (int): minibatch for an S2C update. Default 5000.
        s2c_update_freq (int): train S2C every this many env steps. Default 100.
        s2c_buffer_capacity (int): FIFO capacity in transitions. Default 100000.
        s2c_warmup (int): env steps to collect before the first S2C update.
            Default = s2c_batch_size.
        baseline_fill (float): constant used to fill the safety slot when
            use_sr is False. Default 0.0.
        seed (int): optional seed for the S2C parameter init / RNG.
    """

    # Advertise ONLY the SRPL ids so OmniSafe's registry routes them here with
    # no conflict against the base SafetyGymnasiumEnv (which claims the base
    # ids). We deliberately do NOT list the base ids here; instead __init__
    # calls CMDP.__init__ with our (supported) SRPL id and builds the
    # underlying Safety-Gymnasium env with the base id directly.
    _support_envs: ClassVar[list[str]] = list(_SRPL_TO_BASE.keys())

    # SRPL augmentation already returns the next-episode obs via the base class;
    # keep the same wrapper requirements the base class uses for num_envs=1.
    need_auto_reset_wrapper: bool = True
    need_time_limit_wrapper: bool = True

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        if env_id not in _SRPL_TO_BASE:
            raise ValueError(
                f"{env_id!r} is not an SRPL env id. "
                f"Expected one of {list(_SRPL_TO_BASE)}."
            )
        if num_envs != 1:
            raise NotImplementedError("SafetyGymSRPLEnv supports num_envs=1 only.")

        # ---- SRPL-specific kwargs ----
        # use_sr defaults from the id family ('...SRPL-v0' -> True, '...Base-v0'
        # -> False) but an explicit kwarg overrides it. Note use_sr is NOT taken
        # from the module override (it's an identity property of the env id).
        default_use_sr = env_id in _SR_IDS
        self._use_sr: bool = bool(kwargs.pop("use_sr", default_use_sr))
        # All S2C hyperparameters resolve via: explicit kwarg > module override
        # (set_s2c_config) > hard default. This makes the off-policy config reach
        # the wrapper even though OmniSafe won't forward env_cfgs for off-policy.
        self._safety_horizon: int = int(_cfg(kwargs, "safety_horizon", DEFAULT_SAFETY_HORIZON))
        self._bin_size: int = int(_cfg(kwargs, "bin_size", DEFAULT_BIN_SIZE))
        s2c_hidden_sizes = tuple(_cfg(kwargs, "s2c_hidden_sizes", (64, 64)))
        s2c_activation = str(_cfg(kwargs, "s2c_activation", "tanh")).lower()
        self._s2c_lr: float = float(_cfg(kwargs, "s2c_lr", 1e-5))
        self._s2c_batch_size: int = int(_cfg(kwargs, "s2c_batch_size", 5000))
        self._s2c_update_freq: int = int(_cfg(kwargs, "s2c_update_freq", 100))
        self._s2c_buffer_capacity: int = int(_cfg(kwargs, "s2c_buffer_capacity", 100_000))
        self._baseline_fill: float = float(_cfg(kwargs, "baseline_fill", 0.0))
        self._s2c_warmup: int = int(_cfg(kwargs, "s2c_warmup", self._s2c_batch_size))
        # ---- Transfer support (Figure 6) ----
        # frozen_s2c: never train the S2C (used when applying a transferred or a
        #   random-initialized-but-frozen S2C).
        # load_s2c_path: load S2C weights from this file (implies frozen).
        # s2c_input_dim: build the S2C for this input dim and pad/truncate the
        #   raw obs to it before the S2C forward, so an S2C trained on a source
        #   task (e.g. PointButton1, raw dim 76) can be applied to a target task
        #   (PointGoal1, raw dim 60) whose raw obs is zero-padded to 76. The
        #   policy still sees native_obs + K; only the S2C's input is padded.
        self._frozen_s2c: bool = bool(_cfg(kwargs, "frozen_s2c", False))
        self._load_s2c_path = _cfg(kwargs, "load_s2c_path", None)
        _s2c_input_dim_arg = int(_cfg(kwargs, "s2c_input_dim", 0))
        if self._load_s2c_path is not None:
            self._frozen_s2c = True  # loading a checkpoint implies freezing
        seed = kwargs.pop("seed", None)

        # ---- Build the underlying Safety-Gymnasium env ----
        # We call CMDP.__init__ directly with OUR id (which is in _support_envs,
        # so its assertion passes), then construct the Safety-Gymnasium env with
        # the BASE id ourselves. This mirrors SafetyGymnasiumEnv.__init__ for the
        # num_envs==1 path but avoids its assertion that the id equals a base id.
        base_id = _SRPL_TO_BASE[env_id]
        CMDP.__init__(self, env_id)  # passes: env_id in self.support_envs()

        self._num_envs = num_envs
        self._device = torch.device(device)
        # Match the base wrapper's num_envs==1 settings.
        self.need_time_limit_wrapper = True
        self.need_auto_reset_wrapper = True
        self._env = safety_gymnasium.make(id=base_id, autoreset=True, **kwargs)
        assert isinstance(self._env.action_space, Box), "Only support Box action space."
        assert isinstance(self._env.observation_space, Box), \
            "Only support Box observation space."
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata
        # Keep the SRPL id as our identity even though we built `base_id`.
        self._env_id = env_id

        # ---- Raw dims from the base observation space, then override to aug ----
        assert isinstance(self._observation_space, Box)
        self._raw_obs_dim: int = int(self._observation_space.shape[0])
        # Effective S2C input dim: the transfer override if given, else native.
        self._s2c_input_dim: int = _s2c_input_dim_arg or self._raw_obs_dim

        if seed is not None:
            torch.manual_seed(int(seed))
        _activation_map = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}
        if s2c_activation not in _activation_map:
            raise ValueError(
                f"s2c_activation must be one of {list(_activation_map)}, "
                f"got {s2c_activation!r}"
            )
        self._s2c = S2CModel(
            obs_dim=self._s2c_input_dim,
            safety_horizon=self._safety_horizon,
            bin_size=self._bin_size,
            hidden_sizes=s2c_hidden_sizes,
            activation=_activation_map[s2c_activation],
        ).to(self._device)
        self._K: int = self._s2c.num_bins
        # Load transferred weights if requested.
        if self._load_s2c_path is not None:
            state = torch.load(self._load_s2c_path, map_location=self._device)
            # Accept either a raw state_dict or a {'s2c_state_dict': ...} wrapper.
            if isinstance(state, dict) and "s2c_state_dict" in state:
                state = state["s2c_state_dict"]
            self._s2c.load_state_dict(state)
        # Freeze parameters if this S2C is not to be trained.
        if self._frozen_s2c:
            for p in self._s2c.parameters():
                p.requires_grad_(False)
            self._s2c.eval()
        # Only train when augmenting AND not frozen.
        self._train_s2c: bool = self._use_sr and not self._frozen_s2c
        self._optimizer = torch.optim.Adam(self._s2c.parameters(), lr=self._s2c_lr)

        # One-line config echo: confirms env_cfgs reached the wrapper and
        # documents the exact S2C setup used for this run (reproducibility).
        _transfer = ""
        if self._frozen_s2c:
            src = self._load_s2c_path if self._load_s2c_path else "random-init"
            _transfer = (f" | FROZEN S2C ({src}), s2c_input_dim="
                         f"{self._s2c_input_dim} (raw={self._raw_obs_dim})")
        print(
            f"[SafetyGymSRPLEnv] {env_id} | use_sr={self._use_sr} "
            f"| S2C: hidden={tuple(s2c_hidden_sizes)} act={s2c_activation} "
            f"lr={self._s2c_lr:g} batch={self._s2c_batch_size} "
            f"update_freq={self._s2c_update_freq} H_s={self._safety_horizon} "
            f"bin_size={self._bin_size} (K={self._K}){_transfer}",
            flush=True,
        )

        # Override the advertised observation space to the AUGMENTED space.
        low = np.concatenate(
            [self._observation_space.low, np.zeros(self._K, dtype=np.float32)]
        )
        high = np.concatenate(
            [self._observation_space.high, np.ones(self._K, dtype=np.float32)]
        )
        self._observation_space = Box(low=low, high=high, dtype=np.float32)

        # ---- SRPL bookkeeping ----
        # Single-stream episode buffers (num_envs=1).
        self._ep_obs: list[np.ndarray] = []
        self._ep_costs: list[float] = []
        # FIFO of labeled training pairs for the S2C.
        self._buf_obs: deque[np.ndarray] = deque(maxlen=self._s2c_buffer_capacity)
        self._buf_lbl: deque[int] = deque(maxlen=self._s2c_buffer_capacity)
        self._env_steps: int = 0

        # Per-episode diagnostics surfaced to the OmniSafe logger via spec_log.
        self.env_spec_log: dict[str, Any] = {
            "Metrics/S2C_LossLast": 0.0,
            "Metrics/S2C_Updates": 0,
        }
        self._s2c_loss_last: float = 0.0
        self._s2c_updates: int = 0

    # --------------------------------------------------------------------- #
    # Observation augmentation
    # --------------------------------------------------------------------- #
    def _pad_for_s2c(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Match the raw obs to the S2C's expected input dim (transfer support).

        Zero-pads (if the native obs is smaller, e.g. PointGoal1's 60 -> 76) or
        truncates (if larger) the LAST dimension to ``self._s2c_input_dim``. This
        lets an S2C trained on a source task be applied to a target task whose
        raw observation differs in size. No-op when dims already match.
        """
        cur = raw_obs.shape[-1]
        if cur == self._s2c_input_dim:
            return raw_obs
        if cur < self._s2c_input_dim:
            pad_shape = raw_obs.shape[:-1] + (self._s2c_input_dim - cur,)
            pad = torch.zeros(pad_shape, dtype=raw_obs.dtype, device=raw_obs.device)
            return torch.cat([raw_obs, pad], dim=-1)
        return raw_obs[..., : self._s2c_input_dim]

    def _augment(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Return concat(raw_obs, safety_slot) along the last dim.

        Shape-robust: handles a single observation ``(raw_dim,)`` and a batched
        ``(B, raw_dim)`` (needed for ``final_observation`` under the auto-reset
        wrapper). Output last dim is ``raw_dim + K``.

        SR mode: safety_slot = detached S2C(pad(raw_obs)). The S2C input is
        padded/truncated to ``s2c_input_dim`` for transfer; the augmented
        observation still uses the NATIVE raw_obs, so its dim is raw_dim + K.
        Baseline: safety_slot = constant ``baseline_fill`` (no S2C involvement),
        so the observation has identical shape but carries no safety info.
        """
        raw_obs = raw_obs.to(self._device)
        if self._use_sr:
            s2c_in = self._pad_for_s2c(raw_obs)
            safety = self._s2c.safety_representation(s2c_in)  # detached, (...,K)
        else:
            slot_shape = raw_obs.shape[:-1] + (self._K,)
            safety = torch.full(
                slot_shape, self._baseline_fill,
                dtype=torch.float32, device=self._device,
            )
        return torch.cat([raw_obs, safety], dim=-1)

    # --------------------------------------------------------------------- #
    # CMDP API
    # --------------------------------------------------------------------- #
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raw_obs, info = super().reset(seed=seed, options=options)
        # Starting a fresh episode stream.
        self._ep_obs = [raw_obs.detach().cpu().numpy()]
        self._ep_costs = []
        return self._augment(raw_obs), info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        # Base step returns the real Safety-Gymnasium (obs, reward, cost, term, trunc, info).
        raw_next_obs, reward, cost, terminated, truncated, info = super().step(action)
        self._env_steps += 1

        if self._train_s2c:
            # Record the cost incurred on THIS transition (belongs to the
            # current/ending episode), and the raw observation just produced.
            self._ep_costs.append(float(cost.item()))

            done = bool(terminated.item()) or bool(truncated.item())
            if done:
                # On auto-reset, the true terminal obs is in final_observation;
                # raw_next_obs is already the NEXT episode's first obs.
                self._flush_episode()
                # Begin the next episode's obs stream with the returned obs.
                self._ep_obs = [raw_next_obs.detach().cpu().numpy()]
                self._ep_costs = []
            else:
                self._ep_obs.append(raw_next_obs.detach().cpu().numpy())

            self._maybe_train_s2c()

        aug_obs = self._augment(raw_next_obs)

        # The auto-reset wrapper / adapter bootstraps the value from
        # final_observation, so it must be the AUGMENTED (raw_dim + K) shape.
        # Guard against double-augmentation: only augment if it is still raw-dim.
        if info.get("final_observation", None) is not None:
            fobs = info["final_observation"]
            if isinstance(fobs, torch.Tensor) and fobs.shape[-1] == self._raw_obs_dim:
                info = dict(info)
                info["final_observation"] = self._augment(fobs)

        return aug_obs, reward, cost, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # S2C training
    # --------------------------------------------------------------------- #
    def _flush_episode(self) -> None:
        """Label the just-finished episode and push pairs into the FIFO buffer.

        Alignment (see the off-by-one note): ``self._ep_costs[i]`` is the cost of
        the transition LEAVING state ``self._ep_obs[i]`` (i.e. produced by the
        action taken in ``s_i``). The paper defines delta(s_i) as the number of
        actions until an unsafe state, so a cost on the very next transition must
        give delta = 1. We achieve exactly that by prepending a sentinel 0 to the
        per-step cost sequence and reusing ``label_trajectory``:

            delta(s_i) == label_trajectory([0] + costs)[i]
                       == (first j >= i with c_j > 0) - i + 1   (clipped to H_s)

        ``ep_obs`` and ``ep_costs`` have equal length T at flush time, so we take
        the first T labels and align one-to-one.
        """
        if not self._ep_costs:
            return
        costs = np.asarray(self._ep_costs, dtype=np.float64)
        # Prepend sentinel so a next-step cost maps to delta = 1 (see docstring).
        costs_prepended = np.concatenate([[0.0], costs])
        labels_full = label_trajectory(
            costs_prepended, self._safety_horizon, self._bin_size
        )
        labels = labels_full[: len(costs)]

        obs_arr = np.asarray(self._ep_obs[: len(costs)], dtype=np.float32)
        n = min(len(obs_arr), len(labels))
        for i in range(n):
            self._buf_obs.append(obs_arr[i])
            self._buf_lbl.append(int(labels[i]))

    def _maybe_train_s2c(self) -> None:
        """Train the S2C every ``s2c_update_freq`` steps once warmed up."""
        if len(self._buf_obs) < max(self._s2c_warmup, 1):
            return
        if self._env_steps % self._s2c_update_freq != 0:
            return

        batch = min(self._s2c_batch_size, len(self._buf_obs))
        idx = np.random.randint(0, len(self._buf_obs), size=batch)
        obs_np = np.stack([self._buf_obs[i] for i in idx], axis=0)
        lbl_np = np.asarray([self._buf_lbl[i] for i in idx], dtype=np.int64)

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self._device)
        lbl_t = torch.as_tensor(lbl_np, dtype=torch.long, device=self._device)

        self._s2c.train()
        self._optimizer.zero_grad()
        loss = self._s2c.nll_loss(obs_t, lbl_t)
        loss.backward()
        self._optimizer.step()

        self._s2c_loss_last = float(loss.item())
        self._s2c_updates += 1

    # --------------------------------------------------------------------- #
    # Logging hook (called by OmniSafe at the end of each episode)
    # --------------------------------------------------------------------- #
    def spec_log(self, logger: Any) -> None:
        """Surface S2C diagnostics to the OmniSafe logger."""
        try:
            logger.store(
                {
                    "Metrics/S2C_LossLast": self._s2c_loss_last,
                    "Metrics/S2C_Updates": float(self._s2c_updates),
                }
            )
        except Exception:
            # If these keys are not registered in the logger, skip silently.
            pass

    # --------------------------------------------------------------------- #
    # Persistence (so the trained S2C can be saved/loaded, e.g. for transfer)
    # --------------------------------------------------------------------- #
    @property
    def s2c_model(self) -> S2CModel:
        """Access the underlying S2C model (e.g. to freeze/save for transfer)."""
        return self._s2c

    def save(self) -> dict[str, torch.nn.Module]:
        """Include the S2C in OmniSafe's saved modules dict."""
        saved = {}
        try:
            saved = dict(super().save())
        except Exception:
            saved = {}
        saved["s2c"] = self._s2c
        return saved
