"""The steps-to-cost (S2C) model for SRPL.

Reference: Mani et al., "Safety Representations for Safer Policy Learning",
ICLR 2025 (arXiv:2502.20341), Section 3.3 and Appendix A.3.2.

The S2C model is the heart of the SRPL contribution. It is a small neural
network that maps a raw state ``s`` to a categorical distribution over K
"steps-to-cost" bins (see srpl/labeling.py for how the bin targets are built).
That distribution is the *safety representation*: bin 0 means "a cost is
imminent (1-`bin_size` steps away)", the last bin means "safe within the
horizon H_s".

How it is used in the SRPL framework (Fig. 2 of the paper):

    raw obs  s ──► S2C ──► p = softmax(...)  ∈ Δ^K   (the safety representation)
                              │
              augmented obs:  s' = concat(s, p.detach())  ──► actor / critic

Two separate gradient paths:
    * The S2C is trained ONLY by its own NLL/cross-entropy loss against the
      steps-to-cost bin labels.
    * The policy/critic consume the S2C output but must NOT backpropagate into
      it, otherwise the policy objective would corrupt the safety estimator.
      This module exposes ``safety_representation()`` which returns a detached
      tensor specifically to make that guarantee explicit at the call site.

Architecture (Appendix A.3.2): "the same network architecture as the policy
which in most cases is an MLP with two hidden layers of size 64." For the
Safety-Gym tasks we therefore default to hidden=(64, 64) and K=20
(H_s=80, bin_size=4). The input dimension for SafetyPointGoal1-v0 /
SafetyPointButton1-v0 is 60, so the augmented observation is 60 + 20 = 80.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from srpl.labeling import DEFAULT_SAFETY_HORIZON, DEFAULT_BIN_SIZE, num_bins


class S2CModel(nn.Module):
    """Steps-to-cost model: state -> categorical distribution over K bins.

    Args:
        obs_dim: dimensionality of the RAW observation (the S2C input). This is
            the original env observation, NOT the safety-augmented one. For
            SafetyPointGoal1-v0 / SafetyPointButton1-v0 this is 60.
        safety_horizon: H_s (default 80).
        bin_size: steps per bin (default 4) => K = H_s // bin_size output bins.
        hidden_sizes: MLP hidden layer widths (default (64, 64), matching the
            paper's policy architecture).
        activation: hidden-layer activation module class (default Tanh, the
            OmniSafe on-policy default; pass nn.ReLU for off-policy parity if
            the base algorithm uses ReLU).

    Shape conventions:
        forward / logits / safety_representation accept obs of shape
        (obs_dim,) or (batch, obs_dim) and return (K,) or (batch, K)
        respectively.
    """

    def __init__(self,
                 obs_dim: int,
                 safety_horizon: int = DEFAULT_SAFETY_HORIZON,
                 bin_size: int = DEFAULT_BIN_SIZE,
                 hidden_sizes: Sequence[int] = (64, 64),
                 activation: type[nn.Module] = nn.Tanh) -> None:
        super().__init__()
        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        self.obs_dim = int(obs_dim)
        self.safety_horizon = int(safety_horizon)
        self.bin_size = int(bin_size)
        self.num_bins = num_bins(safety_horizon, bin_size)  # validates H_s/bin
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        layers: list[nn.Module] = []
        last = self.obs_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, self.num_bins))  # logits over K bins
        self.net = nn.Sequential(*layers)

    # ----------------------------------------------------------------- #
    # Forward variants
    # ----------------------------------------------------------------- #
    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw (pre-softmax) logits over the K bins. Shape (..., K)."""
        return self.net(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Safety representation as a probability distribution. Shape (..., K).

        NOTE: this is differentiable w.r.t. the S2C parameters and is what the
        S2C's own training step uses (via :meth:`nll_loss`). For feeding the
        policy/critic, use :meth:`safety_representation` instead, which detaches.
        """
        return F.softmax(self.logits(obs), dim=-1)

    @torch.no_grad()
    def safety_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """Detached safety representation for state augmentation.

        Returned tensor has ``requires_grad=False`` so that concatenating it
        into the policy/critic input cannot backpropagate into the S2C. This is
        the function the SRPL augmentation wrapper should call.
        """
        return F.softmax(self.logits(obs), dim=-1)

    # ----------------------------------------------------------------- #
    # Training loss
    # ----------------------------------------------------------------- #
    def nll_loss(self, obs: torch.Tensor, bin_labels: torch.Tensor) -> torch.Tensor:
        """Negative-log-likelihood (== cross-entropy) loss for the S2C model.

        Implements Eq. (3) of the paper. For a categorical distribution the NLL
        with one-hot targets is exactly cross-entropy on the bin index, so we
        use the numerically stable ``F.cross_entropy`` on the logits.

        Args:
            obs: (batch, obs_dim) raw observations.
            bin_labels: (batch,) int64 bin indices in {0, ..., K-1}, as produced
                by ``srpl.labeling.label_trajectory``.

        Returns:
            Scalar mean cross-entropy loss.
        """
        if bin_labels.dtype != torch.long:
            bin_labels = bin_labels.long()
        return F.cross_entropy(self.logits(obs), bin_labels)

    # ----------------------------------------------------------------- #
    # Convenience
    # ----------------------------------------------------------------- #
    @property
    def augmented_obs_dim(self) -> int:
        """Dimensionality of the augmented observation = obs_dim + K."""
        return self.obs_dim + self.num_bins

    def extra_repr(self) -> str:
        return (f"obs_dim={self.obs_dim}, num_bins={self.num_bins} "
                f"(H_s={self.safety_horizon}, bin_size={self.bin_size}), "
                f"hidden_sizes={self.hidden_sizes} -> "
                f"augmented_obs_dim={self.augmented_obs_dim}")


def augment_observation(obs: torch.Tensor, s2c: S2CModel) -> torch.Tensor:
    """Concatenate the DETACHED S2C safety representation onto ``obs``.

        s' = concat(s, S2C(s).detach())          (Section 3.3, augmented state)

    Works for a single observation (obs_dim,) or a batch (batch, obs_dim);
    returns (obs_dim + K,) or (batch, obs_dim + K) respectively. The S2C output
    is detached (via :meth:`S2CModel.safety_representation`), so gradients from
    whatever consumes the augmented observation never reach the S2C.
    """
    safety = s2c.safety_representation(obs)
    return torch.cat([obs, safety], dim=-1)
