"""SRPL: Safety Representations for Safer Policy Learning (reproduction).

This package implements the SRPL contribution (Mani et al., ICLR 2025) on top of
OmniSafe base algorithms. The SRPL-specific components have no public reference
implementation and are written from scratch here:

    - labeling      : steps-to-cost (delta) labeling of trajectories
    - s2c_model     : the steps-to-cost (S2C) neural network        (Stage 2+)
    - s2c_buffer    : FIFO / replay storage for S2C training data    (Stage 2+)
    - augmentation  : detached state augmentation                    (Stage 2+)
    - algorithms    : SRPL-wrapped PPO-Lag / TD3-Lag / SAC-Lag       (Stage 2+)
"""

from srpl.labeling import (
    steps_to_cost,
    delta_to_bin,
    label_trajectory,
    num_bins,
    PerEnvEpisodeAccumulator,
)
from srpl.s2c_model import S2CModel, augment_observation

__all__ = [
    # labeling
    "steps_to_cost",
    "delta_to_bin",
    "label_trajectory",
    "num_bins",
    "PerEnvEpisodeAccumulator",
    # s2c model
    "S2CModel",
    "augment_observation",
]
