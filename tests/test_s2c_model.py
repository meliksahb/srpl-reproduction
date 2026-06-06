"""Unit tests for srpl.s2c_model.

Covers: output shapes/normalization, the K=20 default, the NLL/cross-entropy
loss, that the S2C can actually learn a trivial mapping, and — most importantly
— the gradient-isolation guarantee: gradients from a consumer of the augmented
observation must NEVER flow into the S2C parameters.

Run:  pytest -q tests/test_s2c_model.py
"""

import numpy as np
import pytest
import torch

from srpl.s2c_model import S2CModel, augment_observation
from srpl.labeling import label_trajectory


OBS_DIM = 60  # SafetyPointGoal1-v0 / SafetyPointButton1-v0 raw observation dim


# --------------------------------------------------------------------------- #
# Construction / shapes
# --------------------------------------------------------------------------- #
def test_default_num_bins_and_dims():
    m = S2CModel(obs_dim=OBS_DIM)            # H_s=80, bin_size=4 -> K=20
    assert m.num_bins == 20
    assert m.augmented_obs_dim == OBS_DIM + 20  # == 80


def test_forward_batched_shape_and_normalization():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(8, OBS_DIM)
    p = m(x)
    assert p.shape == (8, 20)
    # Rows are probability distributions.
    assert torch.allclose(p.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert (p >= 0).all()


def test_forward_unbatched_shape():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(OBS_DIM)
    p = m(x)
    assert p.shape == (20,)
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-5)


def test_logits_shape():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(4, OBS_DIM)
    assert m.logits(x).shape == (4, 20)


def test_architecture_two_hidden_layers_of_64():
    # Paper: MLP with two hidden layers of size 64. Check Linear shapes.
    m = S2CModel(obs_dim=OBS_DIM, hidden_sizes=(64, 64))
    linears = [layer for layer in m.net if isinstance(layer, torch.nn.Linear)]
    assert len(linears) == 3  # 2 hidden + 1 output
    assert linears[0].in_features == OBS_DIM and linears[0].out_features == 64
    assert linears[1].in_features == 64 and linears[1].out_features == 64
    assert linears[2].in_features == 64 and linears[2].out_features == 20


def test_custom_horizon_bin():
    m = S2CModel(obs_dim=10, safety_horizon=40, bin_size=4)  # K = 10
    assert m.num_bins == 10
    assert m(torch.randn(3, 10)).shape == (3, 10)


def test_rejects_bad_obs_dim():
    with pytest.raises(ValueError):
        S2CModel(obs_dim=0)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def test_augment_observation_shape_batched():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(5, OBS_DIM)
    aug = augment_observation(x, m)
    assert aug.shape == (5, OBS_DIM + 20)
    # First OBS_DIM columns are the untouched original observation.
    assert torch.allclose(aug[:, :OBS_DIM], x)
    # Trailing K columns form a valid distribution per row.
    assert torch.allclose(aug[:, OBS_DIM:].sum(dim=-1), torch.ones(5), atol=1e-5)


def test_augment_observation_shape_unbatched():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(OBS_DIM)
    aug = augment_observation(x, m)
    assert aug.shape == (OBS_DIM + 20,)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_nll_loss_is_scalar_and_finite():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(16, OBS_DIM)
    labels = torch.randint(0, 20, (16,))
    loss = m.nll_loss(x, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_nll_loss_accepts_numpy_int_labels_via_tensor():
    # Labels coming straight from label_trajectory are int64 numpy; wrap them.
    m = S2CModel(obs_dim=OBS_DIM)
    costs = [0, 0, 1, 0, 0, 1, 0]
    np_labels = label_trajectory(costs, 80, 4)           # int64 array, len 7
    x = torch.randn(len(np_labels), OBS_DIM)
    loss = m.nll_loss(x, torch.from_numpy(np_labels))
    assert torch.isfinite(loss)


def test_nll_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(7, OBS_DIM)
    labels = torch.randint(0, 20, (7,))
    expected = torch.nn.functional.cross_entropy(m.logits(x), labels)
    assert torch.allclose(m.nll_loss(x, labels), expected)


def test_s2c_can_learn_trivial_mapping():
    # Sanity: the model can fit a simple deterministic state->bin rule, i.e. the
    # training loop / loss actually push the loss down.
    torch.manual_seed(0)
    K = 20
    m = S2CModel(obs_dim=8, safety_horizon=80, bin_size=4)
    # Construct a separable dataset: bin determined by argmax of a fixed proj.
    N = 512
    X = torch.randn(N, 8)
    W = torch.randn(8, K)
    y = (X @ W).argmax(dim=-1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = m.nll_loss(X, y).item()
    for _ in range(300):
        opt.zero_grad()
        loss = m.nll_loss(X, y)
        loss.backward()
        opt.step()
    last = m.nll_loss(X, y).item()
    assert last < first * 0.5, f"loss did not drop enough: {first:.3f} -> {last:.3f}"


# --------------------------------------------------------------------------- #
# CRITICAL: gradient isolation
# --------------------------------------------------------------------------- #
def test_safety_representation_is_detached():
    m = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(4, OBS_DIM)
    rep = m.safety_representation(x)
    assert not rep.requires_grad


def test_policy_gradient_does_not_reach_s2c():
    """The make-or-break guarantee.

    Simulate a downstream 'policy' consuming the augmented observation and
    backpropagating its OWN loss. No gradient may land on the S2C parameters.
    """
    torch.manual_seed(0)
    s2c = S2CModel(obs_dim=OBS_DIM)
    policy_head = torch.nn.Linear(s2c.augmented_obs_dim, 1)  # toy policy

    x = torch.randn(8, OBS_DIM)
    aug = augment_observation(x, s2c)          # uses detached S2C output
    policy_out = policy_head(aug)
    policy_loss = policy_out.pow(2).mean()
    policy_loss.backward()

    # Policy head must receive gradient...
    assert policy_head.weight.grad is not None
    assert policy_head.weight.grad.abs().sum() > 0
    # ...but every S2C parameter must have NO gradient.
    for name, p in s2c.named_parameters():
        assert p.grad is None, f"S2C param '{name}' unexpectedly received a gradient"


def test_s2c_own_loss_does_reach_s2c():
    """Conversely, the S2C's own loss MUST produce gradients on its params."""
    torch.manual_seed(0)
    s2c = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(8, OBS_DIM)
    labels = torch.randint(0, 20, (8,))
    s2c.nll_loss(x, labels).backward()
    grads = [p.grad for _, p in s2c.named_parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_augmented_obs_does_not_share_autograd_with_s2c_after_detach():
    # Even if obs itself requires grad (e.g. some upstream feature encoder),
    # the S2C branch is detached, so S2C params stay grad-free.
    s2c = S2CModel(obs_dim=OBS_DIM)
    x = torch.randn(3, OBS_DIM, requires_grad=True)
    aug = augment_observation(x, s2c)
    aug.sum().backward()
    # x can have grad (from the identity branch); S2C params must not.
    for _, p in s2c.named_parameters():
        assert p.grad is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
