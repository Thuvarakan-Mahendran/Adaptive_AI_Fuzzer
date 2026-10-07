"""
tests/test_networks.py – Actor and Critic shape/output tests (§25.3).
"""

import torch
import pytest

from ppo_scheduler.config import PPOConfig
from ppo_scheduler.networks import Actor, Critic


@pytest.fixture
def cfg():
    return PPOConfig(ACTION_DIM=8)


def test_actor_output_shape(cfg):
    actor = Actor(cfg)
    x = torch.zeros(1, 192)
    out = actor(x)
    assert out.shape == (1, cfg.ACTION_DIM), f"Expected (1,{cfg.ACTION_DIM}), got {out.shape}"


def test_critic_output_shape(cfg):
    critic = Critic(cfg)
    x = torch.zeros(1, 192)
    out = critic(x)
    assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"


def test_actor_batch(cfg):
    actor = Actor(cfg)
    x = torch.zeros(32, 192)
    out = actor(x)
    assert out.shape == (32, cfg.ACTION_DIM)


def test_critic_batch(cfg):
    critic = Critic(cfg)
    x = torch.zeros(32, 192)
    out = critic(x)
    assert out.shape == (32, 1)


def test_actor_finite_output(cfg):
    actor = Actor(cfg)
    x = torch.randn(16, 192)
    out = actor(x)
    assert torch.all(torch.isfinite(out)), "Actor output contains non-finite values"


def test_critic_finite_output(cfg):
    critic = Critic(cfg)
    x = torch.randn(16, 192)
    out = critic(x)
    assert torch.all(torch.isfinite(out)), "Critic output contains non-finite values"


def test_actor_distribution(cfg):
    """Actor distribution sums to 1 (probability)."""
    from torch.distributions import Categorical
    actor = Actor(cfg)
    x = torch.zeros(4, 192)
    dist = actor.get_distribution(x)
    probs_sum = dist.probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(4), atol=1e-5), \
        "Categorical probabilities do not sum to 1"


def test_invalid_state_dim_wrong_shape(cfg):
    """Network on wrong input dimension raises RuntimeError."""
    actor = Actor(cfg)
    x = torch.zeros(1, 100)   # wrong dim
    with pytest.raises(RuntimeError):
        actor(x)
