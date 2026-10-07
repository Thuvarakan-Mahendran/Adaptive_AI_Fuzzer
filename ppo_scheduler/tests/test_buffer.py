"""
tests/test_buffer.py – RolloutBuffer storage, GAE, and clearing tests (§25.4).
"""

import numpy as np
import pytest
import torch

from ppo_scheduler.buffer import RolloutBuffer
from ppo_scheduler.config import PPOConfig


@pytest.fixture
def cfg():
    return PPOConfig(ACTION_DIM=4, ROLLOUT_LENGTH=8)


def make_state() -> np.ndarray:
    s = np.zeros(192, dtype=np.float32)
    s[0:5] = 1.0   # some coverage bits
    s[128] = 0.5
    s[162] = 1.0   # one-hot last-op at idx 2
    s[176:192] = 0.3
    return s


def fill_buffer(buf: RolloutBuffer, cfg: PPOConfig) -> None:
    for i in range(cfg.ROLLOUT_LENGTH):
        buf.add(
            state      = make_state(),
            action     = i % cfg.ACTION_DIM,
            reward     = float(i),
            done       = (i == cfg.ROLLOUT_LENGTH - 1),
            value      = 1.0,
            log_prob   = -0.5,
            next_state = make_state(),
        )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_transitions_stored(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    assert len(buf) == cfg.ROLLOUT_LENGTH


def test_buffer_is_full(cfg):
    buf = RolloutBuffer(cfg)
    assert not buf.is_full()
    fill_buffer(buf, cfg)
    assert buf.is_full()


def test_add_beyond_capacity_raises(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    with pytest.raises(RuntimeError, match="full"):
        buf.add(make_state(), 0, 0.0, False, 0.0, -0.1, make_state())


# ---------------------------------------------------------------------------
# Old log-probs unchanged
# ---------------------------------------------------------------------------

def test_old_log_probs_unchanged(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    original = list(buf._old_log_probs)  # snapshot before GAE
    buf.compute_returns_and_advantages(next_value=0.0,
                                       gamma=cfg.GAMMA,
                                       gae_lambda=cfg.GAE_LAMBDA)
    assert buf._old_log_probs == original, "Old log-probs were modified during GAE computation"


# ---------------------------------------------------------------------------
# GAE shapes
# ---------------------------------------------------------------------------

def test_gae_output_shape(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    buf.compute_returns_and_advantages(0.0, cfg.GAMMA, cfg.GAE_LAMBDA)
    assert buf._advantages.shape == (cfg.ROLLOUT_LENGTH,)
    assert buf._returns.shape    == (cfg.ROLLOUT_LENGTH,)


def test_returns_advantages_finite(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    buf.compute_returns_and_advantages(0.0, cfg.GAMMA, cfg.GAE_LAMBDA)
    assert torch.all(torch.isfinite(buf._advantages))
    assert torch.all(torch.isfinite(buf._returns))


def test_advantage_normalized(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    buf.compute_returns_and_advantages(0.0, cfg.GAMMA, cfg.GAE_LAMBDA)
    adv = buf._advantages.numpy()
    assert abs(adv.mean()) < 0.5, "Advantages should be approximately zero-mean"


# ---------------------------------------------------------------------------
# Minibatch iteration
# ---------------------------------------------------------------------------

def test_minibatches_cover_all_data(cfg):
    buf = RolloutBuffer(cfg, device="cpu")
    fill_buffer(buf, cfg)
    buf.compute_returns_and_advantages(0.0, cfg.GAMMA, cfg.GAE_LAMBDA)
    total = 0
    for batch in buf.get_minibatches(4):
        states, actions, lps, advs, rets = batch
        total += states.shape[0]
    assert total == cfg.ROLLOUT_LENGTH


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_buffer_clears(cfg):
    buf = RolloutBuffer(cfg)
    fill_buffer(buf, cfg)
    buf.compute_returns_and_advantages(0.0, cfg.GAMMA, cfg.GAE_LAMBDA)
    buf.clear()
    assert len(buf) == 0
    assert buf._advantages is None
    assert buf._returns is None
    assert not buf.is_full()
