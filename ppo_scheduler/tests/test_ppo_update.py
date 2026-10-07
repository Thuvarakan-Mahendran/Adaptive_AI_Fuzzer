"""
tests/test_ppo_update.py – PPO action selection, update, and numerical stability (§25.5).
"""

import copy

import numpy as np
import pytest
import torch

from ppo_scheduler.agent import PPOAgent
from ppo_scheduler.buffer import RolloutBuffer
from ppo_scheduler.config import PPOConfig
from ppo_scheduler.state_preprocessor import StatePreprocessor


@pytest.fixture
def cfg():
    # Force CPU in tests to avoid device mismatch on CUDA-capable hosts
    return PPOConfig(ACTION_DIM=4, ROLLOUT_LENGTH=32, PPO_EPOCHS=2,
                     MINIBATCH_SIZE=8, DEVICE="cpu")


def make_valid_state() -> np.ndarray:
    s = np.zeros(192, dtype=np.float32)
    s[0:10] = 1.0
    s[128:130] = 0.5
    s[162] = 1.0
    s[176:192] = 0.3
    return s


def fill_buffer(buf: RolloutBuffer, cfg: PPOConfig, agent: PPOAgent,
                preprocessor: StatePreprocessor):
    state = make_valid_state()
    for _ in range(cfg.ROLLOUT_LENGTH):
        action, log_prob, value = agent.select_action(state)
        next_state = make_valid_state()
        # Store raw states; agent.update() normalizes them internally
        buf.add(state, action, 1.0, False, value, log_prob, next_state)
        state = next_state


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def test_action_in_valid_range(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    state = make_valid_state()
    for _ in range(20):
        action, _, _ = agent.select_action(state)
        assert 0 <= action < cfg.ACTION_DIM


def test_log_prob_finite(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    state = make_valid_state()
    _, log_prob, _ = agent.select_action(state)
    assert np.isfinite(log_prob), f"log_prob={log_prob}"


def test_value_finite(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    state = make_valid_state()
    _, _, value = agent.select_action(state)
    assert np.isfinite(value), f"value={value}"


def test_deterministic_same_action(cfg):
    """Deterministic mode always returns the same action for the same state."""
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    state = make_valid_state()
    actions = {agent.select_action(state, deterministic=True)[0] for _ in range(5)}
    assert len(actions) == 1, "Deterministic mode returned different actions"


# ---------------------------------------------------------------------------
# PPO update changes parameters
# ---------------------------------------------------------------------------

def test_ppo_update_changes_parameters(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    buf = RolloutBuffer(cfg, device="cpu")
    fill_buffer(buf, cfg, agent, preprocessor)

    # snapshot parameters before update
    before = {n: p.data.clone() for n, p in agent.actor.named_parameters()}

    agent.update(buf)

    after = {n: p.data for n, p in agent.actor.named_parameters()}
    changed = any(not torch.equal(before[n], after[n]) for n in before)
    assert changed, "PPO update did not change any actor parameters"


# ---------------------------------------------------------------------------
# Loss values
# ---------------------------------------------------------------------------

def test_ppo_update_returns_finite_losses(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    buf = RolloutBuffer(cfg, device="cpu")
    fill_buffer(buf, cfg, agent, preprocessor)

    stats = agent.update(buf)
    for k, v in stats.items():
        assert np.isfinite(v), f"Loss {k}={v} is non-finite"


def test_entropy_positive(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    buf = RolloutBuffer(cfg, device="cpu")
    fill_buffer(buf, cfg, agent, preprocessor)
    stats = agent.update(buf)
    assert stats["entropy"] > 0, f"Entropy should be positive, got {stats['entropy']}"


# ---------------------------------------------------------------------------
# No NaN introduced
# ---------------------------------------------------------------------------

def test_no_nan_after_update(cfg):
    preprocessor = StatePreprocessor(cfg)
    agent = PPOAgent(cfg, preprocessor)
    buf = RolloutBuffer(cfg, device="cpu")
    fill_buffer(buf, cfg, agent, preprocessor)
    agent.update(buf)

    for name, param in agent.actor.named_parameters():
        assert not torch.any(torch.isnan(param.data)), f"NaN in actor.{name}"
    for name, param in agent.critic.named_parameters():
        assert not torch.any(torch.isnan(param.data)), f"NaN in critic.{name}"
