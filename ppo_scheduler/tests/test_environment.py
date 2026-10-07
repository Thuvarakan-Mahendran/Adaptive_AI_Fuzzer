"""
tests/test_environment.py – Mock environment tests (§25.6).
"""

import numpy as np
import pytest

from ppo_scheduler.config import PPOConfig
from ppo_scheduler.environment import MockMutationEnvironment
from ppo_scheduler.state_preprocessor import validate_state


@pytest.fixture
def cfg():
    return PPOConfig(ACTION_DIM=4)


@pytest.fixture
def env(cfg):
    return MockMutationEnvironment(cfg, max_steps=20, rng_seed=0)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_returns_valid_state(env, cfg):
    state = env.reset()
    assert state.shape == (192,)
    validate_state(state, action_dim=cfg.ACTION_DIM)


def test_reset_is_deterministic(cfg):
    e1 = MockMutationEnvironment(cfg, rng_seed=0)
    e2 = MockMutationEnvironment(cfg, rng_seed=0)
    s1 = e1.reset()
    s2 = e2.reset()
    assert np.allclose(s1, s2), "reset() not deterministic for same seed"


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

def test_step_returns_valid_state(env, cfg):
    env.reset()
    for action in range(cfg.ACTION_DIM):
        env.reset()
        next_state, reward, done, info = env.step(action)
        assert next_state.shape == (192,)
        validate_state(next_state, action_dim=cfg.ACTION_DIM)


def test_step_reward_components_in_info(env):
    env.reset()
    _, reward, _, info = env.step(0)
    assert "delta_coverage" in info
    assert "i_crash"        in info
    assert "i_novelty"      in info
    assert "reward"         in info


def test_reward_formula(env, cfg):
    """Verify reward = Δcoverage + λ_crash * I_crash + λ_novelty * I_novelty."""
    env.reset()
    _, reward, _, info = env.step(0)
    expected = (
        info["delta_coverage"]
        + cfg.LAMBDA_CRASH   * info["i_crash"]
        + cfg.LAMBDA_NOVELTY * info["i_novelty"]
    )
    assert abs(reward - expected) < 1e-5, f"Reward mismatch: {reward} vs {expected}"


def test_invalid_action_raises(env, cfg):
    env.reset()
    with pytest.raises(ValueError, match="out of range"):
        env.step(cfg.ACTION_DIM)  # one past end


def test_done_after_max_steps(cfg):
    env = MockMutationEnvironment(cfg, max_steps=5, rng_seed=0)
    env.reset()
    done = False
    for i in range(5):
        _, _, done, _ = env.step(0)
    assert done, "Environment should be done after max_steps"


def test_state_dim_always_192(env, cfg):
    env.reset()
    for _ in range(30):
        action = np.random.randint(0, cfg.ACTION_DIM)
        next_state, _, done, _ = env.step(action)
        assert next_state.shape == (192,)
        if done:
            env.reset()


def test_coverage_bits_binary(env, cfg):
    """Coverage section must remain binary throughout an episode."""
    state = env.reset()
    for _ in range(20):
        action = 0
        next_state, _, done, _ = env.step(action)
        cov = next_state[0:128]
        assert set(cov.tolist()).issubset({0.0, 1.0}), \
            f"Non-binary coverage bits: {set(cov.tolist())}"
        state = next_state
        if done:
            break
