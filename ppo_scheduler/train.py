"""
train.py – End-to-end PPO training loop (§22, §28).

Usage
-----
    python -m ppo_scheduler.train
    python -m ppo_scheduler.train --num_rollouts 200 --action_dim 4

The loop follows §22 exactly:
  initialize environment → initialize agent
  while training:
      state = env.reset()
      for step in rollout (ROLLOUT_LENGTH):
          action, log_prob, value = agent.select_action(state)
          next_state, reward, done, info = env.step(action)
          buffer.add(...)
          state = next_state (or env.reset() on episode end)
          if buffer.is_full():
              agent.update(buffer)
              buffer.clear()
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import Deque, Dict, List

import numpy as np
import torch

from ppo_scheduler.agent import PPOAgent
from ppo_scheduler.buffer import RolloutBuffer
from ppo_scheduler.config import PPOConfig
from ppo_scheduler.environment import MockMutationEnvironment
from ppo_scheduler.state_preprocessor import StatePreprocessor


# ---------------------------------------------------------------------------
# Reproducibility (§26)
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch for reproducibility.

    Note: CUDA operations may still be nondeterministic by default.
    Set torch.backends.cudnn.deterministic = True if full CUDA
    determinism is required (at a performance cost).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: PPOConfig, num_rollouts: int = 100) -> None:
    """Run the PPO training loop against the mock environment.

    Args:
        config:       PPOConfig with all hyperparameters.
        num_rollouts: Total number of rollout updates to perform.
    """
    seed_everything(config.SEED)
    device = config.resolve_device()
    print(f"[train] device={device}  action_dim={config.ACTION_DIM}  "
          f"rollout_length={config.ROLLOUT_LENGTH}  ppo_epochs={config.PPO_EPOCHS}")

    # Components
    preprocessor = StatePreprocessor(config)
    preprocessor.train_mode()

    env    = MockMutationEnvironment(config, rng_seed=config.SEED)
    agent  = PPOAgent(config, preprocessor)
    buffer = RolloutBuffer(config, device=device)

    # Logging accumulators
    ep_rewards:     Deque[float] = deque(maxlen=20)
    ep_coverages:   Deque[float] = deque(maxlen=20)
    ep_crashes:     Deque[int]   = deque(maxlen=20)
    ep_novelties:   Deque[int]   = deque(maxlen=20)
    action_counts:  List[int]    = [0] * config.ACTION_DIM

    state = env.reset()
    ep_reward   = 0.0
    ep_coverage = 0.0
    ep_crashes_ep  = 0
    ep_novelties_ep = 0

    for rollout_idx in range(num_rollouts):
        for _step in range(config.ROLLOUT_LENGTH):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            buffer.add(
                state      = state,
                action     = action,
                reward     = reward,
                done       = done,
                value      = value,
                log_prob   = log_prob,
                next_state = next_state,
            )

            # Track episode stats
            ep_reward         += reward
            ep_coverage        = info["coverage_bits"]
            ep_crashes_ep     += info["i_crash"]
            ep_novelties_ep   += info["i_novelty"]
            action_counts[action] += 1

            state = next_state
            if done:
                ep_rewards.append(ep_reward)
                ep_coverages.append(ep_coverage)
                ep_crashes.append(ep_crashes_ep)
                ep_novelties.append(ep_novelties_ep)
                ep_reward = ep_crashes_ep = ep_novelties_ep = 0
                state = env.reset()

            # --- Rollout boundary: update and reset buffer ---
            if buffer.is_full():
                train_stats = agent.update(buffer)
                buffer.clear()

                # Periodic logging (§28)
                if (rollout_idx + 1) % config.LOG_INTERVAL == 0:
                    mean_rew = np.mean(ep_rewards) if ep_rewards else 0.0
                    mean_cov = np.mean(ep_coverages) if ep_coverages else 0.0
                    mean_cr  = np.mean(ep_crashes)   if ep_crashes  else 0.0
                    mean_nov = np.mean(ep_novelties) if ep_novelties else 0.0
                    act_dist = [f"{a:.0f}" for a in action_counts]
                    print(
                        f"[rollout {rollout_idx+1:04d}] "
                        f"mean_reward={mean_rew:.3f}  "
                        f"coverage_bits={mean_cov:.1f}  "
                        f"crashes={mean_cr:.2f}  "
                        f"novelties={mean_nov:.2f}  "
                        f"policy_loss={train_stats['policy_loss']:.4f}  "
                        f"value_loss={train_stats['value_loss']:.4f}  "
                        f"entropy={train_stats['entropy']:.4f}  "
                        f"actions={act_dist}"
                    )

        # Checkpoint every 50 rollouts
        if (rollout_idx + 1) % 50 == 0:
            ckpt_path = os.path.join(
                config.CHECKPOINT_DIR, f"step_{rollout_idx+1:04d}.pt"
            )
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            agent.save(ckpt_path)
            print(f"[train] Checkpoint saved → {ckpt_path}")

    print("[train] Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO mutation-strategy scheduler training")
    p.add_argument("--num_rollouts", type=int, default=100)
    p.add_argument("--action_dim",   type=int, default=8)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(
        ACTION_DIM      = args.action_dim,
        SEED            = args.seed,
        LOG_INTERVAL    = args.log_interval,
        CHECKPOINT_DIR  = args.checkpoint_dir,
    )
    train(cfg, num_rollouts=args.num_rollouts)
