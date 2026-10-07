"""
evaluate.py – Load checkpoint and run deterministic evaluation (§24).

Usage
-----
    python -m ppo_scheduler.evaluate --checkpoint checkpoints/step_0050.pt
    python -m ppo_scheduler.evaluate --checkpoint checkpoints/step_0050.pt \
        --num_episodes 5 --action_dim 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List

import numpy as np

from ppo_scheduler.agent import PPOAgent
from ppo_scheduler.config import PPOConfig
from ppo_scheduler.environment import MockMutationEnvironment
from ppo_scheduler.state_preprocessor import StatePreprocessor
from ppo_scheduler.train import seed_everything


def evaluate(
    checkpoint_path: str,
    config:          PPOConfig,
    num_episodes:    int = 10,
) -> Dict:
    """Evaluate a trained PPO agent deterministically.

    Steps:
    1. Create StatePreprocessor in eval mode (no stats updates).
    2. Load agent from checkpoint (restores normalization statistics).
    3. Run num_episodes episodes with deterministic action selection.
    4. Report total/mean reward, coverage, crashes, novelties,
       and action-selection frequencies.

    Args:
        checkpoint_path: Path to the saved checkpoint.
        config:          Must match the checkpoint's action_dim.
        num_episodes:    Number of episodes to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    seed_everything(config.SEED)

    preprocessor = StatePreprocessor(config)
    preprocessor.eval_mode()   # freeze normalization stats

    agent = PPOAgent(config, preprocessor)
    agent.load(checkpoint_path)
    print(f"[evaluate] Loaded checkpoint: {checkpoint_path}")

    env = MockMutationEnvironment(config, rng_seed=config.SEED + 9999)

    total_rewards:  List[float] = []
    total_coverage: List[float] = []
    total_crashes:  List[int]   = []
    total_novelties: List[int]  = []
    action_counter: Counter     = Counter()

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_coverage = 0.0
        ep_crashes = 0
        ep_novelties = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)

            ep_reward   += reward
            ep_coverage  = float(info["coverage_bits"])
            ep_crashes  += info["i_crash"]
            ep_novelties += info["i_novelty"]
            action_counter[action] += 1
            state = next_state

        total_rewards.append(ep_reward)
        total_coverage.append(ep_coverage)
        total_crashes.append(ep_crashes)
        total_novelties.append(ep_novelties)
        print(
            f"  ep={ep+1:02d}  reward={ep_reward:.2f}  "
            f"coverage={ep_coverage:.0f}  crashes={ep_crashes}  "
            f"novelties={ep_novelties}"
        )

    results = {
        "total_reward":     float(np.sum(total_rewards)),
        "mean_reward":      float(np.mean(total_rewards)),
        "mean_coverage":    float(np.mean(total_coverage)),
        "total_crashes":    int(np.sum(total_crashes)),
        "total_novelties":  int(np.sum(total_novelties)),
        "action_frequencies": dict(action_counter),
    }

    print("\n[evaluate] Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO agent evaluation")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--num_episodes",  type=int, default=10)
    p.add_argument("--action_dim",    type=int, default=8)
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PPOConfig(ACTION_DIM=args.action_dim, SEED=args.seed)
    evaluate(args.checkpoint, cfg, num_episodes=args.num_episodes)
