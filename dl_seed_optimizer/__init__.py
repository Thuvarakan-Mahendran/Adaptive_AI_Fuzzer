"""
dl_seed_optimizer – DL-based seed prioritization module for AFL++.

Architecture:
    AFL++ execution trace
      → Tokenization (edge_id → token_id)
      → Embedding(65537, 32)
      → GRU(hidden=128, layers=1)
      → latent vector [128-D]
      → DQN MLP(128 → 64 → ReLU → 1)
      → scalar Q(seed)

Q-values rank seeds in descending order: higher Q → fuzz first.
Training uses DQN with experience replay (50K buffer), Huber loss,
Adam(lr=0.001), γ=0.95, and hard target-network sync every 1000 steps.

This module is completely independent from ppo_scheduler.
It communicates with AFL++ via a separate UNIX socket: /tmp/afl_dl_bridge.sock
"""

__version__ = "0.1.0"
__all__ = [
    "config",
    "tokenizer",
    "models",
    "replay_buffer",
    "trainer",
    "reward",
    "trace_reader",
    "dl_daemon",
]
