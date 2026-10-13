"""
config.py – Central configuration dataclass for the DL Seed Optimizer.

All hyperparameters, architecture constants, and path defaults are defined
here. No values are hard-coded in other modules.

Parameter sources:
  - Paper-specified: α=0.001, γ=0.95 (from the adaptive fuzzing paper)
  - Implementation choices: all other values (noted inline)
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class DLConfig:
    """Validated configuration for the DL Seed Optimizer.

    Separation from PPOConfig (ppo_scheduler/config.py) is intentional:
    this module learns seed prioritization (Q-values over seeds), while
    PPO learns mutation strategy selection (policy over operators).
    They share NO parameters and NO state representations.
    """

    # -----------------------------------------------------------------------
    # Tokenizer / Vocabulary
    # -----------------------------------------------------------------------
    VOCAB_SIZE: int = 65537
    """Total vocabulary size: 0=padding, 1..65536=AFL++ edge IDs.
    Fixed at 65537 to match AFL++ MAP_SIZE (2^16 edges). No OOV handling
    needed since edge IDs are bounded by AFL++ instrumentation.
    Implementation choice."""

    # -----------------------------------------------------------------------
    # Neural network architecture (Embedding → GRU → DQN MLP)
    # -----------------------------------------------------------------------
    EMBED_DIM: int = 32
    """Dimension of the learned edge embedding vectors.
    Each token (edge ID) maps to a 32-D dense vector.
    Implementation choice (spec §9). Embedding table: 65537×32×4 ≈ 8 MB."""

    GRU_HIDDEN: int = 128
    """GRU hidden state size. The final hidden state h_n[batch, 128]
    is the latent seed representation fed into the DQN head.
    Implementation choice (spec §10)."""

    GRU_LAYERS: int = 1
    """Number of stacked GRU layers. Single layer per spec §10."""

    GRU_BIDIRECTIONAL: bool = False
    """Unidirectional GRU preserves execution order.
    Bidirectional is explicitly excluded (spec §10): execution
    ordering is semantically meaningful."""

    DQN_HIDDEN: int = 64
    """Hidden layer size of the DQN MLP head.
    Architecture: Linear(128→64) → ReLU → Linear(64→1).
    Implementation choice (spec §13)."""

    # -----------------------------------------------------------------------
    # DQN training hyperparameters
    # -----------------------------------------------------------------------
    LEARNING_RATE: float = 0.001
    """Adam optimizer learning rate α.
    Paper-specified value (spec §27). Do not change without paper citation."""

    GAMMA: float = 0.95
    """Discount factor γ for TD targets: Q_target = r + γ·Q_next·(1−done).
    Paper-specified value (spec §23). Do not change without paper citation."""

    BUFFER_CAPACITY: int = 50_000
    """Maximum number of transitions stored in the replay buffer.
    Circular (FIFO eviction). Implementation choice (spec §20)."""

    BATCH_SIZE: int = 32
    """Number of transitions sampled per gradient update.
    Implementation choice (spec §21)."""

    TARGET_SYNC: int = 1_000
    """Number of gradient updates between hard target-network copies.
    Hard copy (not Polyak averaging). Implementation choice (spec §28)."""

    MIN_BUFFER_SIZE: int = 1_000
    """Minimum replay buffer occupancy before training begins.
    Ensures sufficient diversity before first gradient update.
    Implementation choice."""

    TRAIN_INTERVAL: int = 4
    """Run one gradient step every N new transitions pushed to buffer.
    Implementation choice."""

    MAX_GRAD_NORM: float = 1.0
    """Maximum gradient norm for gradient clipping.
    Guards against exploding gradients on long traces.
    Implementation choice."""

    # -----------------------------------------------------------------------
    # Reward formula weights
    # -----------------------------------------------------------------------
    LAMBDA_CRASH: float = 10.0
    """Reward weight for crash discoveries.
    Matches LAMBDA_CRASH in adaptive-fuzzer/c_mutator/aggregator.c (line 22)
    and ppo_scheduler/config.py for cross-module consistency."""

    LAMBDA_NOVELTY: float = 1.0
    """Reward weight for novel coverage discoveries.
    Matches LAMBDA_NOVELTY in adaptive-fuzzer/c_mutator/aggregator.c (line 23)."""

    # -----------------------------------------------------------------------
    # Seed ranking / acceptance
    # -----------------------------------------------------------------------
    ACCEPT_PERCENTILE: float = 50.0
    """Seeds with Q-value above this percentile of all cached Q-values
    are accepted (accept=1); others are skipped (accept=0).
    Implementation choice. 50.0 = top half."""

    EPSILON_START: float = 0.1
    """Initial ε for ε-greedy exploration: with probability ε, a random
    seed is selected instead of the highest-Q seed. Implementation choice."""

    EPSILON_MIN: float = 0.01
    """Minimum ε after decay. Implementation choice."""

    EPSILON_DECAY: float = 0.9995
    """Multiplicative ε decay applied per episode. Implementation choice."""

    # -----------------------------------------------------------------------
    # IPC / Communication
    # -----------------------------------------------------------------------
    DL_SOCKET_PATH: str = "/tmp/afl_dl_bridge.sock"
    """UNIX domain socket path for the DL daemon.
    Separate from PPO daemon's /tmp/afl_ai_bridge.sock to ensure
    independent lifecycle and latency isolation."""

    TRACE_DIR: str = "/dev/shm/afl_dl_traces"
    """Directory for trace files written by the C mutator and read by
    the DL daemon. Uses tmpfs (/dev/shm) for memory-speed I/O."""

    # -----------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------
    DEVICE: Optional[str] = None
    """Target device. None = auto-detect (CUDA if available, else CPU)."""

    CHECKPOINT_DIR: str = "dl_checkpoints"
    """Directory for saving/loading model checkpoints."""

    LOG_INTERVAL: int = 100
    """Log training statistics every N gradient steps."""

    SEED: int = 42
    """Global random seed for reproducibility."""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError for invalid hyperparameter combinations."""
        if self.VOCAB_SIZE < 2:
            raise ValueError(f"VOCAB_SIZE must be >= 2, got {self.VOCAB_SIZE}")

        if self.EMBED_DIM < 1:
            raise ValueError(f"EMBED_DIM must be >= 1, got {self.EMBED_DIM}")

        if self.GRU_HIDDEN < 1:
            raise ValueError(f"GRU_HIDDEN must be >= 1, got {self.GRU_HIDDEN}")

        if self.GRU_LAYERS < 1:
            raise ValueError(f"GRU_LAYERS must be >= 1, got {self.GRU_LAYERS}")

        if self.DQN_HIDDEN < 1:
            raise ValueError(f"DQN_HIDDEN must be >= 1, got {self.DQN_HIDDEN}")

        if not (0 < self.LEARNING_RATE <= 1.0):
            raise ValueError(
                f"LEARNING_RATE must be in (0, 1], got {self.LEARNING_RATE}"
            )

        if not (0 < self.GAMMA <= 1.0):
            raise ValueError(f"GAMMA must be in (0, 1], got {self.GAMMA}")

        if self.BUFFER_CAPACITY < self.BATCH_SIZE:
            raise ValueError(
                f"BUFFER_CAPACITY ({self.BUFFER_CAPACITY}) must be >= "
                f"BATCH_SIZE ({self.BATCH_SIZE})"
            )

        if self.MIN_BUFFER_SIZE < self.BATCH_SIZE:
            raise ValueError(
                f"MIN_BUFFER_SIZE ({self.MIN_BUFFER_SIZE}) must be >= "
                f"BATCH_SIZE ({self.BATCH_SIZE})"
            )

        if self.TARGET_SYNC < 1:
            raise ValueError(
                f"TARGET_SYNC must be >= 1, got {self.TARGET_SYNC}"
            )

        if not (0.0 < self.ACCEPT_PERCENTILE <= 100.0):
            raise ValueError(
                f"ACCEPT_PERCENTILE must be in (0, 100], got {self.ACCEPT_PERCENTILE}"
            )

        if not (0.0 <= self.EPSILON_START <= 1.0):
            raise ValueError(
                f"EPSILON_START must be in [0, 1], got {self.EPSILON_START}"
            )

        if not (0.0 <= self.EPSILON_MIN <= self.EPSILON_START):
            raise ValueError(
                f"EPSILON_MIN must be in [0, EPSILON_START], got {self.EPSILON_MIN}"
            )

    def resolve_device(self) -> str:
        """Return the device string to use (auto-detects CUDA)."""
        import torch

        if self.DEVICE is not None:
            return self.DEVICE
        return "cuda" if torch.cuda.is_available() else "cpu"
