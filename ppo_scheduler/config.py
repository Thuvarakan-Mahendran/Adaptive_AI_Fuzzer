"""
config.py – Central configuration for the PPO mutation-strategy scheduler.

All hyperparameters and state-layout constants are defined here.
No values are hard-coded in other modules.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class PPOConfig:
    """Validated PPO + state configuration.

    Design choices (not specified by the paper):
    - STATE_DIM = 192 with exact slice layout is our project-level contract.
    - ACTION_DIM is configurable (K mutation strategies, 2 ≤ K ≤ 16).
    - GAMMA / GAE_LAMBDA / CLIP_EPSILON follow standard PPO recommendations.
    - Separate actor/critic learning rates for finer control.
    - LAMBDA_CRASH = 10.0 and LAMBDA_NOVELTY = 1.0 reflect the paper's
      conceptual reward formula weighting crash discoveries heavily.
    """

    # ---- State layout (§4 – fixed contract) --------------------------------
    STATE_DIM: int = 192
    """Total state dimension.  Must always equal 192."""

    # ---- Action space -------------------------------------------------------
    ACTION_DIM: int = 8
    """Number of mutation strategies (K).  Must satisfy 2 ≤ K ≤ 16."""

    # ---- Discount and GAE ---------------------------------------------------
    GAMMA: float = 0.95
    """Discount factor γ.  Chosen following the paper (§16)."""

    GAE_LAMBDA: float = 0.95
    """GAE λ.  Same value as GAMMA per the paper's recommendation."""

    # ---- PPO clipping -------------------------------------------------------
    CLIP_EPSILON: float = 0.2
    """PPO clip parameter ε.  Standard value; kept at 0.2 per §17."""

    # ---- Learning rates -----------------------------------------------------
    ACTOR_LR: float = 3e-4
    """Adam learning rate for the actor network."""

    CRITIC_LR: float = 1e-3
    """Adam learning rate for the critic network (larger to speed up value
    convergence relative to the policy)."""

    # ---- Rollout / update schedule -----------------------------------------
    ROLLOUT_LENGTH: int = 128
    """Number of environment steps per rollout before a PPO update."""

    PPO_EPOCHS: int = 10
    """Number of gradient epochs over each completed rollout."""

    MINIBATCH_SIZE: int = 32
    """Size of each randomly shuffled minibatch within a PPO epoch."""

    # ---- Loss coefficients --------------------------------------------------
    VALUE_COEF: float = 0.5
    """Coefficient on the critic (value) loss term."""

    ENTROPY_COEF: float = 0.01
    """Coefficient on the entropy bonus; encourages exploration."""

    MAX_GRAD_NORM: float = 0.5
    """Maximum gradient norm for gradient clipping."""

    # ---- Reward weights -----------------------------------------------------
    LAMBDA_CRASH: float = 10.0
    """Weight on the crash-indicator term in the reward formula (§15)."""

    LAMBDA_NOVELTY: float = 1.0
    """Weight on the novelty-indicator term in the reward formula (§15)."""

    # ---- Coverage window ----------------------------------------------------
    COVERAGE_WINDOW: int = 100
    """Rolling window length W for differential coverage bits (§5.2)."""

    # ---- Misc ---------------------------------------------------------------
    DEVICE: Optional[str] = None
    """Target device.  None = auto-detect (CUDA if available, else CPU)."""

    LOG_INTERVAL: int = 10
    """Log training statistics every this many rollout updates."""

    SEED: int = 42
    """Global random seed for reproducibility."""

    CHECKPOINT_DIR: str = "checkpoints"
    """Directory for saving/loading checkpoints."""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError for invalid hyperparameter combinations."""
        if self.STATE_DIM != 192:
            raise ValueError(f"STATE_DIM must be 192, got {self.STATE_DIM}")

        if self.ACTION_DIM <= 1:
            raise ValueError(f"ACTION_DIM must be > 1, got {self.ACTION_DIM}")
        if self.ACTION_DIM > 16:
            raise ValueError(
                f"ACTION_DIM must be ≤ 16 (state layout reserves only 16 "
                f"operator slots), got {self.ACTION_DIM}"
            )

        if not (0 < self.GAMMA <= 1.0):
            raise ValueError(f"GAMMA must be in (0, 1], got {self.GAMMA}")
        if not (0.0 <= self.GAE_LAMBDA <= 1.0):
            raise ValueError(
                f"GAE_LAMBDA must be in [0, 1], got {self.GAE_LAMBDA}"
            )
        if self.CLIP_EPSILON <= 0:
            raise ValueError(
                f"CLIP_EPSILON must be > 0, got {self.CLIP_EPSILON}"
            )
        if self.ROLLOUT_LENGTH <= 0:
            raise ValueError(
                f"ROLLOUT_LENGTH must be > 0, got {self.ROLLOUT_LENGTH}"
            )
        if self.PPO_EPOCHS <= 0:
            raise ValueError(
                f"PPO_EPOCHS must be > 0, got {self.PPO_EPOCHS}"
            )
        if self.MINIBATCH_SIZE <= 0:
            raise ValueError(
                f"MINIBATCH_SIZE must be > 0, got {self.MINIBATCH_SIZE}"
            )

    def resolve_device(self) -> str:
        """Return the device string to use (auto-detects CUDA)."""
        import torch

        if self.DEVICE is not None:
            return self.DEVICE
        return "cuda" if torch.cuda.is_available() else "cpu"
