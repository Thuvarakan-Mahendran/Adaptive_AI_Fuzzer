"""
replay_buffer.py – Experience replay buffer for DQN training.

Implements a fixed-capacity circular replay buffer storing Transition tuples.
Transitions are sampled uniformly at random during training.

Transition format:
    seed_id:           int           — AFL++ queue entry ID
    trace_tokens:      LongTensor    — [seq_len] tokenized edge IDs
    trace_len:         int           — actual trace length (before padding)
    reward:            float         — R(seed) after fuzzing
    next_trace_tokens: LongTensor    — [seq_len'] trace for the next seed
    next_trace_len:    int           — next trace actual length
    done:              bool          — True at episode boundary

Buffer parameters (from DLConfig):
    BUFFER_CAPACITY = 50,000  (implementation choice, spec §20)
    BATCH_SIZE      = 32      (implementation choice, spec §21)
"""

from __future__ import annotations

import dataclasses
import random
import threading
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from dl_seed_optimizer.config import DLConfig


@dataclasses.dataclass
class Transition:
    """One DQN experience tuple (s, r, s').

    For seed prioritization:
      - 'state'  is the current seed's execution trace.
      - 'reward' is the fuzzing outcome reward R(seed).
      - 'next'   is the trace of the next seed selected after this one.

    Note: there is no explicit 'action' field because in seed selection
    the 'action' IS the seed chosen (the current seed_id).
    """

    seed_id: int
    """AFL++ queue entry index of the current seed."""

    trace_tokens: torch.LongTensor
    """Tokenized edge IDs for current seed. Shape: (trace_len,)."""

    trace_len: int
    """Actual length of trace_tokens (before padding)."""

    reward: float
    """Scalar reward R observed after fuzzing this seed."""

    next_trace_tokens: torch.LongTensor
    """Tokenized edge IDs for the next selected seed. Shape: (next_trace_len,)."""

    next_trace_len: int
    """Actual length of next_trace_tokens (before padding)."""

    done: bool
    """True if this transition marks an episode boundary (e.g., campaign reset)."""


def collate_transitions(batch: List[Transition]) -> Dict[str, torch.Tensor]:
    """Collate a list of Transitions into batched padded tensors.

    Variable-length traces are right-padded with PAD_TOKEN_ID=0 to the
    maximum length within the batch. Sequences are NOT globally truncated.

    Args:
        batch: List of Transition objects (length = batch_size).

    Returns:
        Dict with keys:
          'tokens'       — LongTensor [B, max_len]   padded current traces
          'lengths'      — LongTensor [B]             current trace lengths
          'rewards'      — FloatTensor [B]            scalar rewards
          'dones'        — FloatTensor [B]            done flags (0.0 or 1.0)
          'next_tokens'  — LongTensor [B, max_len']   padded next traces
          'next_lengths' — LongTensor [B]             next trace lengths
    """
    tokens = [t.trace_tokens for t in batch]
    lengths = torch.tensor([t.trace_len for t in batch], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
    dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32)

    next_tokens_list = [t.next_trace_tokens for t in batch]
    next_lengths = torch.tensor([t.next_trace_len for t in batch], dtype=torch.long)

    # Pad variable-length sequences to max within this batch
    padded = nn.utils.rnn.pad_sequence(
        tokens, batch_first=True, padding_value=0
    )
    next_padded = nn.utils.rnn.pad_sequence(
        next_tokens_list, batch_first=True, padding_value=0
    )

    return {
        "tokens": padded,
        "lengths": lengths,
        "rewards": rewards,
        "dones": dones,
        "next_tokens": next_padded,
        "next_lengths": next_lengths,
    }


class ReplayBuffer:
    """Fixed-capacity circular replay buffer with thread-safe push/sample.

    Uses a list-based circular buffer with position pointer.
    When capacity is reached, oldest transitions are overwritten (FIFO).

    Thread safety:
        push() and sample() are protected by a threading.Lock.
        This allows the main thread (IPC handler) to push transitions
        while the background thread (DQN trainer) samples from the buffer.

    Args:
        capacity:   Maximum number of transitions to store (default: 50,000).
    """

    def __init__(self, capacity: int = 50_000) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.capacity = capacity
        self._buffer: List[Transition] = []
        self._position: int = 0
        self._lock = threading.Lock()

        # Counter for triggering training: increments on each push
        self._total_pushed: int = 0

    def push(self, transition: Transition) -> None:
        """Add one transition to the buffer (thread-safe).

        If the buffer is at capacity, the oldest transition is overwritten.

        Args:
            transition: Transition namedtuple to store.
        """
        with self._lock:
            if len(self._buffer) < self.capacity:
                self._buffer.append(transition)
            else:
                self._buffer[self._position] = transition
            self._position = (self._position + 1) % self.capacity
            self._total_pushed += 1

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions (thread-safe).

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Transition objects (length = min(batch_size, len(buffer))).

        Raises:
            ValueError: If buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                raise ValueError("Cannot sample from an empty ReplayBuffer.")
            n = min(batch_size, len(self._buffer))
            return random.sample(self._buffer, n)

    def __len__(self) -> int:
        """Current number of stored transitions (thread-safe)."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_pushed(self) -> int:
        """Total transitions pushed since creation (monotonically increasing)."""
        with self._lock:
            return self._total_pushed

    def is_ready(self, min_size: int) -> bool:
        """Return True if buffer has at least min_size transitions.

        Used by the trainer to decide whether to begin training.

        Args:
            min_size: Minimum buffer occupancy required.
        """
        with self._lock:
            return len(self._buffer) >= min_size

    def clear(self) -> None:
        """Remove all transitions (for testing / campaign reset)."""
        with self._lock:
            self._buffer.clear()
            self._position = 0
            self._total_pushed = 0
