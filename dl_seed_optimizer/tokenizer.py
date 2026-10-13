"""
tokenizer.py – Edge ID tokenization for the DL Seed Optimizer.

Converts raw AFL++ edge IDs (uint16, range 0..65535) to token indices
for the PyTorch Embedding layer.

Design choice: Fixed vocabulary (65,537 tokens)
  - Token 0:         Padding (padding_idx in nn.Embedding)
  - Tokens 1..65536: Edge IDs (edge_id + 1)

Rationale for fixed over dynamic vocabulary:
  - AFL++ edge IDs are bounded by MAP_SIZE (2^16 = 65536): no OOV possible.
  - No dictionary lookup overhead per token.
  - Embedding table is pre-allocated: 65537 × 32 × 4 bytes ≈ 8 MB (small).
  - Simpler: no serialization of vocabulary state needed.
"""

from __future__ import annotations

import numpy as np
import torch

# Vocabulary constants — fixed for the lifetime of the module.
PAD_TOKEN_ID: int = 0
"""Token ID reserved for padding. nn.Embedding(padding_idx=0) maps this
to a zero vector and excludes it from gradient updates."""

VOCAB_SIZE: int = 65537
"""Total vocabulary size: 0=pad, 1..65536=edge IDs.
Matches AFL++ MAP_SIZE (64 KB bitmap = 65536 possible edge indices)."""


def tokenize_fixed(edge_ids: np.ndarray) -> torch.LongTensor:
    """Convert an array of AFL++ edge IDs to token ID tensor.

    Mapping: token_id = edge_id + 1
      - edge_id=0 → token_id=1
      - edge_id=65535 → token_id=65536
      - padding is token_id=0 (never emitted by this function)

    Args:
        edge_ids: 1-D array of uint16 edge IDs from trace file.
                  Shape: (trace_len,). Dtype: uint16 or compatible int type.

    Returns:
        torch.LongTensor of shape (trace_len,) with values in [1, 65536].

    Raises:
        ValueError: If edge_ids contains values outside [0, 65535].

    Examples:
        >>> edges = np.array([0, 100, 65535], dtype=np.uint16)
        >>> tokenize_fixed(edges)
        tensor([    1,   101, 65536])
    """
    arr = np.asarray(edge_ids, dtype=np.int64)

    if len(arr) == 0:
        return torch.zeros(0, dtype=torch.long)

    if arr.min() < 0 or arr.max() > 65535:
        raise ValueError(
            f"edge_ids must be in [0, 65535]; got range "
            f"[{arr.min()}, {arr.max()}]"
        )

    return torch.tensor(arr + 1, dtype=torch.long)


def detokenize_fixed(token_ids: torch.LongTensor) -> np.ndarray:
    """Reverse tokenize_fixed: convert token IDs back to edge IDs.

    Mapping: edge_id = token_id - 1
    Padding token (0) is mapped to -1 and should be ignored by callers.

    Args:
        token_ids: 1-D LongTensor of token IDs.

    Returns:
        np.ndarray of int64 edge IDs. Padding tokens become -1.
    """
    arr = token_ids.numpy().astype(np.int64)
    return arr - 1  # pad (0) → -1; valid tokens [1,65536] → [0,65535]


def pad_traces(
    token_lists: list[torch.LongTensor],
    pad_value: int = PAD_TOKEN_ID,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Pad a list of variable-length token sequences into a single tensor.

    Args:
        token_lists: List of 1-D LongTensors, each of shape (trace_len_i,).
        pad_value:   Value to fill padding positions (default: PAD_TOKEN_ID=0).

    Returns:
        Tuple of:
          - padded:  LongTensor of shape (batch, max_len), right-padded.
          - lengths: LongTensor of shape (batch,) with each sequence's
                     actual length.

    Examples:
        >>> t1 = torch.tensor([1, 2, 3])
        >>> t2 = torch.tensor([4, 5])
        >>> padded, lengths = pad_traces([t1, t2])
        >>> padded
        tensor([[1, 2, 3], [4, 5, 0]])
        >>> lengths
        tensor([3, 2])
    """
    lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(
        token_lists, batch_first=True, padding_value=pad_value
    )
    return padded, lengths
