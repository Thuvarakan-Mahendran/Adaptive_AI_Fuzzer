"""
models.py – Neural network modules for the DL Seed Optimizer.

Architecture (complete pipeline):

    trace tokens [batch, seq_len]          (variable-length, padded)
        ↓
    Embedding(VOCAB_SIZE=65537, EMBED_DIM=32, padding_idx=0)
        ↓
    embedded [batch, seq_len, 32]
        ↓
    pack_padded_sequence  (no truncation — full traces)
        ↓
    GRU(input=32, hidden=128, layers=1, batch_first=True)
        ↓
    h_n [1, batch, 128]  →  latent [batch, 128]
        ↓
    DQN MLP: Linear(128→64) → ReLU → Linear(64→1)
        ↓
    Q(seed) [batch]  — one scalar per seed

Parameters (locked):
    VOCAB_SIZE  = 65537   (0=pad, 1..65536=edge IDs)  Implementation choice
    EMBED_DIM   = 32      Implementation choice (spec §9)
    GRU_HIDDEN  = 128     Implementation choice (spec §10)
    GRU_LAYERS  = 1       Spec §10
    DQN_HIDDEN  = 64      Implementation choice (spec §13)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dl_seed_optimizer.config import DLConfig
from dl_seed_optimizer.tokenizer import PAD_TOKEN_ID


class SeedEncoder(nn.Module):
    """Encodes a variable-length execution trace into a 128-D latent vector.

    Pipeline:
        token_ids [batch, seq_len]
            → Embedding [batch, seq_len, embed_dim]
            → GRU (packed, no truncation)
            → final hidden state h_n [batch, gru_hidden]

    The complete ordered trace is processed without truncation.
    pack_padded_sequence handles variable-length sequences efficiently.
    """

    def __init__(
        self,
        vocab_size: int = 65537,
        embed_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        """Initialize SeedEncoder.

        Args:
            vocab_size:   Total vocabulary including pad (default: 65537).
            embed_dim:    Embedding dimension (spec §9, default: 32).
            hidden_size:  GRU hidden size (spec §10, default: 128).
            num_layers:   Number of GRU layers (spec §10, default: 1).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Learnable edge embedding. padding_idx=0 → zero vector for padding,
        # no gradient contribution from padding positions.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_TOKEN_ID,
        )

        # Single-layer unidirectional GRU.
        # batch_first=True: input/output shape is [batch, seq_len, features].
        # Unidirectional: execution order is semantically meaningful.
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(
        self,
        token_ids: torch.LongTensor,   # [batch, max_seq_len] padded
        lengths: torch.LongTensor,     # [batch] actual trace lengths
    ) -> torch.Tensor:                 # [batch, hidden_size]
        """Encode a padded batch of traces to latent vectors.

        Args:
            token_ids: Long tensor of shape [batch, max_seq_len].
                       Values in [0, VOCAB_SIZE-1]; 0 = padding.
            lengths:   Long tensor of shape [batch], giving the actual
                       (unpadded) length of each sequence.

        Returns:
            Latent tensor of shape [batch, hidden_size].
            Each row is the final GRU hidden state for one seed trace.
        """
        # Embed: [batch, max_seq_len] → [batch, max_seq_len, embed_dim]
        token_ids = token_ids.to(self.embedding.weight.device)
        embedded = self.embedding(token_ids)

        # Pack for efficient variable-length GRU processing.
        # enforce_sorted=False: handles arbitrary length ordering.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # GRU forward: h_n has shape [num_layers, batch, hidden_size]
        _, h_n = self.gru(packed)

        # Take the final layer's hidden state: [batch, hidden_size]
        return h_n[-1]  # equivalent to h_n.squeeze(0) when num_layers=1


class DQNHead(nn.Module):
    """Maps a latent seed vector to a scalar Q-value.

    Architecture: Linear(latent_dim→hidden_dim) → ReLU → Linear(hidden_dim→1)

    This is the DQN's action-value head. Since we treat each seed as a
    distinct 'state' (not action), Q(seed) is a scalar rating of how
    valuable it is to fuzz that seed next.
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64) -> None:
        """Initialize DQNHead.

        Args:
            latent_dim:  Input dimension (= GRU hidden size, default: 128).
            hidden_dim:  Hidden layer size (spec §13, default: 64).
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of latent vectors.

        Args:
            latent: Tensor of shape [batch, latent_dim].

        Returns:
            Tensor of shape [batch] — one scalar Q-value per seed.
        """
        return self.net(latent).squeeze(-1)  # [batch, 1] → [batch]


class SeedQNetwork(nn.Module):
    """End-to-end network: trace tokens → scalar Q-value.

    Composes SeedEncoder (Embedding + GRU) with DQNHead (MLP).
    Used for both online (training) and target (TD target) networks.

    The online network is updated each gradient step.
    The target network is hard-copied from online every 1,000 steps.
    """

    def __init__(self, config: DLConfig) -> None:
        """Initialize SeedQNetwork from DLConfig.

        Args:
            config: DLConfig instance with all hyperparameters.
        """
        super().__init__()

        self.encoder = SeedEncoder(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.EMBED_DIM,
            hidden_size=config.GRU_HIDDEN,
            num_layers=config.GRU_LAYERS,
        )
        self.head = DQNHead(
            latent_dim=config.GRU_HIDDEN,
            hidden_dim=config.DQN_HIDDEN,
        )

    def forward(
        self,
        token_ids: torch.LongTensor,   # [batch, max_seq_len]
        lengths: torch.LongTensor,     # [batch]
    ) -> torch.Tensor:                 # [batch]
        """Forward pass: trace tokens → Q-values.

        Args:
            token_ids: Padded token ID tensor [batch, max_seq_len].
            lengths:   Actual sequence lengths [batch].

        Returns:
            Q-value tensor of shape [batch].
        """
        latent = self.encoder(token_ids, lengths)  # [batch, GRU_HIDDEN]
        return self.head(latent)                    # [batch]

    def score_single(self, token_ids: torch.LongTensor) -> float:
        """Score a single (unbatched) trace. Inference convenience method.

        Args:
            token_ids: 1-D LongTensor of shape (seq_len,).

        Returns:
            Scalar Q-value as a Python float.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            batched = token_ids.unsqueeze(0).to(device)   # [1, seq_len]
            length = torch.tensor([token_ids.shape[0]], device=device)   # [1]
            q = self.forward(batched, length)
        return q.item()
