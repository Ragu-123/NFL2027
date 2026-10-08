"""PyTorch Trajectory Temporal Attention Encoder (NK-TrajNet).

Combines multi-scale 1D temporal convolutions (short and medium receptive fields)
with multi-head temporal self-attention to discover decisive athletic phase transitions.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryTemporalAttentionEncoder(nn.Module):
    """Multi-Scale 1D Temporal Convolutional Neural Network with 4-Head Self-Attention.

    Encodes continuous 7-channel trajectory dynamics:
    [speed, accel, jerk, dir, normal_accel, curvature, power]
    into fixed-dimension latent representations and attention weights alpha(t).
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize TrajectoryTemporalAttentionEncoder.

        Args:
            in_channels: Number of input kinematic channels (default: 7)
            hidden_dim: Latent representation dimension (default: 32)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        half_dim = hidden_dim // 2
        # Short receptive field (kernel 3 -> 0.3s window at 10 Hz)
        self.conv_short = nn.Conv1d(
            in_channels, half_dim, kernel_size=3, padding=1
        )
        # Medium receptive field (kernel 7 -> 0.7s window at 10 Hz)
        self.conv_med = nn.Conv1d(
            in_channels, half_dim, kernel_size=7, padding=3
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-head temporal self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_dim)

        # Temporal attention pooling projection
        self.pool_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Trajectory tensor of shape (B, C, T) where C=in_channels, T=timesteps
            mask: Optional boolean key_padding_mask of shape (B, T) where True denotes padded elements

        Returns:
            Tuple of:
            - pooled: Latent vector representation of shape (B, hidden_dim)
            - alpha: Continuous temporal attention weights of shape (B, T)
        """
        # Multi-scale 1D convolutions
        c1 = F.gelu(self.conv_short(x))
        c2 = F.gelu(self.conv_med(x))
        feat = self.dropout(self.bn(torch.cat([c1, c2], dim=1)))  # (B, hidden_dim, T)

        # Transpose to (B, T, hidden_dim) for attention
        feat_t = feat.transpose(1, 2)

        # Self-attention with residual connection & layer norm
        attn_out, _ = self.attn(feat_t, feat_t, feat_t, key_padding_mask=mask)
        feat_t = self.ln(feat_t + attn_out)

        # Attention pooling
        scores = self.pool_proj(feat_t).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        alpha = F.softmax(scores, dim=-1)  # (B, T)

        # Weighted sum across timesteps
        pooled = torch.sum(feat_t * alpha.unsqueeze(-1), dim=1)  # (B, hidden_dim)

        return pooled, alpha

    @torch.no_grad()
    def encode_sequences(
        self,
        traj_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        max_len: int = 100,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract latent embeddings and attention weights across a batch.

        Args:
            traj_tensor: Tensor of shape (N, C, T)
            seq_lens: Tensor of shape (N,)
            max_len: Cap on sequence length to evaluate
            batch_size: Batch size for inference
            device: Torch device

        Returns:
            Tuple of (latent_embeddings [N, hidden_dim], attention_weights [N, max_len])
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        N = traj_tensor.shape[0]
        T = min(max_len, traj_tensor.shape[2])

        all_pooled = []
        all_alpha = []

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            bx = traj_tensor[start_idx:end_idx, :, :T].to(device)
            blens = seq_lens[start_idx:end_idx].to(device)

            bmask = torch.arange(T, device=device).unsqueeze(0) >= blens.unsqueeze(1)
            pooled, alpha = self(bx, mask=bmask)

            all_pooled.append(pooled.cpu().numpy())
            all_alpha.append(alpha.cpu().numpy())

        embeddings = np.concatenate(all_pooled, axis=0) if all_pooled else np.empty((0, self.hidden_dim))
        attentions = np.concatenate(all_alpha, axis=0) if all_alpha else np.empty((0, T))
        return embeddings, attentions


class TrajectoryRegressor(nn.Module):
    """End-to-end neural regressor combining TrajectoryTemporalAttentionEncoder with MLP."""

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 32,
        num_heads: int = 4,
        extra_dim: int = 0,
        dropout: float = 0.1,
    ):
        """Initialize TrajectoryRegressor."""
        super().__init__()
        self.encoder = TrajectoryTemporalAttentionEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        total_dim = hidden_dim + extra_dim
        self.head = nn.Sequential(
            nn.Linear(total_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        extra_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting continuous target value."""
        pooled, _ = self.encoder(x, mask=mask)
        if extra_feats is not None:
            pooled = torch.cat([pooled, extra_feats], dim=1)
        return self.head(pooled).squeeze(-1)
