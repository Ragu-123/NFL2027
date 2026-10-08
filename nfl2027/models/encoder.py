"""PyTorch Trajectory Temporal Attention Encoder & Hierarchical Cross-Drill Transformer (HTT-Genome).

Combines:
1. Level 1 (Intra-Drill Temporal Phase Attention): Multi-scale 1D temporal convolutions
   (short, medium, long receptive fields) with temporal self-attention to discover decisive athletic phase transitions.
2. Level 2 (Inter-Drill Cross-Attention): Cross-drill transformer fusing Straight-Line Burst,
   Lateral Cutting Elasticity, Change of Direction, and Football Skill Drills into the unified
   Prospect Movement Genome.
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraDrillTemporalAttentionEncoder(nn.Module):
    """Level 1: Multi-Scale 1D Conv + Temporal Self-Attention over a single drill trajectory."""

    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 48,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize IntraDrillTemporalAttentionEncoder.

        Args:
            in_channels: Number of input SE(2)-equivariant kinematic channels (default: 8)
            embed_dim: Embedding dimension (default: 48)
            num_heads: Attention heads (default: 4)
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        c1 = embed_dim // 3
        c2 = embed_dim // 3
        c3 = embed_dim - (c1 + c2)

        # Multi-scale 1D convolutions: short (0.3s), medium (0.7s), long (1.5s) windows
        self.conv_short = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(in_channels, c2, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(in_channels, c3, kernel_size=15, padding=7)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.pool_proj = nn.Linear(embed_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Trajectory tensor of shape (B, C, T)
            mask: Boolean key_padding_mask of shape (B, T) where True denotes padded steps

        Returns:
            Tuple of:
            - pooled: Latent vector representation of shape (B, embed_dim)
            - alpha: Continuous temporal attention weights of shape (B, T)
        """
        if x.shape[0] == 0:
            pooled = torch.empty((0, self.embed_dim), device=x.device, dtype=x.dtype)
            alpha = torch.empty((0, x.shape[2]), device=x.device, dtype=x.dtype)
            return pooled, alpha

        c_short = F.gelu(self.conv_short(x))
        c_med = F.gelu(self.conv_med(x))
        c_long = F.gelu(self.conv_long(x))
        feat = self.dropout(self.bn(torch.cat([c_short, c_med, c_long], dim=1)))
        feat_t = feat.transpose(1, 2)  # (B, T, embed_dim)

        safe_mask = None
        all_masked = None
        if mask is not None:
            all_masked = mask.all(dim=-1)
            safe_mask = mask.clone()
            if all_masked.any():
                safe_mask[all_masked, 0] = False

        attn_out, _ = self.attn(feat_t, feat_t, feat_t, key_padding_mask=safe_mask)
        feat_t = self.ln(feat_t + attn_out)

        scores = self.pool_proj(feat_t).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        alpha = F.softmax(scores, dim=-1)

        if all_masked is not None and all_masked.any():
            alpha = alpha.masked_fill(all_masked.unsqueeze(-1), 0.0)
            feat_t = feat_t.masked_fill(all_masked.unsqueeze(-1).unsqueeze(-1), 0.0)

        pooled = torch.sum(feat_t * alpha.unsqueeze(-1), dim=1)  # (B, embed_dim)
        return pooled, alpha


class MultiDrillCrossAttentionGenomeTransformer(nn.Module):
    """Level 2: Cross-Drill Attention fusing Burst, Shuttle, 3-Cone, and Skill into Prospect Movement Genome."""

    def __init__(
        self,
        embed_dim: int = 48,
        num_modalities: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize MultiDrillCrossAttentionGenomeTransformer.

        Args:
            embed_dim: Representation dimension (default: 48)
            num_modalities: Number of drill modalities (default: 4: forty, shuttle, 3cone, skill)
            num_heads: Cross-attention heads (default: 4)
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.type_embed = nn.Embedding(num_modalities + 1, embed_dim)
        self.genome_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        drill_tokens: torch.Tensor,
        drill_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass generating Prospect Movement Genome.

        Args:
            drill_tokens: Tensor of shape (B, num_modalities, embed_dim)
            drill_mask: Boolean tensor of shape (B, num_modalities) where True denotes MISSING drill

        Returns:
            Tuple of:
            - prospect_genome: Latent genome representation (B, embed_dim)
            - attn_weights: Cross-drill attention weight matrix (B, 1 + M, 1 + M)
        """
        B = drill_tokens.shape[0]
        genome_tokens = self.genome_token.expand(B, -1, -1)
        tokens = torch.cat([genome_tokens, drill_tokens], dim=1)

        type_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.type_embed(type_ids)

        if drill_mask is not None:
            genome_mask = torch.zeros((B, 1), dtype=torch.bool, device=drill_tokens.device)
            full_mask = torch.cat([genome_mask, drill_mask], dim=1)
        else:
            full_mask = None

        attn_out, attn_weights = self.cross_attn(
            tokens, tokens, tokens, key_padding_mask=full_mask
        )
        tokens = self.ln1(tokens + attn_out)
        tokens = self.ln2(tokens + self.mlp(tokens))

        prospect_genome = tokens[:, 0, :]
        return prospect_genome, attn_weights


class TrajectoryTemporalAttentionEncoder(nn.Module):
    """Backward-compatible single-drill 7-channel trajectory encoder."""

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        half_dim = hidden_dim // 2
        self.conv_short = nn.Conv1d(in_channels, half_dim, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(in_channels, half_dim, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.pool_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] == 0:
            pooled = torch.empty((0, self.hidden_dim), device=x.device, dtype=x.dtype)
            alpha = torch.empty((0, x.shape[2]), device=x.device, dtype=x.dtype)
            return pooled, alpha

        c1 = F.gelu(self.conv_short(x))
        c2 = F.gelu(self.conv_med(x))
        feat = self.dropout(self.bn(torch.cat([c1, c2], dim=1)))
        feat_t = feat.transpose(1, 2)

        if mask is not None:
            all_masked = mask.all(dim=-1)
            safe_mask = mask.clone()
            if all_masked.any():
                safe_mask[all_masked, 0] = False
        else:
            all_masked = None
            safe_mask = None

        attn_out, _ = self.attn(feat_t, feat_t, feat_t, key_padding_mask=safe_mask)
        feat_t = self.ln(feat_t + attn_out)

        scores = self.pool_proj(feat_t).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        alpha = F.softmax(scores, dim=-1)

        if all_masked is not None and all_masked.any():
            alpha = alpha.masked_fill(all_masked.unsqueeze(-1), 0.0)
            feat_t = feat_t.masked_fill(all_masked.unsqueeze(-1).unsqueeze(-1), 0.0)

        pooled = torch.sum(feat_t * alpha.unsqueeze(-1), dim=1)
        return pooled, alpha


class TrajectoryRegressor(nn.Module):
    """End-to-end neural regressor combining trajectory encoder with MLP head."""

    def __init__(
        self,
        in_channels: int = 7,
        hidden_dim: int = 32,
        num_heads: int = 4,
        extra_dim: int = 0,
        dropout: float = 0.1,
    ):
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
        pooled, _ = self.encoder(x, mask=mask)
        if extra_feats is not None:
            pooled = torch.cat([pooled, extra_feats], dim=1)
        return self.head(pooled).squeeze(-1)


class HierarchicalMovementGenomeNetwork(nn.Module):
    """Unified Hierarchical Trajectory Transformer with Multi-Drill Cross-Attention (HTT-Genome).

    Synthesizes Level 1 (Intra-Drill Temporal Phase Attention) across multiple drills and
    Level 2 (Multi-Drill Cross-Attention) into the 48-dimensional Prospect Movement Genome.
    Includes multi-task kinematic pretraining and athletic profiling heads.
    """

    def __init__(
        self,
        in_channels: int = 8,
        embed_dim: int = 48,
        num_modalities: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.intra_encoder = IntraDrillTemporalAttentionEncoder(
            in_channels=in_channels, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.inter_transformer = MultiDrillCrossAttentionGenomeTransformer(
            embed_dim=embed_dim, num_modalities=num_modalities, num_heads=num_heads, dropout=dropout
        )
        self.kinematic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 4),
        )
        self.profile_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

    def encode_drill(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode single drill trajectory with Level 1 temporal phase attention."""
        return self.intra_encoder(x, mask=mask)

    def forward(
        self, drill_tokens: torch.Tensor, drill_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize multiple drill tokens into Prospect Movement Genome."""
        return self.inter_transformer(drill_tokens, drill_mask=drill_mask)
