"""Unit tests for PyTorch Trajectory Temporal Attention Encoder."""

import numpy as np
import pytest
import torch

from nfl2027.models.encoder import (
    TrajectoryTemporalAttentionEncoder,
    TrajectoryRegressor,
    IntraDrillTemporalAttentionEncoder,
    MultiDrillCrossAttentionGenomeTransformer,
)
from nfl2027.data.dataset import CombineDrillDataset


def test_encoder_output_shapes():
    """Verify encoder outputs correct embedding and attention tensor shapes."""
    batch_size = 8
    channels = 7
    timesteps = 100
    hidden_dim = 32

    encoder = TrajectoryTemporalAttentionEncoder(
        in_channels=channels, hidden_dim=hidden_dim, num_heads=4
    )
    x = torch.randn(batch_size, channels, timesteps)

    pooled, alpha = encoder(x)

    assert pooled.shape == (batch_size, hidden_dim)
    assert alpha.shape == (batch_size, timesteps)


def test_attention_weights_sum_to_one():
    """Verify attention weights form a valid probability distribution summing to 1.0."""
    encoder = TrajectoryTemporalAttentionEncoder(in_channels=7, hidden_dim=32, num_heads=4)
    encoder.eval()
    x = torch.randn(4, 7, 60)

    with torch.no_grad():
        _, alpha = encoder(x)

    sums = alpha.sum(dim=-1).numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_encoder_padding_masking():
    """Verify that padded timesteps receive zero attention weight when masked."""
    encoder = TrajectoryTemporalAttentionEncoder(in_channels=7, hidden_dim=32, num_heads=4)
    encoder.eval()
    x = torch.randn(2, 7, 50)
    # Sequence 0 has length 30, sequence 1 has length 40
    seq_lens = torch.tensor([30, 40])
    mask = torch.arange(50).unsqueeze(0) >= seq_lens.unsqueeze(1)

    with torch.no_grad():
        _, alpha = encoder(x, mask=mask)

    # For sequence 0, timesteps >= 30 must have near-zero attention
    np.testing.assert_allclose(alpha[0, 30:].numpy(), 0.0, atol=1e-5)
    # For sequence 1, timesteps >= 40 must have near-zero attention
    np.testing.assert_allclose(alpha[1, 40:].numpy(), 0.0, atol=1e-5)
    # Valid portions must sum to 1.0
    np.testing.assert_allclose(alpha.sum(dim=-1).numpy(), 1.0, atol=1e-5)


def test_encoder_backpropagation():
    """Verify that gradients propagate to all convolutional and attention layers."""
    encoder = TrajectoryTemporalAttentionEncoder(in_channels=7, hidden_dim=32, num_heads=4)
    x = torch.randn(4, 7, 50, requires_grad=True)

    pooled, alpha = encoder(x)
    loss = pooled.sum()
    loss.backward()

    # Check gradients on conv layers
    assert encoder.conv_short.weight.grad is not None
    assert torch.all(torch.isfinite(encoder.conv_short.weight.grad))
    assert encoder.conv_med.weight.grad is not None
    assert torch.all(torch.isfinite(encoder.conv_med.weight.grad))
    # Check gradient on input
    assert x.grad is not None


def test_trajectory_regressor():
    """Verify end-to-end trajectory regressor predicts scalar targets."""
    model = TrajectoryRegressor(in_channels=7, hidden_dim=32, num_heads=4)
    x = torch.randn(6, 7, 80)
    y_pred = model(x)

    assert y_pred.shape == (6,)
    loss = ((y_pred - torch.ones(6)) ** 2).mean()
    loss.backward()
    assert model.head[0].weight.grad is not None


def test_combine_drill_dataset():
    """Verify CombineDrillDataset indexing and collate function."""
    N, C, T = 12, 7, 50
    trajectories = np.random.randn(N, C, T).astype(np.float32)
    seq_lens = np.random.randint(20, 50, size=N)
    targets = np.random.randn(N).astype(np.float32)

    dataset = CombineDrillDataset(trajectories, seq_lens, targets=targets)
    assert len(dataset) == N

    sample = dataset[0]
    assert "trajectory" in sample
    assert "mask" in sample
    assert "target" in sample
    assert sample["trajectory"].shape == (C, T)
    assert sample["mask"].shape == (T,)

    # Test batch collate
    batch = [dataset[i] for i in range(4)]
    collated = CombineDrillDataset.collate_fn(batch)
    assert collated["trajectory"].shape == (4, C, T)
    assert collated["mask"].shape == (4, T)
    assert collated["target"].shape == (4,)


def test_intra_drill_encoder():
    """Verify Level 1 IntraDrillTemporalAttentionEncoder forward pass and attention."""
    encoder = IntraDrillTemporalAttentionEncoder(in_channels=8, embed_dim=48, num_heads=4)
    encoder.eval()
    x = torch.randn(4, 8, 80)
    seq_lens = torch.tensor([50, 60, 70, 80])
    mask = torch.arange(80).unsqueeze(0) >= seq_lens.unsqueeze(1)

    with torch.no_grad():
        pooled, alpha = encoder(x, mask=mask)

    assert pooled.shape == (4, 48)
    assert alpha.shape == (4, 80)
    # Masked elements should have near zero attention
    np.testing.assert_allclose(alpha[0, 50:].numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(alpha.sum(dim=-1).numpy(), 1.0, atol=1e-5)


def test_multi_drill_cross_attention_transformer():
    """Verify Level 2 MultiDrillCrossAttentionGenomeTransformer fuses modalities into genome."""
    transformer = MultiDrillCrossAttentionGenomeTransformer(embed_dim=48, num_modalities=4, num_heads=4)
    transformer.eval()

    B = 6
    drill_tokens = torch.randn(B, 4, 48)
    # Simulate missing drills: prospect 0 missing modalities 1 and 2
    drill_mask = torch.zeros(B, 4, dtype=torch.bool)
    drill_mask[0, 1] = True
    drill_mask[0, 2] = True

    with torch.no_grad():
        genome, attn_weights = transformer(drill_tokens, drill_mask=drill_mask)

    assert genome.shape == (B, 48)
    assert attn_weights.shape == (B, 5, 5)  # 1 genome token + 4 modalities
    # Padded modalities should receive zero cross-attention weight from genome token
    assert attn_weights[0, 0, 2].item() < 1e-4  # modality 1 (index 2 in tokens)


def test_multi_drill_extreme_masking():
    """Verify stability when 3 of 4 modalities or all modalities are masked."""
    transformer = MultiDrillCrossAttentionGenomeTransformer(embed_dim=48, num_modalities=4, num_heads=4)
    transformer.eval()

    B = 4
    drill_tokens = torch.randn(B, 4, 48)
    drill_mask = torch.zeros(B, 4, dtype=torch.bool)
    # Prospect 0: only 1 drill available (modalities 1, 2, 3 masked)
    drill_mask[0, 1:] = True
    # Prospect 1: ALL drills missing (modalities 0, 1, 2, 3 masked)
    drill_mask[1, :] = True
    # Prospect 2: 2 drills missing
    drill_mask[2, :2] = True

    with torch.no_grad():
        genome, attn_weights = transformer(drill_tokens, drill_mask=drill_mask)

    assert genome.shape == (B, 48)
    assert not torch.isnan(genome).any(), "Genome tensor contains NaNs under extreme masking"
    assert not torch.isinf(genome).any(), "Genome tensor contains Infs under extreme masking"
    # When all drills are masked, genome token should attend to itself with weight 1.0
    assert torch.isclose(attn_weights[1, 0, 0], torch.tensor(1.0), atol=1e-4)


def test_hierarchical_movement_genome_network():
    """Verify end-to-end HierarchicalMovementGenomeNetwork encoding and gradient flow."""
    from nfl2027.models.encoder import HierarchicalMovementGenomeNetwork

    net = HierarchicalMovementGenomeNetwork(in_channels=8, embed_dim=48, num_modalities=4)
    B, C, T = 4, 8, 50
    x = torch.randn(B, C, T, requires_grad=True)

    # 1. Level 1 single drill encode
    pooled, alpha = net.encode_drill(x)
    assert pooled.shape == (B, 48)
    assert alpha.shape == (B, T)

    # 2. Level 1 pretraining loss backpropagation
    kin_preds = net.kinematic_head(pooled)
    loss1 = kin_preds.sum()
    loss1.backward()
    assert x.grad is not None

    # 3. Level 2 multi-drill cross-attention
    drill_tokens = torch.randn(B, 4, 48, requires_grad=True)
    mask = torch.zeros(B, 4, dtype=torch.bool)
    mask[0, 2:] = True
    genome, cross_attn = net(drill_tokens, drill_mask=mask)

    assert genome.shape == (B, 48)
    assert cross_attn.shape == (B, 5, 5)

    profile_preds = net.profile_head(genome)
    loss2 = profile_preds.sum()
    loss2.backward()
    assert drill_tokens.grad is not None

