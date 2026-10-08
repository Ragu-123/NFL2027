"""PyTorch Dataset implementations for combine trajectory tensors."""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class CombineDrillDataset(Dataset):
    """PyTorch Dataset for multi-channel kinematic trajectory sequences.

    Provides trajectory tensors, padding masks, and optional regression labels.
    """

    def __init__(
        self,
        trajectories: Union[np.ndarray, torch.Tensor],
        seq_lens: Union[np.ndarray, torch.Tensor, List[int]],
        targets: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None,
        meta_keys: Optional[List[Tuple]] = None,
        max_len: Optional[int] = None,
        channels_first: Optional[bool] = None,
    ):
        """Initialize CombineDrillDataset.

        Args:
            trajectories: Array or tensor of shape (N, C, T) or (N, T, C)
            seq_lens: Sequence lengths array, tensor, or list of shape (N,)
            targets: Optional regression target values (N,)
            meta_keys: Optional list of drill metadata tuples
            max_len: Optional slice cap on sequence length T
            channels_first: Optional bool. If True, expects (N, C, T). If False, expects (N, T, C).
                           If None, auto-detects based on channel dimensions.
        """
        if isinstance(trajectories, np.ndarray):
            trajectories = torch.from_numpy(trajectories).float()
        else:
            trajectories = trajectories.float()

        # Handle channel layout: target internal representation is (N, C, T)
        if trajectories.ndim == 3:
            if channels_first is True:
                pass
            elif channels_first is False:
                trajectories = trajectories.permute(0, 2, 1)
            else:
                # Auto-detect: if last dim matches standard channels (7) and middle does not, permute
                if trajectories.shape[2] == 7 and trajectories.shape[1] != 7:
                    trajectories = trajectories.permute(0, 2, 1)

        if max_len is not None and max_len < trajectories.shape[2]:
            trajectories = trajectories[:, :, :max_len]

        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.long()
        elif isinstance(seq_lens, np.ndarray):
            seq_lens = torch.from_numpy(seq_lens).long()
        else:
            seq_lens = torch.tensor(seq_lens, dtype=torch.long)

        self.trajectories = trajectories
        self.seq_lens = seq_lens
        self.meta_keys = meta_keys
        self.num_samples = trajectories.shape[0]
        self.num_channels = trajectories.shape[1]
        self.seq_len = trajectories.shape[2]

        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.float()
            elif isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets).float()
            else:
                targets = torch.tensor(targets, dtype=torch.float32)
            self.targets = targets
        else:
            self.targets = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        x = self.trajectories[idx]
        L = self.seq_lens[idx]
        # Padding mask: True where padded (t >= L), False where valid (t < L)
        # Note: PyTorch MultiheadAttention key_padding_mask treats True as masked positions
        t_indices = torch.arange(self.seq_len, device=x.device)
        mask = t_indices >= L

        item = {
            "trajectory": x,
            "seq_len": L,
            "mask": mask,
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.meta_keys is not None:
            item["meta"] = self.meta_keys[idx]

        return item

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Batch collation function."""
        trajectories = torch.stack([b["trajectory"] for b in batch], dim=0)
        seq_lens = torch.stack([b["seq_len"] for b in batch], dim=0)
        masks = torch.stack([b["mask"] for b in batch], dim=0)

        out = {
            "trajectory": trajectories,
            "seq_len": seq_lens,
            "mask": masks,
        }

        if "target" in batch[0]:
            out["target"] = torch.stack([b["target"] for b in batch], dim=0)
        if "meta" in batch[0]:
            out["meta"] = [b["meta"] for b in batch]

        return out
