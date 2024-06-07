import torch
import numpy as np
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("sinusoids", self.get_sinusoids(seq_len=max_len))

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        features: torch.Tensor [batch, time_steps, num_series, embedding dimension]
            An embedding which will be modified by the position encoding.
        time_steps: torch.IntTensor [batch, time_steps, 1]
            The time step for each entry in the input.
        Returns:
        --------
        output_encoded: torch.Tensor [batch, time steps, embedding dimension]
            The modified embedding.
        """
        pos_idx = torch.arange(features.size(1), device=features.device).view(1, -1)
        return features + self.sinusoids[pos_idx.to(torch.long)]

    def get_sinusoids(
        self, seq_len: int, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        pos = (
            torch.arange(0, seq_len, device=device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.embedding_dim)
        )
        dim = (
            torch.arange(0, self.embedding_dim, device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-np.log(10000) * (2 * (dim // 2) / self.embedding_dim))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        return pos
