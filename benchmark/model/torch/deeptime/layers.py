import math
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import nn, Tensor


class GaussianFourierFeatureTransform(nn.Module):
    """
    https://github.com/ndahlquist/pytorch-fourier-feature-networks
    Given an input of size [..., time, dim], returns a tensor of size [..., n_fourier_feats, time].
    """

    def __init__(self, input_dim: int, n_fourier_feats: int, scales: List[float]):
        super().__init__()
        self.input_dim = input_dim
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

        n_scale_feats = n_fourier_feats // (2 * len(scales))
        assert (
            n_scale_feats * 2 * len(scales) == n_fourier_feats
        ), f"n_fourier_feats: {n_fourier_feats} must be divisible by 2 * len(scales) = {2 * len(scales)}"
        B_size = (input_dim, n_scale_feats)
        B = torch.cat([torch.randn(B_size) * scale for scale in scales], dim=1)
        self.register_buffer("B", B)

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() >= 2
        ), f"Expected 2 or more dimensional input (got {x.dim()}D input)"
        time, dim = x.shape[-2], x.shape[-1]

        assert (
            dim == self.input_dim
        ), f"Expected input to have {self.input_dim} channels (got {dim} channels)"

        x = torch.einsum("... t n, n d -> ... t d", [x, self.B])
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class INRLayer(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, dropout: Optional[float] = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self._layer(x)
        return self.norm(out)

    def _layer(self, x: Tensor) -> Tensor:
        return self.dropout(torch.relu(self.linear(x)))


class INR(nn.Module):
    def __init__(
        self,
        in_feats: int,
        layers: int,
        layer_size: int,
        n_fourier_feats: int,
        scales: List[float],
        dropout: Optional[float] = 0.1,
    ):
        super().__init__()
        self.features = (
            nn.Linear(in_feats, layer_size)
            if n_fourier_feats == 0
            else GaussianFourierFeatureTransform(in_feats, n_fourier_feats, scales)
        )
        in_size = layer_size if n_fourier_feats == 0 else n_fourier_feats
        layers = [INRLayer(in_size, layer_size, dropout=dropout)] + [
            INRLayer(layer_size, layer_size, dropout=dropout) for _ in range(layers - 1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] = 0.0):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(
        self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None
    ) -> Tuple[Tensor, Tensor]:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(
        self, X: Tensor, Y: Tensor, reg_coeff: float
    ) -> Tuple[Tensor, Tensor]:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)


class DeepTime(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        datetime_feats: int,
        layer_size: int,
        inr_layers: int,
        n_fourier_feats: int,
        scales: List[float],
    ):
        super().__init__()
        self.inr = INR(
            in_feats=datetime_feats + 1,
            layers=inr_layers,
            layer_size=layer_size,
            n_fourier_feats=n_fourier_feats,
            scales=scales,
        )
        self.adaptive_weights = RidgeRegressor()

        self.prediction_length = prediction_length
        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

    def forward(
        self, x: Tensor, x_time: Optional[Tensor], y_time: Optional[Tensor]
    ) -> Tensor:
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, self.prediction_length).to(x.device)

        if y_time is not None:
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, "1 t 1 -> b t 1", b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)
            time_reprs = self.inr(coords)
        else:
            time_reprs = repeat(self.inr(coords), "1 t d -> b t d", b=batch_size)

        lookback_reprs = time_reprs[:, : -self.prediction_length]
        horizon_reprs = time_reprs[:, -self.prediction_length :]
        w, b = self.adaptive_weights(lookback_reprs, x)
        preds = self.forecast(horizon_reprs, w, b)
        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum("... d o, ... t d -> ... t o", [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, "t -> 1 t 1")
