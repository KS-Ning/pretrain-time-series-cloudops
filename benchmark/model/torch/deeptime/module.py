from functools import cached_property
from typing import Optional, List, Dict, Tuple, Callable

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.util import weighted_average

from .layers import DeepTime
from util.torch.scaler import NOPScaler, StdScaler


class DeepTimeModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        target_dim: int,
        time_dim: int,
        scaling: bool = True,
        d_model: int = 256,
        num_layers: int = 5,
        num_fourier_feats: int = 4096,
        scales: Optional[List[float]] = None,
    ):
        super().__init__()
        scales = scales or [0.01, 0.1, 1, 5, 10, 20, 50, 100]

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.time_dim = time_dim
        self.scaling = scaling
        self.target_shape = () if target_dim == 1 else (target_dim,)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_fourier_feats = num_fourier_feats
        self.scales = scales

        # Scaling
        self.scaler = (
            StdScaler(dim=1, keepdim=True)
            if scaling
            else NOPScaler(dim=1, keepdim=True)
        )

        self.deep_time = DeepTime(
            prediction_length=prediction_length,
            datetime_feats=time_dim,
            layer_size=d_model,
            inr_layers=num_layers,
            n_fourier_feats=num_fourier_feats,
            scales=scales,
        )

    @cached_property
    def past_length(self) -> int:
        return self.context_length

    def input_info(
        self, batch_size: int = 1
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        info = {
            "past_target": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
            "past_observed_values": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
            "past_time_feat": (
                (batch_size, self.past_length, self.time_dim),
                torch.float,
            ),
            "future_time_feat": (
                (batch_size, self.prediction_length, self.time_dim),
                torch.float,
            ),
        }

        return info

    @cached_property
    def training_input_names(self) -> List[str]:
        return list(
            ["future_target", "future_observed_values"] + self.prediction_input_names
        )

    @cached_property
    def prediction_input_names(self) -> List[str]:
        return list(self.input_info().keys())

    def loss(
        self,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        loss_fn: Callable = torch.nn.functional.mse_loss,
    ) -> torch.Tensor:
        preds = self(
            past_target, past_observed_values, past_time_feat, future_time_feat
        ).squeeze(1)

        loss_per_sample = weighted_average(
            loss_fn(preds, future_target, reduction="none").flatten(1),
            future_observed_values.flatten(1),
            dim=1,
        )
        return loss_per_sample.mean()

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(self.target_shape) == 0:
            past_target = past_target.unsqueeze(-1)
            past_observed_values = past_observed_values.unsqueeze(-1)

        # Scaling
        scaled_past_target, loc, scale = self.scaler(past_target, past_observed_values)
        preds = self.deep_time(scaled_past_target, past_time_feat, future_time_feat)
        preds = loc + preds * scale

        if len(self.target_shape) == 0:
            preds = preds.squeeze(-1)

        return preds.unsqueeze(1)  # return num_samples=1
