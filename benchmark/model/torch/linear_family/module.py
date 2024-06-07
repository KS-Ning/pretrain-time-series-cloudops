from functools import cached_property
from typing import List, Dict, Tuple, Callable

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.util import (
    weighted_average,
)

from .layers import Linear, DLinear, NLinear


class LinearFamilyModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        target_dim: int,
        model_type: str,
        individual: bool,
    ):
        super().__init__()

        if model_type == "linear":
            linear_model = Linear
        elif model_type == "dlinear":
            linear_model = DLinear
        elif model_type == "nlinear":
            linear_model = NLinear
        else:
            raise ValueError(
                f"Unknown model type: {model_type}, "
                f"model type should be one of ('linear', 'dlinear', 'nlinear')."
            )

        self.linear_model = linear_model(
            seq_len=context_length,
            pred_len=prediction_length,
            enc_in=target_dim,
            individual=individual,
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.target_shape = () if target_dim == 1 else (target_dim,)

    @cached_property
    def past_length(self) -> int:
        return self.context_length

    def input_info(
        self, batch_size: int = 1
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        return {
            "past_target": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
            "past_observed_values": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
        }

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
        loss_fn: Callable = torch.nn.functional.mse_loss,
    ) -> torch.Tensor:
        preds = self(past_target, past_observed_values).squeeze(1)

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
    ) -> torch.Tensor:
        if len(self.target_shape) == 0:
            past_target = past_target.unsqueeze(-1)

        preds = self.linear_model(past_target)

        if len(self.target_shape) == 0:
            preds = preds.squeeze(-1)

        return preds.unsqueeze(1)  # num_samples=1
