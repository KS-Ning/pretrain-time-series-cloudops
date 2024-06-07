from functools import cached_property
from typing import Callable, Optional

import torch
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature

from .module import DeepTimeModel
from util.torch.lightning_module import LightningModule


class DeepTime(LightningModule):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        time_features: Optional[list[TimeFeature]] = None,
        scaling: bool = True,
        # DeepTime arguments
        target_dim: int = 1,
        d_model: int = 256,
        num_layers: int = 5,
        num_fourier_feats: int = 4096,
        scales: Optional[list[float]] = None,
        # Training arguments
        loss: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ):
        super().__init__(
            freq,
            prediction_length,
            target_shape=() if target_dim == 1 else (target_dim,),
            context_length=context_length,
            time_features=time_features,
            age_feature=True if context_length is None else False,
            lags_seq=[],
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()
        self.model = DeepTimeModel(
            self.freq,
            self.context_length,
            self.prediction_length,
            target_dim,
            self.time_dim,
            self.scaling,
            d_model,
            num_layers,
            num_fourier_feats,
            scales,
        )
        self.loss_fn = self.get_loss_fn(loss)

    @staticmethod
    def get_loss_fn(loss: str) -> Callable:
        if loss == "mse":
            return torch.nn.functional.mse_loss
        elif loss == "mae":
            return torch.nn.functional.l1_loss
        else:
            raise ValueError(
                f"Unknown loss function: {loss}, "
                f"loss function should be one of ('mse', 'mae')."
            )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs, loss_fn=self.loss_fn)
