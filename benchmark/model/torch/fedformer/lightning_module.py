from typing import Callable, Optional

import torch
from gluonts.itertools import prod
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature

from .module import FEDformerModel
from util.torch.lightning_module import LightningModule


class FEDformer(LightningModule):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        static_dim: int = 0,
        dynamic_dim: int = 0,
        past_dynamic_dim: int = 0,
        static_cardinalities: Optional[list[int]] = None,
        dynamic_cardinalities: Optional[list[int]] = None,
        past_dynamic_cardinalities: Optional[list[int]] = None,
        static_embedding_dim: Optional[list[int]] = None,
        dynamic_embedding_dim: Optional[list[int]] = None,
        past_dynamic_embedding_dim: Optional[list[int]] = None,
        time_features: Optional[list[TimeFeature]] = None,
        scaling: bool = True,
        # FEDformer arguments
        target_dim: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,  # dimension of fcn
        activation: str = "gelu",
        dropout: float = 0.05,
        version: str = "Fourier",  # Fourier, Wavelets
        modes: int = 64,
        mode_select: str = "random",
        base: str = "legendre",
        cross_activation: str = "tanh",
        L: int = 3,
        moving_avg: Optional[list[int]] = None,
        # Trainer arguments
        loss: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__(
            freq,
            prediction_length,
            target_shape=() if target_dim == 1 else (target_dim,),
            context_length=context_length,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            past_dynamic_dim=past_dynamic_dim,
            static_cardinalities=static_cardinalities,
            dynamic_cardinalities=dynamic_cardinalities,
            past_dynamic_cardinalities=past_dynamic_cardinalities,
            static_embedding_dim=static_embedding_dim,
            dynamic_embedding_dim=dynamic_embedding_dim,
            past_dynamic_embedding_dim=past_dynamic_embedding_dim,
            time_features=time_features,
            lags_seq=[],
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()
        self.model = FEDformerModel(
            self.freq,
            self.context_length,
            self.prediction_length,
            prod(self.target_shape),
            self.time_dim,
            self.static_dim,
            self.dynamic_dim,
            self.past_dynamic_dim,
            self.static_cardinalities,
            self.dynamic_cardinalities,
            self.past_dynamic_cardinalities,
            self.static_embedding_dim,
            self.dynamic_embedding_dim,
            self.past_dynamic_embedding_dim,
            self.scaling,
            d_model,
            n_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            activation,
            dropout,
            version,
            modes,
            mode_select,
            base,
            cross_activation,
            L,
            moving_avg,
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
