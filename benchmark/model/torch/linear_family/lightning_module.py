from typing import Callable, Optional

import torch
from gluonts.itertools import prod
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature

from .module import LinearFamilyModel
from util.torch.lightning_module import LightningModule


class LinearFamily(LightningModule):
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
        scaling: bool = True,
        # LinearFamily arguments
        target_dim: int = 1,
        model_type: str = "linear",
        individual: bool = False,
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
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            past_dynamic_dim=past_dynamic_dim,
            static_cardinalities=static_cardinalities,
            dynamic_cardinalities=dynamic_cardinalities,
            past_dynamic_cardinalities=past_dynamic_cardinalities,
            static_embedding_dim=static_embedding_dim,
            dynamic_embedding_dim=dynamic_embedding_dim,
            past_dynamic_embedding_dim=past_dynamic_embedding_dim,
            time_features=[],
            age_feature=False,
            lags_seq=[],
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()
        self.model = LinearFamilyModel(
            self.freq,
            self.context_length,
            self.prediction_length,
            prod(self.target_shape),
            model_type,
            individual,
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
