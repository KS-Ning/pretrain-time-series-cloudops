from typing import Optional

import torch
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood

from .module import PatchTSTModel
from util.torch.lightning_module import LightningModule


class PatchTST(LightningModule):
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
        lags_seq: list[int] = None,
        num_parallel_samples: int = 100,
        distr_output: DistributionOutput = StudentTOutput(),
        scaling: bool = True,
        # PatchTST arguements
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 32,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
        # Training
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
    ):
        super().__init__(
            freq,
            prediction_length,
            distr_output.event_shape,
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
        self.model = PatchTSTModel(
            self.freq,
            self.context_length,
            self.prediction_length,
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
            num_parallel_samples=num_parallel_samples,
            distr_output=distr_output,
            scaling=self.scaling,
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
        )
        self.loss_fn = loss

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs) -> torch.Tensor:
        return self.model.loss(*args, **kwargs)
