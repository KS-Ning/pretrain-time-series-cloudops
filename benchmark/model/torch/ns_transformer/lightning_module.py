from typing import Optional

from gluonts.itertools import prod
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood

from .module import NSTransformerModel
from util.torch.lightning_module import LightningModule


class NSTransformer(LightningModule):
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
        # NSTransformer arguments
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.1,
        p_hidden_dims: list[int] = [128, 128],
        p_hidden_layers: int = 2,
        # Trainer arguments
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__(
            freq,
            prediction_length,
            target_shape=distr_output.event_shape,
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
            lags_seq=lags_seq,
            scaling=False,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()
        self.model = NSTransformerModel(
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
            self.lags_seq,
            num_parallel_samples,
            distr_output,
            d_model,
            n_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            activation,
            dropout,
            p_hidden_dims,
            p_hidden_layers,
        )
        self.loss_fn = loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs, loss=self.loss_fn)
