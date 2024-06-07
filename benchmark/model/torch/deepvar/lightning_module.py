from typing import Optional

from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood

from .module import DeepVARModel
from util.torch.lightning_module import LightningModule


class DeepVAR(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``DeepARModel`` with PyTorch Lightning.
    This is a thin layer around a (wrapped) ``DeepARModel`` object,
    that exposes the methods to evaluate training and validation loss.
    Parameters
    ----------
    model
        ``DeepARModel`` to be trained.
    loss
        Loss function to be used for training,
        default: ``NegativeLogLikelihood()``.
    lr
        Learning rate, default: ``1e-3``.
    weight_decay
        Weight decay regularization parameter, default: ``1e-8``.
    patience
        Patience parameter for learning rate scheduler, default: ``10``.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        static_dim: int = 0,
        static_cardinalities: Optional[list[int]] = None,
        static_embedding_dim: Optional[list[int]] = None,
        time_features: Optional[list[TimeFeature]] = None,
        scaling: bool = True,
        lags_seq: Optional[list[int]] = None,
        num_parallel_samples: int = 100,
        distr_output: DistributionOutput = StudentTOutput(),
        # DeepVAR arguments
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        # Trainer arguments
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__(
            freq,
            prediction_length,
            distr_output.event_shape,
            context_length=context_length,
            static_dim=static_dim,
            dynamic_dim=0,
            past_dynamic_dim=0,
            static_cardinalities=static_cardinalities,
            dynamic_cardinalities=None,
            past_dynamic_cardinalities=None,
            static_embedding_dim=static_embedding_dim,
            dynamic_embedding_dim=None,
            past_dynamic_embedding_dim=None,
            time_features=time_features,
            lags_seq=lags_seq,
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()

        self.model = DeepVARModel(
            self.freq,
            self.context_length,
            self.prediction_length,
            time_dim=self.time_dim,
            static_dim=self.static_dim,
            dynamic_dim=self.dynamic_dim,
            static_cardinalities=self.static_cardinalities,
            static_embedding_dim=self.static_embedding_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            distr_output=distr_output,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=num_parallel_samples,
        )
        self.loss_fn = loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.model.loss(*args, **kwargs, loss=self.loss_fn)
