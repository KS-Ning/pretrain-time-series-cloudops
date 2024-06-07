import copy
from typing import Callable, Optional
from collections import defaultdict

import pytorch_lightning as pl
import torch
from gluonts.core.component import validated
from gluonts.time_feature import get_seasonality
from torch import Tensor

from .module import NBEATSGenericModel, NBEATSInterpretableModel
from util.torch.lightning_module import LightningModule


class NBEATS(LightningModule):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        # NBEATS arguments
        model_type: str = "generic",
        trend_blocks: int = 3,
        trend_layers: int = 4,
        trend_layer_size: int = 256,
        degree_of_polynomial: int = 3,
        seasonality_blocks: int = 3,
        seasonality_layers: int = 4,
        seasonality_layer_size: int = 2048,
        num_of_harmonics: int = 1,
        stacks: int = 30,
        layers: int = 4,
        layer_size: int = 512,
        scaling: bool = True,
        # Training
        loss: str = "smape",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
    ):
        super().__init__(
            freq,
            prediction_length,
            (),
            context_length=context_length,
            static_dim=0,
            dynamic_dim=0,
            past_dynamic_dim=0,
            static_cardinalities=None,
            dynamic_cardinalities=None,
            past_dynamic_cardinalities=None,
            static_embedding_dim=None,
            dynamic_embedding_dim=None,
            past_dynamic_embedding_dim=None,
            time_features=[],
            age_feature=False,
            lags_seq=[],
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()

        self.model_type = model_type

        if model_type == "interpretable":
            self.model = NBEATSInterpretableModel(
                self.prediction_length,
                self.context_length,
                trend_blocks,
                trend_layers,
                trend_layer_size,
                degree_of_polynomial,
                seasonality_blocks,
                seasonality_layers,
                seasonality_layer_size,
                num_of_harmonics,
                self.scaling,
            )
        elif model_type == "generic":
            self.model = NBEATSGenericModel(
                self.prediction_length,
                self.context_length,
                stacks,
                layers,
                layer_size,
                self.scaling,
            )
        else:
            raise ValueError(f"Unknown model type {model_type}")
        self.periodicity = get_seasonality(freq)
        self.loss_fn = self.get_loss_fn(loss)

    def mape_loss(
        self,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        denominator = torch.abs(future_target)
        flag = (denominator == 0).float()

        absolute_error = torch.abs(future_target - forecast) * future_observed_values

        mape = (100 / self.prediction_length) * torch.mean(
            (absolute_error * (1 - flag)) / (denominator + flag),
            dim=1,
        )
        return mape

    def smape_loss(
        self,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        # Stop gradient required for stable learning
        denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
        flag = (denominator == 0).float()

        absolute_error = torch.abs(future_target - forecast) * future_observed_values

        smape = (200 / self.prediction_length) * torch.mean(
            (absolute_error * (1 - flag)) / (denominator + flag),
            dim=1,
        )

        return smape

    def mase_loss(
        self,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        whole_target = torch.cat([past_target, future_target], dim=1)
        seasonal_error = torch.mean(
            torch.abs(
                whole_target[:, self.periodicity :]
                - whole_target[:, : -self.periodicity]
            ),
            dim=1,
        )
        flag = (seasonal_error == 0).float()

        absolute_error = torch.abs(future_target - forecast) * future_observed_values

        mase = (torch.mean(absolute_error, dim=1) * (1 - flag)) / (
            seasonal_error + flag
        )

        return mase

    def get_loss_fn(
        self, loss: str
    ) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        if loss == "smape":
            return self.smape_loss
        elif loss == "mape":
            return self.mape_loss
        elif loss == "mase":
            return self.mase_loss
        else:
            raise ValueError(
                f"Unknown loss function: {loss}, "
                f"loss function should be one of ('mse', 'mae')."
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs) -> torch.Tensor:
        forecast = self.model(kwargs["past_target"], kwargs["past_observed_values"])
        return self.loss_fn(
            forecast,
            kwargs["future_target"],
            kwargs["past_target"],
            kwargs["future_observed_values"],
        ).mean()


class NBEATSEnsemble(LightningModule):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
        checkpoints: list[str] = list(),
    ):
        super().__init__(
            freq,
            prediction_length,
            (),
            context_length=context_length,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        models = defaultdict(list)
        for path in checkpoints:
            module = NBEATS.load_from_checkpoint(path)
            group = f"{module.model_type},{module.context_length}"
            models[group].append(module.model)
        self.models = torch.nn.ModuleDict(
            {k: torch.nn.ModuleList(v) for k, v in models.items()}
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        forecasts = []
        for group, models in self.models.items():
            context_length = int(group.split(",")[-1])
            base_model = copy.deepcopy(models[0])
            base_model.to("meta")
            params, buffers = torch.func.stack_module_state(models)

            def call_model(p, b, past_target, past_observed_values):
                return torch.func.functional_call(
                    base_model,
                    (p, b),
                    (past_target, past_observed_values),
                )

            past_target = kwargs["past_target"][:, -context_length:]
            past_observed_values = kwargs["past_observed_values"][:, -context_length:]
            forecast = torch.vmap(call_model, (0, 0, None, None))(
                params, buffers, past_target, past_observed_values
            )
            forecasts.append(forecast)

        return torch.cat(forecasts, dim=0).transpose(0, 1)
