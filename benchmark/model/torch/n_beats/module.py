import numpy as np
from torch import nn, Tensor
from gluonts.core.component import validated

from .layers import NBEATSBlock, GenericBasis, SeasonalityBasis, TrendBasis
from util.torch.scaler import StdScaler, NOPScaler


class NBEATSInterpretableModel(nn.Module):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        trend_blocks: int,
        trend_layers: int,
        trend_layer_size: int,
        degree_of_polynomial: int,
        seasonality_blocks: int,
        seasonality_layers: int,
        seasonality_layer_size: int,
        num_of_harmonics: int,
        scale: bool = False,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length

        if scale:
            self.scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        trend_block = NBEATSBlock(
            width=trend_layer_size,
            num_block_layers=trend_layers,
            theta_size=2 * (degree_of_polynomial + 1),
            prediction_length=prediction_length,
            context_length=context_length,
            basis_function=TrendBasis(
                degree_of_polynomial=degree_of_polynomial,
                backcast_size=context_length,
                forecast_size=prediction_length,
            ),
        )
        seasonality_block = NBEATSBlock(
            width=seasonality_layer_size,
            num_block_layers=seasonality_layers,
            theta_size=4
            * int(
                np.ceil(num_of_harmonics / 2 * prediction_length)
                - (num_of_harmonics - 1)
            ),
            prediction_length=prediction_length,
            context_length=context_length,
            basis_function=SeasonalityBasis(
                harmonics=num_of_harmonics,
                backcast_size=context_length,
                forecast_size=prediction_length,
            ),
        )
        self.blocks = nn.ModuleList(
            [trend_block for _ in range(trend_blocks)]
            + [seasonality_block for _ in range(seasonality_blocks)]
        )

    def forward(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
    ) -> Tensor:
        past_target, loc, scale = self.scaler(past_target, past_observed_values)
        residuals = past_target.flip(dims=(1,))
        input_mask = past_observed_values.flip(dims=(1,))
        forecast = past_target[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return loc + forecast * scale


class NBEATSGenericModel(nn.Module):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        stacks: int,
        layers: int,
        layer_size: int,
        scale: bool = False,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length

        if scale:
            self.scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    width=layer_size,
                    num_block_layers=layers,
                    theta_size=context_length + prediction_length,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    basis_function=GenericBasis(
                        backcast_size=context_length,
                        forecast_size=prediction_length,
                    ),
                )
                for _ in range(stacks)
            ]
        )

    def forward(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
    ) -> Tensor:
        past_target, loc, scale = self.scaler(past_target, past_observed_values)
        residuals = past_target.flip(dims=(1,))
        input_mask = past_observed_values.flip(dims=(1,))
        forecast = past_target[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return loc + forecast * scale
