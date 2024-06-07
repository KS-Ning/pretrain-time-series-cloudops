import torch
from torch import nn, Tensor
from gluonts.core.component import validated


class NBEATSBlock(nn.Module):
    @validated()
    def __init__(
        self,
        width: int,
        num_block_layers: int,
        theta_size: int,
        prediction_length: int,
        context_length: int,
        basis_function: nn.Module,
    ):
        super().__init__()

        self.width = width
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_length = theta_size
        self.prediction_length = prediction_length
        self.context_length = context_length

        self.fc_stack = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.context_length, self.width)
                    if i == 0
                    else nn.Linear(self.width, self.width),
                    nn.ReLU(),
                )
                for i in range(self.num_block_layers)
            ]
        )

        self.basis_parameters = nn.Linear(self.width, self.expansion_coefficient_length)
        self.basis_function = basis_function

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = self.fc_stack(x)
        theta = self.basis_parameters(x)
        backcast, forecast = self.basis_function(theta)
        return backcast, forecast


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class SeasonalityBasis(nn.Module):
    @validated()
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()

        frequency = torch.cat(
            [
                torch.zeros(1),
                torch.arange(harmonics, harmonics / 2 * forecast_size) / harmonics,
            ]
        ).unsqueeze(1)

        backcast_grid = (
            -2
            * torch.pi
            * (torch.arange(backcast_size).unsqueeze(0) / backcast_size)
            * frequency
        )

        forecast_grid = (
            2
            * torch.pi
            * (torch.arange(forecast_size).unsqueeze(0) / forecast_size)
            * frequency
        )

        self.register_buffer(
            "backcast_cos_template",
            torch.cos(backcast_grid),
        )
        self.register_buffer(
            "backcast_sin_template",
            torch.sin(backcast_grid),
        )
        self.register_buffer(
            "forecast_cos_template",
            torch.cos(forecast_grid),
        )
        self.register_buffer(
            "forecast_sin_template",
            torch.sin(forecast_grid),
        )

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        params_per_harmonic = theta.shape[1] // 4

        backcast_harmonics_cos = (
            theta[:, :params_per_harmonic] @ self.backcast_cos_template
        )
        backcast_harmonics_sin = (
            theta[:, params_per_harmonic : 2 * params_per_harmonic]
            @ self.backcast_sin_template
        )
        backcast = backcast_harmonics_cos + backcast_harmonics_sin
        forecast_harmonics_cos = (
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic]
            @ self.forecast_cos_template
        )
        forecast_harmonics_sin = (
            theta[:, 3 * params_per_harmonic :] @ self.forecast_sin_template
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        self.polynomial_size = (
            degree_of_polynomial + 1
        )  # degree of polynomial with constant term
        self.register_buffer(
            "backcast_time",
            torch.cat(
                [
                    torch.pow(torch.arange(backcast_size) / backcast_size, i).unsqueeze(
                        0
                    )
                    for i in range(self.polynomial_size)
                ],
                dim=0,
            ),
        )
        self.register_buffer(
            "forecast_time",
            torch.cat(
                [
                    torch.pow(torch.arange(forecast_size) / forecast_size, i).unsqueeze(
                        0
                    )
                    for i in range(self.polynomial_size)
                ],
                dim=0,
            ),
        )

    def forward(self, theta: Tensor):
        backcast = theta[:, self.polynomial_size :] @ self.backcast_time
        forecast = theta[:, : self.polynomial_size] @ self.forecast_time
        return backcast, forecast
