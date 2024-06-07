from typing import List, Optional

import torch
from gluonts.core.component import validated
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from .epsilon_theta import EpsilonTheta
from .gaussian_diffusion import GaussianDiffusion, DiffusionOutput
from ..deepvar.module import DeepVARModel
from util.torch.scaler import MeanScaler


class TimeGradModel(DeepVARModel):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        time_dim: int,
        static_dim: int,
        dynamic_dim: int,
        static_cardinalities: list[int],
        static_embedding_dim: list[int],
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
        target_dim: int = 1,
        conditioning_length: int = 100,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
    ):
        denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )

        diffusion = GaussianDiffusion(
            denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        DiffusionOutput(diffusion, input_size=target_dim, cond_size=conditioning_length)
        super().__init__(
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            time_dim=time_dim,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            static_cardinalities=static_cardinalities,
            static_embedding_dim=static_embedding_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            distr_output=DiffusionOutput(
                diffusion, input_size=target_dim, cond_size=conditioning_length
            ),
            lags_seq=lags_seq,
            scaling=scaling,
            num_parallel_samples=num_parallel_samples,
        )
        self.denoise_fn = denoise_fn
        self.diffusion = diffusion
        if scaling:
            self.scaler = MeanScaler(dim=1, keepdim=True)

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        loss: DistributionLoss = NegativeLogLikelihood(),
        future_only: bool = False,
        aggregate_by=torch.mean,
    ) -> torch.Tensor:
        params, loc, scale, _, _, _ = self.unroll_lagged_rnn(
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_static_cat,
            future_target,
        )

        if future_only:
            sliced_params = [p[:, -self.prediction_length :] for p in params]
            if self.scaling:
                self.diffusion.scale = scale
            # diffusion.log_prob is actually the neg log likelihood
            loss_values = self.diffusion.log_prob(future_target, sliced_params[0])
            loss_weights, _ = future_observed_values.min(dim=-1)
            loss = weighted_average(loss_values, weights=loss_weights, dim=1)
        else:
            if self.scaling:
                self.diffusion.scale = scale
            context_target = past_target[:, -self.context_length :]
            target = torch.cat(
                (context_target, future_target),
                dim=1,
            )
            context_observed = past_observed_values[:, -self.context_length :]
            observed_values = torch.cat(
                (context_observed, future_observed_values), dim=1
            )
            loss_weights, _ = observed_values.min(dim=-1)
            loss_values = self.diffusion.log_prob(target, params[0])
            loss = weighted_average(loss_values, weights=loss_weights, dim=1)

        return aggregate_by(
            loss,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.
        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, loc, scale, _, static_feat, state = self.unroll_lagged_rnn(
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat[:, :1],
            feat_static_real,
            feat_static_cat,
        )

        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=1) for s in state
        ]

        repeated_params = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=0) for s in params
        ]

        sliced_params = [p[:, -1:] for p in repeated_params]
        if self.scaling:
            self.diffusion.scale = repeated_scale
        next_sample = self.diffusion.sample(cond=sliced_params[0])
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )

            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = self.lagged_sequence_values(
                self.lags_seq,
                repeated_past_target,
                torch.zeros_like(scaled_next_sample),
                dim=1,
            )
            rnn_input = torch.cat((next_lags, next_features), dim=-1)

            output, repeated_state = self.rnn(rnn_input, repeated_state)

            params = self.param_proj(output)
            if self.scaling:
                self.diffusion.scale = repeated_scale
            next_sample = self.diffusion.sample(cond=params[0])
            future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
            + self.distr_output.event_shape
        )
