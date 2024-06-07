from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.itertools import prod
from gluonts.torch.distributions import (
    DistributionOutput,
)
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    unsqueeze_expand,
)

from ..distributions import IndependentStudentTOutput
from util.torch.scaler import NOPScaler, StdScaler


class DeepVARModel(nn.Module):
    """
    Module implementing the DeepAR model, see [SFG17]_.
    *Note:* the code of this model is unrelated to the implementation behind
    `SageMaker's DeepAR Forecasting Algorithm
    <https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html>`_.
    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the RNN unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the RNN.
    hidden_size
        Size of the hidden layers in the RNN.
    dropout_rate
        Dropout rate to be applied at training time.
    distr_output
        Type of distribution to be output by the model at each time step
    lags_seq
        Indices of the lagged observations that the RNN takes as input. For
        example, ``[1]`` indicates that the RNN only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the RNN takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    num_parallel_samples
        Number of samples to produce when unrolling the RNN in the prediction
        time range.
    """

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
        distr_output: DistributionOutput = IndependentStudentTOutput(1),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        num_parallel_samples: int = 100,
    ) -> None:
        super().__init__()

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_dim = time_dim
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.static_cardinalities = static_cardinalities
        self.static_embedding_dim = static_embedding_dim
        self.lags_seq = lags_seq
        self.num_parallel_samples = num_parallel_samples
        self.scaling = scaling

        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(hidden_size)
        self.target_shape = distr_output.event_shape
        self.target_dim = prod(self.target_shape)

        # Scaling
        self.scaler = (
            StdScaler(dim=1, keepdim=True)
            if scaling
            else NOPScaler(dim=1, keepdim=True)
        )

        # Embeddings
        self.static_cat_embedder = (
            FeatureEmbedder(
                cardinalities=static_cardinalities,
                embedding_dims=static_embedding_dim,
            )
            if len(static_cardinalities) > 0
            else None
        )

        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

    @property
    def rnn_input_size(self) -> int:
        return (
            self.target_dim * len(self.lags_seq)
            + self.time_dim
            + self.static_dim
            + self.dynamic_dim
            + sum(self.static_embedding_dim)
            + self.target_dim  # log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    @staticmethod
    def lagged_sequence_values(
        indices: List[int],
        prior_sequence: torch.Tensor,
        sequence: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        lags = lagged_sequence_values(indices, prior_sequence, sequence, dim)
        if lags.dim() > 3:
            lags = lags.reshape(lags.shape[0], lags.shape[1], -1)
        return lags

    def prepare_rnn_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        input, loc, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[1]

        if future_length == 1:
            input = torch.cat(
                (
                    input,
                    torch.zeros(
                        (input.shape[0], 1) + self.target_shape,
                        dtype=input.dtype,
                        device=input.device,
                    ),
                ),
                dim=1,
            )
        elif future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, (future_target[:, :future_length] - loc) / scale),
                dim=1,
            )

        prior_input = (past_target[:, : -self.context_length] - loc) / scale
        lags = self.lagged_sequence_values(self.lags_seq, prior_input, input, dim=1)

        # Features
        dynamic_feats = torch.cat(
            [
                past_time_feat[:, -self.context_length :],
                future_time_feat,
            ],
            dim=1,
        )

        log_scale = torch.log(scale).view(scale.shape[0], -1)
        static_feats = [log_scale]
        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if feat_static_cat is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        static_feats = torch.cat(static_feats, dim=-1)
        expanded_static_feats = unsqueeze_expand(
            static_feats, dim=1, size=dynamic_feats.size(1)
        )
        features = torch.cat((expanded_static_feats, dynamic_feats), dim=-1)
        return torch.cat([lags, features], dim=-1), loc, scale, static_feats

    def unroll_lagged_rnn(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Applies the underlying RNN to the provided target data and covariates.
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
            Tensor of dynamic real features in the future,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length)``.
        Returns
        -------
        Tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the RNN
            - Static input to the RNN
            - Output state from the RNN
        """
        rnn_input, loc, scale, static_feat = self.prepare_rnn_input(
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_static_cat,
            future_target,
        )

        output, new_state = self.rnn(rnn_input)

        params = self.param_proj(output)
        return params, loc, scale, output, static_feat, new_state

    @torch.jit.ignore
    def output_distribution(
        self, params, loc=None, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        """
        Instantiate the output distribution
        Parameters
        ----------
        params
            Tuple of distribution parameters.
        scale
            (Optional) scale tensor.
        trailing_n
            If set, the output distribution is created only for the last
            ``trailing_n`` time points.
        Returns
        -------
        torch.distributions.Distribution
            Output distribution from the model.
        """
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, loc=loc, scale=scale)

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

        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            - repeated_loc
        ) / repeated_scale
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=1) for s in state
        ]

        repeated_params = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=0) for s in params
        ]
        distr = self.output_distribution(
            repeated_params, trailing_n=1, loc=repeated_loc, scale=repeated_scale
        )
        next_sample = distr.sample()
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = (next_sample - repeated_loc) / repeated_scale
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
            distr = self.output_distribution(
                params, loc=repeated_loc, scale=repeated_scale
            )
            next_sample = distr.sample()
            future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
            + self.distr_output.event_shape
        )

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
            distr = self.output_distribution(
                params, loc, scale, trailing_n=self.prediction_length
            )
            if self.target_shape:
                future_observed_values, _ = future_observed_values.min(dim=-1)
            loss_values = loss(distr, future_target) * future_observed_values
        else:
            distr = self.output_distribution(params, loc=loc, scale=scale)
            context_target = past_target[:, -self.context_length :]
            target = torch.cat(
                (context_target, future_target),
                dim=1,
            )
            context_observed = past_observed_values[:, -self.context_length :]
            observed_values = torch.cat(
                (context_observed, future_observed_values), dim=1
            )
            if self.target_shape:
                observed_values, _ = observed_values.min(dim=-1)
            loss_values = loss(distr, target) * observed_values

        return aggregate_by(
            loss_values,
        )
