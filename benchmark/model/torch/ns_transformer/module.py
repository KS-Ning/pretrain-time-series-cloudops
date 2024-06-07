from functools import cached_property
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    unsqueeze_expand,
    weighted_average,
)

from .layers import (
    DataEmbedding,
    DecoderLayer,
    Decoder,
    EncoderLayer,
    Encoder,
    DSAttention,
    AttentionLayer,
    Projector,
)


class NSTransformerModel(nn.Module):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        target_dim: int,
        time_dim: int,
        static_dim: int,
        dynamic_dim: int,
        past_dynamic_dim: int,
        static_cardinalities: List[int],
        dynamic_cardinalities: List[int],
        past_dynamic_cardinalities: List[int],
        static_embedding_dim: List[int],
        dynamic_embedding_dim: List[int],
        past_dynamic_embedding_dim: List[int],
        lags_seq: List[int],
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
        p_hidden_dims: List[int] = [128, 128],
        p_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.label_length = prediction_length // 2

        self.target_dim = target_dim
        self.time_dim = time_dim
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.past_dynamic_dim = past_dynamic_dim
        self.static_cardinalities = static_cardinalities
        self.dynamic_cardinalities = dynamic_cardinalities
        self.past_dynamic_cardinalities = past_dynamic_cardinalities
        self.static_embedding_dim = static_embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        self.past_dynamic_embedding_dim = past_dynamic_embedding_dim
        self.lags_seq = lags_seq
        self.num_parallel_samples = num_parallel_samples

        self.target_shape = () if target_dim == 1 else (target_dim,)

        # Embeddings
        self.static_cat_embedder = (
            FeatureEmbedder(
                cardinalities=static_cardinalities,
                embedding_dims=static_embedding_dim,
            )
            if len(static_cardinalities) > 0
            else None
        )
        self.dynamic_cat_embedder = (
            FeatureEmbedder(
                cardinalities=dynamic_cardinalities,
                embedding_dims=dynamic_embedding_dim,
            )
            if len(dynamic_cardinalities) > 0
            else None
        )
        self.past_dynamic_cat_embedder = (
            FeatureEmbedder(
                cardinalities=past_dynamic_cardinalities,
                embedding_dims=past_dynamic_embedding_dim,
            )
            if len(past_dynamic_cardinalities) > 0
            else None
        )

        self.enc_embedding = DataEmbedding(
            target_dim=self.encoder_target_dim,
            feat_dim=self.encoder_feat_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.dec_embedding = DataEmbedding(
            target_dim=target_dim,
            feat_dim=self.decoder_feat_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(
                            mask_flag=True,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        DSAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_decoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            # projection=nn.Linear(d_model, target_dim, bias=True),
            projection=None,
        )

        self.tau_learner = Projector(
            target_dim,
            context_length,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = Projector(
            target_dim,
            context_length,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=context_length,
        )

        self.distr_output = distr_output
        self.out_proj = distr_output.get_args_proj(d_model)
        self.target_shape = distr_output.event_shape

    @property
    def encoder_target_dim(self) -> int:
        return self.target_dim * (
            len(self.lags_seq) + 1
        )  # encoder considers current time step

    @property
    def encoder_feat_dim(self) -> int:
        return (
            self.time_dim
            + self.static_dim
            + self.dynamic_dim
            + self.past_dynamic_dim
            + sum(self.static_embedding_dim)
            + sum(self.dynamic_embedding_dim)
            + sum(self.past_dynamic_embedding_dim)
        )

    @property
    def decoder_feat_dim(self) -> int:
        return (
            self.time_dim
            + self.static_dim
            + self.dynamic_dim
            + sum(self.static_embedding_dim)
            + sum(self.dynamic_embedding_dim)
        )

    @property
    def past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def input_info(
        self, batch_size: int = 1
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        info = {
            "past_target": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
            "past_observed_values": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
            "past_time_feat": (
                (batch_size, self.past_length, self.time_dim),
                torch.float,
            ),
            "future_time_feat": (
                (batch_size, self.prediction_length, self.time_dim),
                torch.float,
            ),
        }
        if self.static_dim > 0:
            info["feat_static_real"] = (
                (batch_size, self.static_dim),
                torch.float,
            )
        if self.dynamic_dim > 0:
            info["feat_dynamic_real"] = (
                (
                    batch_size,
                    self.past_length + self.prediction_length,
                    self.dynamic_dim,
                ),
                torch.float,
            )
        if self.past_dynamic_dim > 0:
            info["past_feat_dynamic_real"] = (
                (batch_size, self.past_length, self.past_dynamic_dim),
                torch.float,
            )
        if len(self.static_cardinalities) > 0:
            info["feat_static_cat"] = (
                (batch_size, len(self.static_cardinalities)),
                torch.long,
            )
        if len(self.dynamic_cardinalities) > 0:
            info["feat_dynamic_cat"] = (
                (
                    batch_size,
                    self.past_length + self.prediction_length,
                    len(self.dynamic_cardinalities),
                ),
                torch.long,
            )
        if len(self.past_dynamic_cardinalities) > 0:
            info["past_feat_dynamic_cat"] = (
                (batch_size, self.past_length, len(self.past_dynamic_cardinalities)),
                torch.long,
            )
        return info

    @cached_property
    def training_input_names(self) -> List[str]:
        return list(
            ["future_target", "future_observed_values"] + self.prediction_input_names
        )

    @cached_property
    def prediction_input_names(self) -> List[str]:
        return list(self.input_info().keys())

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

    def create_encoder_inputs(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        pre_context = past_target[:, : -self.context_length]

        if len(self.target_shape) == 0:
            context = context.unsqueeze(-1)
            observed_context = observed_context.unsqueeze(-1)
            pre_context = pre_context.unsqueeze(-1)

        mean = (
            weighted_average(context, observed_context, dim=1).unsqueeze(1)
        ).detach()
        scaled_context = context - mean
        std = torch.sqrt(
            weighted_average(scaled_context**2, observed_context, dim=1).unsqueeze(1)
            + 1e-5
        ).detach()
        scaled_context = scaled_context / std

        scaled_pre_context = (pre_context - mean) / std

        encoder_targets = self.lagged_sequence_values(
            [0] + self.lags_seq, scaled_pre_context, scaled_context, dim=1
        )

        tau = self.tau_learner(context, std).exp()
        delta = self.delta_learner(context, mean)

        # embeddings
        time_feat = past_time_feat[:, -self.context_length :]
        static_feats = []
        dynamic_feats = [time_feat]

        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if feat_dynamic_real is not None:
            dynamic_feats.append(
                feat_dynamic_real[
                    :, self.past_length - self.context_length : self.past_length
                ]
            )
        if past_feat_dynamic_real is not None:
            dynamic_feats.append(past_feat_dynamic_real[:, -self.context_length :])
        if feat_static_cat is not None and self.static_cat_embedder is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        if feat_dynamic_cat is not None and self.dynamic_cat_embedder is not None:
            dynamic_cat_embed = self.dynamic_cat_embedder(
                feat_dynamic_cat[
                    :, self.past_length - self.context_length : self.past_length
                ]
            )
            dynamic_feats.append(dynamic_cat_embed)
        if (
            past_feat_dynamic_cat is not None
            and self.past_dynamic_cat_embedder is not None
        ):
            past_dynamic_cat_embed = self.past_dynamic_cat_embedder(
                past_feat_dynamic_cat[:, -self.context_length :]
            )
            dynamic_feats.append(past_dynamic_cat_embed)
        static_feats = unsqueeze_expand(
            torch.cat(static_feats, dim=-1), dim=1, size=self.context_length
        )
        dynamic_feats = torch.cat(dynamic_feats, dim=-1)
        encoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)

        enc_in = self.enc_embedding(encoder_targets, encoder_feats)

        return enc_in, scaled_context, tau, delta, mean, std

    def create_decoder_inputs(
        self,
        scaled_context: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
    ):
        zeros = torch.zeros(
            [scaled_context.shape[0], self.prediction_length, scaled_context.shape[2]],
            device=scaled_context.device,
        )
        dec_targets = torch.cat([scaled_context[:, -self.label_length :], zeros], dim=1)

        # features
        time_feat = torch.cat(
            [past_time_feat[:, -self.label_length :], future_time_feat],
            dim=1,
        )
        static_feats = []
        dynamic_feats = [time_feat]

        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if feat_dynamic_real is not None:
            dynamic_feats.append(
                feat_dynamic_real[:, -self.label_length - self.prediction_length :]
            )
        if feat_static_cat is not None and self.static_cat_embedder is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        if feat_dynamic_cat is not None and self.dynamic_cat_embedder is not None:
            dynamic_feats.append(
                self.dynamic_cat_embedder(
                    feat_dynamic_cat[:, -self.label_length - self.prediction_length :]
                )
            )
        static_feats = unsqueeze_expand(
            torch.cat(static_feats, dim=-1),
            dim=1,
            size=self.label_length + self.prediction_length,
        )
        dynamic_feats = torch.cat(dynamic_feats, dim=-1)
        decoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)

        dec_in = self.dec_embedding(dec_targets, decoder_feats)
        return dec_in

    def output(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: torch.Tensor,
        feat_dynamic_real: torch.Tensor,
        past_feat_dynamic_real: torch.Tensor,
        feat_static_cat: torch.Tensor,
        feat_dynamic_cat: torch.Tensor,
        past_feat_dynamic_cat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_in, scaled_context, tau, delta, mean, std = self.create_encoder_inputs(
            past_target,
            past_observed_values,
            past_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )
        enc_out, _ = self.encoder(enc_in, tau=tau, delta=delta)

        dec_in = self.create_decoder_inputs(
            scaled_context,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
        )
        dec_out = self.decoder(dec_in, enc_out, tau=tau, delta=delta)

        if len(self.target_shape) == 0:
            dec_out = dec_out.squeeze(-1)
            mean = mean.squeeze(-1)
            std = std.squeeze(-1)

        return dec_out[:, -self.prediction_length :], mean, std

    def loss(
        self,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
        loss: DistributionLoss = NegativeLogLikelihood(),
    ) -> torch.Tensor:
        out, mean, std = self.output(
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )
        distr_params = self.out_proj(out)
        distr = self.distr_output.distribution(distr_params, loc=mean, scale=std)

        if self.target_shape:
            future_observed_values = future_observed_values.min(dim=-1).values

        loss_per_sample = weighted_average(
            loss(distr, future_target),
            future_observed_values,
            dim=1,
        )
        return loss_per_sample.mean()

    # for prediction
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        num_parallel_samples = num_parallel_samples or self.num_parallel_samples
        out, mean, std = self.output(
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )

        distr_params = self.out_proj(out)
        distr = self.distr_output.distribution(distr_params, loc=mean, scale=std)
        samples = distr.sample(torch.Size([num_parallel_samples]))
        return samples.transpose(0, 1)
