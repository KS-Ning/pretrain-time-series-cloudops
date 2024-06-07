from functools import cached_property
from typing import Callable, Optional

import torch
import torch.nn as nn
from gluonts.core.component import validated
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import weighted_average, unsqueeze_expand

from .layers import (
    series_decomp,
    DataEmbedding_wo_pos,
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    AutoCorrelation,
    AutoCorrelationLayer,
    my_Layernorm,
)
from util.torch.scaler import NOPScaler, StdScaler


class AutoformerModel(nn.Module):
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
        static_cardinalities: list[int],
        dynamic_cardinalities: list[int],
        past_dynamic_cardinalities: list[int],
        static_embedding_dim: list[int],
        dynamic_embedding_dim: list[int],
        past_dynamic_embedding_dim: list[int],
        scaling: bool = True,
        # Autoformer arguments
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.1,
        factor: int = 3,
        moving_avg: int = 25,
    ) -> None:
        super().__init__()

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.label_length = prediction_length // 2

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

        self.target_dim = target_dim
        self.target_shape = () if target_dim == 1 else (target_dim,)

        self.scaler = (
            StdScaler(dim=1, keepdim=True)
            if scaling
            else NOPScaler(dim=1, keepdim=True)
        )

        # Input decomposition
        self.decomp = series_decomp(kernel_size=moving_avg)

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
        self.enc_embedding = DataEmbedding_wo_pos(
            target_dim=self.target_dim, feat_dim=self.encoder_feat_dim, d_model=d_model
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            target_dim=self.target_dim, feat_dim=self.decoder_feat_dim, d_model=d_model
        )

        # autoformer enc-decoder and mask initializer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    dim_feedforward,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    self.target_dim,
                    dim_feedforward,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_decoder_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, self.target_dim, bias=True),
        )

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

    @cached_property
    def past_length(self) -> int:
        return self.context_length

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # target
        scaled_past_target, loc, scale = self.scaler(past_target, past_observed_values)

        # embeddings
        static_feats = []
        dynamic_feats = [past_time_feat]

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

        if len(self.target_shape) == 0:
            scaled_past_target = scaled_past_target.unsqueeze(-1)
            past_observed_values = past_observed_values.unsqueeze(-1)

        enc_in = self.enc_embedding(scaled_past_target, encoder_feats)

        return enc_in, loc, scale, scaled_past_target, past_observed_values

    def create_decoder_inputs(
        self,
        scaled_past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # decomp init
        mean = (
            weighted_average(scaled_past_target, past_observed_values, dim=1)
            .unsqueeze(1)
            .repeat(1, self.prediction_length, 1)
        )
        zeros = torch.zeros(
            [
                scaled_past_target.shape[0],
                self.prediction_length,
                scaled_past_target.shape[2],
            ],
            device=scaled_past_target.device,
        )
        seasonal_init, trend_init = self.decomp(scaled_past_target)
        trend_init = torch.cat([trend_init[:, -self.label_length :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_length :, :], zeros], dim=1
        )

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

        dec_in = self.dec_embedding(seasonal_init, decoder_feats)
        return dec_in, trend_init

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
    ) -> torch.Tensor:
        # encoder
        (
            enc_in,
            loc,
            scale,
            scaled_past_target,
            past_observed_values,
        ) = self.create_encoder_inputs(
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
        enc_out, attns = self.encoder(enc_in, attn_mask=None)

        # decoder
        dec_in, trend_init = self.create_decoder_inputs(
            scaled_past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
        )
        seasonal_part, trend_part = self.decoder(
            dec_in, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )

        dec_out = trend_part + seasonal_part

        if len(self.target_shape) == 0:
            dec_out = dec_out.squeeze(-1)
        return loc + dec_out[:, -self.prediction_length :] * scale

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
        loss_fn: Callable = torch.nn.functional.mse_loss,
    ) -> torch.Tensor:
        preds = self.output(
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

        loss_per_sample = weighted_average(
            loss_fn(preds, future_target, reduction="none").flatten(1),
            future_observed_values.flatten(1),
            dim=1,
        )
        return loss_per_sample.mean()

    # for prediction
    def forward(
        self,
        *,
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
    ) -> torch.Tensor:
        preds = self.output(
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

        preds = preds.reshape((preds.shape[0], 1, preds.shape[1]) + self.target_shape)

        return preds
