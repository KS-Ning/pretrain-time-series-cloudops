# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import List, Optional

import torch
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature

from .module import TemporalFusionTransformerModel
from util.torch.lightning_module import LightningModule


class TemporalFusionTransformer(LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``TemporalFusionTransformerModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``TemporalFusionTransformerModel``
    object, that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model
        ``TemporalFusionTransformerModel`` to be trained.
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
        context_length: int,
        prediction_length: int,
        static_dim: int = 0,
        dynamic_dim: int = 0,
        past_dynamic_dim: int = 0,
        static_cardinalities: Optional[list[int]] = None,
        dynamic_cardinalities: Optional[list[int]] = None,
        past_dynamic_cardinalities: Optional[list[int]] = None,
        time_features: Optional[list[TimeFeature]] = None,
        quantiles: Optional[List[float]] = None,
        num_heads: int = 4,
        d_hidden: int = 32,
        d_var: int = 32,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        patience: int = 10,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            freq,
            prediction_length,
            (),
            context_length=context_length,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            past_dynamic_dim=past_dynamic_dim,
            static_cardinalities=static_cardinalities,
            dynamic_cardinalities=dynamic_cardinalities,
            past_dynamic_cardinalities=past_dynamic_cardinalities,
            static_embedding_dim=[
                d_var for _ in range(len(static_cardinalities or []))
            ],
            dynamic_embedding_dim=[
                d_var for _ in range(len(dynamic_cardinalities or []))
            ],
            past_dynamic_embedding_dim=[
                d_var for _ in range(len(past_dynamic_cardinalities or []))
            ],
            time_features=time_features,
            lags_seq=[],
            scaling=True,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            forecast_type="quantile",
        )
        self.save_hyperparameters()

        self.quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.model = TemporalFusionTransformerModel(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            quantiles=self.quantiles,
            d_feat_static_real=[1 for _ in range(self.static_dim)],
            c_feat_static_cat=self.static_cardinalities,
            d_feat_dynamic_real=[1 for _ in range(self.dynamic_dim + self.time_dim)],
            c_feat_dynamic_cat=self.dynamic_cardinalities,
            d_past_feat_dynamic_real=[1 for _ in range(self.past_dynamic_dim)],
            c_past_feat_dynamic_cat=self.past_dynamic_cardinalities,
            num_heads=num_heads,
            d_hidden=d_hidden,
            d_var=d_var,
            dropout_rate=dropout_rate,
        )

    def adjust_inputs(self, kwargs: dict) -> dict:
        if self.time_dim > 0:
            past_time_feat = kwargs["past_time_feat"]
            future_time_feat = kwargs["future_time_feat"]
            time_feat = torch.cat([past_time_feat, future_time_feat], dim=1)
            if self.dynamic_dim > 0:
                kwargs["feat_dynamic_real"] = torch.cat(
                    [kwargs["feat_dynamic_real"], time_feat], dim=-1
                )
            else:
                kwargs["feat_dynamic_real"] = time_feat
            del kwargs["past_time_feat"]
            del kwargs["future_time_feat"]
        return kwargs

    def forward(self, *args, **kwargs):
        kwargs = self.adjust_inputs(kwargs)
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs):
        kwargs = self.adjust_inputs(kwargs)
        return self.model.loss(*args, **kwargs)
