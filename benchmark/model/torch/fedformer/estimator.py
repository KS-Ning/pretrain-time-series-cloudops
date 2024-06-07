from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from torch.utils.data import DataLoader

from .lightning_module import FEDformerLightningModule
from ..utils import InstanceSplitter, IterableDataset, SampleForecastGenerator


class FEDformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        target_dim: int,
        prediction_length: int,
        context_length: Optional[int] = None,
        static_dim: int = 0,
        dynamic_dim: int = 0,
        past_dynamic_dim: int = 0,
        static_cardinalities: Optional[List[int]] = None,
        dynamic_cardinalities: Optional[List[int]] = None,
        past_dynamic_cardinalities: Optional[List[int]] = None,
        static_embedding_dim: Optional[List[int]] = None,
        dynamic_embedding_dim: Optional[List[int]] = None,
        past_dynamic_embedding_dim: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        scaling: bool = True,
        # FEDformer arguments
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,  # dimension of fcn
        activation: str = "gelu",
        dropout: float = 0.05,
        version: str = "Fourier",  # Fourier, Wavelets
        modes: int = 64,
        mode_select: str = "random",
        base: str = "legendre",
        cross_activation: str = "tanh",
        L: int = 3,
        moving_avg: Optional[List[int]] = None,
        # Trainer arguments
        loss: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length

        self.target_dim = target_dim
        self.static_dim = static_dim or 0
        self.dynamic_dim = dynamic_dim or 0
        self.past_dynamic_dim = past_dynamic_dim or 0
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []
        self.static_embedding_dim = (
            static_embedding_dim or []
            if static_embedding_dim is not None or static_cardinalities is None
            else [min(50, (cat + 1) // 2) for cat in static_cardinalities]
        )
        self.dynamic_embedding_dim = (
            dynamic_embedding_dim or []
            if dynamic_embedding_dim is not None or dynamic_cardinalities is None
            else [min(50, (cat + 1) // 2) for cat in dynamic_cardinalities]
        )
        self.past_dynamic_embedding_dim = (
            past_dynamic_embedding_dim or []
            if past_dynamic_embedding_dim is not None
            or past_dynamic_cardinalities is None
            else [min(50, (cat + 1) // 2) for cat in past_dynamic_cardinalities]
        )
        self.time_features = time_features or time_features_from_frequency_str(freq)

        # Output
        self.target_shape = () if target_dim == 1 else (target_dim,)

        # Scaling
        self.scaling = scaling

        # FEDformer
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.version = version
        self.modes = modes
        self.mode_select = mode_select
        self.base = base
        self.cross_activation = cross_activation
        self.L = L
        self.label_length = self.context_length // 2
        self.moving_avg = moving_avg or [24]

        # Training
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.static_dim == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.dynamic_dim == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.past_dynamic_dim == 0:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if len(self.static_cardinalities) == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)
        if len(self.dynamic_cardinalities) == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)
        if len(self.past_dynamic_cardinalities) == 0:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_CAT)

        transforms = [
            RemoveFields(field_names=remove_field_names),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1 + len(self.target_shape),
                dtype=np.float32,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=self.time_features,
                pred_length=self.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=self.prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
            ),
            AsNumpyArray(
                field=FieldName.FEAT_TIME,
                expected_ndim=2,
                dtype=np.float32,
            ),
        ]

        if self.static_dim > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=np.float32,
                )
            ]

        if self.dynamic_dim > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=2,
                    dtype=np.int64,
                )
            ]

        if self.past_dynamic_dim > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    expected_ndim=2,
                    dtype=np.float32,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    output_field=f"observed_{FieldName.PAST_FEAT_DYNAMIC_REAL}",
                ),
                RemoveFields(
                    field_names=[f"observed_{FieldName.PAST_FEAT_DYNAMIC_REAL}"]
                ),
            ]

        if len(self.static_cardinalities) > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=np.int64
                )
            ]

        if len(self.dynamic_cardinalities) > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_CAT, expected_ndim=2, dtype=np.int64
                )
            ]

        if len(self.past_dynamic_cardinalities) > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.PAST_FEAT_DYNAMIC_CAT,
                    expected_ndim=2,
                    dtype=np.int64,
                )
            ]

        return Chain(transforms)

    def _create_instance_splitter(self, module: FEDformerLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        ts_fields = []
        if self.dynamic_dim > 0:
            ts_fields.append(FieldName.FEAT_DYNAMIC_REAL)
        if len(self.dynamic_cardinalities) > 0:
            ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)

        past_ts_fields = []
        if self.past_dynamic_dim > 0:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if len(self.past_dynamic_cardinalities) > 0:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_CAT)

        return InstanceSplitter(
            instance_sampler=instance_sampler,
            past_length=module.model.past_length,
            future_length=self.prediction_length,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: Optional[FEDformerLightningModule] = None,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "training"
        ) + SelectFields(module.model.training_input_names, allow_missing=False)

        training_instances = transformation.apply(
            Cyclic(data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(data),
                shuffle_buffer_length=shuffle_buffer_length,
            )
        )

        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=self.batch_size,
                    **kwargs,
                )
            ),
            self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: Optional[FEDformerLightningModule] = None,
        **kwargs,
    ) -> Iterable:
        transformation = self._create_instance_splitter(
            module, "validation"
        ) + SelectFields(module.model.training_input_names, allow_missing=False)

        validation_instances = transformation.apply(data)

        return DataLoader(
            IterableDataset(validation_instances),
            batch_size=self.batch_size,
            **kwargs,
        )

    def create_lightning_module(self) -> FEDformerLightningModule:
        model_kwargs = {
            "freq": self.freq,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "target_dim": self.target_dim,
            "time_dim": len(self.time_features) + 1,
            "static_dim": self.static_dim,
            "dynamic_dim": self.dynamic_dim,
            "past_dynamic_dim": self.past_dynamic_dim,
            "static_cardinalities": self.static_cardinalities,
            "dynamic_cardinalities": self.dynamic_cardinalities,
            "past_dynamic_cardinalities": self.past_dynamic_cardinalities,
            "static_embedding_dim": self.static_embedding_dim,
            "dynamic_embedding_dim": self.dynamic_embedding_dim,
            "past_dynamic_embedding_dim": self.past_dynamic_embedding_dim,
            "scaling": self.scaling,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "activation": self.activation,
            "dropout": self.dropout,
            "version": self.version,
            "modes": self.modes,
            "mode_select": self.mode_select,
            "base": self.base,
            "cross_activation": self.cross_activation,
            "L": self.L,
            "moving_avg": self.moving_avg,
        }
        return FEDformerLightningModule(
            model_kwargs,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: FEDformerLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=module.model.prediction_input_names,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            forecast_generator=SampleForecastGenerator(),
        )
