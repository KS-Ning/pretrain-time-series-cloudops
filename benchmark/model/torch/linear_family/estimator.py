from typing import Any, Dict, Iterable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic, IterableSlice, PseudoShuffled
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    Transformation,
    Chain,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    SelectFields,
    InstanceSplitter,
)
from gluonts.transform.sampler import InstanceSampler
from torch.utils.data import DataLoader

from .lightning_module import LinearFamilyLightningModule
from ..utils import IterableDataset


class LinearFamilyEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        context_length: Optional[int] = None,
        model_type: str = "linear",
        individual: bool = False,
        loss="mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        trainer_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ):
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
        self.target_shape = () if target_dim == 1 else (target_dim,)

        self.model_type = model_type
        self.individual = individual

        # Training
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = trainer_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        transforms = [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1 + len(self.target_shape),
                dtype=np.float32,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
        ]
        return Chain(transforms)

    def _create_instance_splitter(
        self, module: LinearFamilyLightningModule, mode: str
    ) -> InstanceSplitter:
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model.past_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
            ],
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: LinearFamilyLightningModule,
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
        module: Optional[LinearFamilyLightningModule] = None,
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

    def create_lightning_module(self) -> pl.LightningModule:
        model_kwargs = {
            "freq": self.freq,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "target_dim": self.target_dim,
            "model_type": self.model_type,
            "individual": self.individual,
        }
        return LinearFamilyLightningModule(
            model_kwargs,
            loss=self.loss,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: LinearFamilyLightningModule,
    ) -> PyTorchPredictor:
        transformation = (
            transformation
            + self._create_instance_splitter(module, "test")
            + SelectFields(
                module.model.prediction_input_names + [FieldName.FORECAST_START]
            )
        )

        return PyTorchPredictor(
            input_transform=transformation,
            input_names=module.model.prediction_input_names,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
