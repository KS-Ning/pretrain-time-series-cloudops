from dataclasses import dataclass
from typing import Any, Optional, Type

import hydra
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_dataset_builder
from gluonts.dataset.split import split, OffsetSplitter
from gluonts.dataset.common import ProcessDataEntry
from gluonts.itertools import Map
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from .model import torch as torch_models
from .model.torch.distributions import (
    IndependentStudentTOutput,
    MultivariateStudentTOutput,
)
from .callbacks.pytorch_lightning import PyTorchLightningPruningCallback
from util.evaluation import evaluate_model, get_metrics
from util.prepare import (
    create_training_data_loader,
    create_validation_data_loader,
    create_predictor,
)

from util.torch.lightning_module import LightningModule


@dataclass
class Experiment:
    dataset_name: str
    data_path: str
    output_dir: str
    model_name: str
    test: bool
    batch_size: int
    epochs: int
    num_batches_per_epoch: int
    num_workers: int
    num_samples: Optional[int] = 100

    def __post_init__(self):
        self.forecaster_cls = self.get_forecaster_cls(self.model_name)

        self.dataset = load_dataset(
            path=self.data_path,
            name=self.dataset_name,
            split="train_test",
        )

        self.ds_config = load_dataset_builder(
            path=self.data_path,
            name=self.dataset_name,
        ).config

        if self.test:
            self.train_offset = -(
                self.ds_config.prediction_length
                + self.ds_config.stride * (self.ds_config.rolling_evaluations - 1)
            )
            self.validation_offset = None
        else:
            self.validation_offset = -(
                self.ds_config.prediction_length
                + self.ds_config.stride * (self.ds_config.rolling_evaluations - 1)
            )
            self.train_offset = (
                self.validation_offset - self.ds_config.prediction_length
            )

    @staticmethod
    def get_forecaster_cls(model_name: str) -> Type[LightningModule]:
        return getattr(torch_models, model_name)

    def get_params(self, trial) -> dict[str, Any]:
        target_dim = self.ds_config.target_dim
        static_cardinalities = self.ds_config.feat_static_cat_cardinalities("train_test")
        static_dim = self.ds_config.feat_static_real_dim
        past_dynamic_dim = self.ds_config.past_feat_dynamic_real_dim

        model_cfg = {
            "freq": self.ds_config.freq,
            "prediction_length": self.ds_config.prediction_length,
            "lr": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "patience": 3,
        }

        if self.model_name == "TemporalFusionTransformer":
            model_cfg["num_heads"] = trial.suggest_categorical(
                "num_heads", [2, 4, 8]
            )  # default: 4
            model_cfg["d_hidden"] = trial.suggest_categorical(
                "d_hidden", [16, 32, 64]
            )  # default: 32
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8, 12])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim
            model_cfg["past_dynamic_dim"] = past_dynamic_dim

        elif self.model_name == "Autoformer":
            model_cfg["loss"] = "mae"
            model_cfg["factor"] = trial.suggest_int("factor", 2, 5)  # default: 3
            model_cfg["moving_avg"] = trial.suggest_categorical(
                "moving_avg", [13, 25, 37]
            )  # default: 25
            model_cfg["target_dim"] = target_dim
            model_cfg["n_heads"] = 8
            model_cfg["d_model"] = 512
            model_cfg["num_encoder_layers"] = trial.suggest_categorical(
                "num_encoder_layers", [1, 2, 3]
            )  # default: 2
            model_cfg["num_decoder_layers"] = trial.suggest_categorical(
                "num_decoder_layers", [1, 2, 3]
            )  # default: 2
            model_cfg["dim_feedforward"] = 2048
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim
            model_cfg["past_dynamic_dim"] = past_dynamic_dim

        elif self.model_name == "FEDformer":
            model_cfg["loss"] = "mae"
            model_cfg["target_dim"] = target_dim
            model_cfg["version"] = trial.suggest_categorical(
                "version", ["Fourier", "Wavelets"]
            )
            model_cfg["modes"] = 64  # default: 64
            model_cfg["mode_select"] = "random"  # default: random
            model_cfg["base"] = "legendre"  # default: legendre
            model_cfg["cross_activation"] = "tanh"  # default: tanh
            model_cfg["L"] = 3  # default: 3
            model_cfg["moving_avg"] = [24]
            model_cfg["n_heads"] = 8
            model_cfg["d_model"] = 512
            model_cfg["num_encoder_layers"] = trial.suggest_categorical(
                "num_encoder_layers", [1, 2]
            )  # default: 2
            model_cfg["num_decoder_layers"] = trial.suggest_categorical(
                "num_decoder_layers", [1, 2]
            )  # default: 2
            model_cfg["dim_feedforward"] = 2048
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim
            model_cfg["past_dynamic_dim"] = past_dynamic_dim

        elif self.model_name == "NSTransformer":
            if target_dim > 1:
                distr_output = trial.suggest_categorical(
                    "distr_output", ["independent", "multivariate"]
                )
                if distr_output == "independent":
                    model_cfg["distr_output"] = IndependentStudentTOutput(target_dim)
                elif distr_output == "multivariate":
                    model_cfg["distr_output"] = MultivariateStudentTOutput(target_dim)

            model_cfg["p_hidden_dims"] = [
                trial.suggest_categorical("p_hidden_dim", [64, 128, 256])
            ] * 2
            model_cfg["p_hidden_layers"] = 2
            model_cfg["n_heads"] = 8
            model_cfg["d_model"] = 512
            model_cfg["num_encoder_layers"] = trial.suggest_categorical(
                "num_encoder_layers", [1, 2, 3]
            )  # default: 2
            model_cfg["num_decoder_layers"] = trial.suggest_categorical(
                "num_decoder_layers", [1, 2, 3]
            )  # default: 2
            model_cfg["dim_feedforward"] = 2048
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim
            model_cfg["past_dynamic_dim"] = past_dynamic_dim

        elif self.model_name == "PatchTST":
            if target_dim > 1:
                distr_output = trial.suggest_categorical(
                    "distr_output", ["independent", "multivariate"]
                )
                if distr_output == "independent":
                    model_cfg["distr_output"] = IndependentStudentTOutput(target_dim)
                elif distr_output == "multivariate":
                    model_cfg["distr_output"] = MultivariateStudentTOutput(target_dim)

            model_cfg["nhead"] = 16
            model_cfg["d_model"] = trial.suggest_categorical(
                "d_model", [128, 256, 512]
            )
            model_cfg["num_layers"] = trial.suggest_categorical(
                "num__layers", [2, 3, 4]
            )  # default: 3
            model_cfg["dim_feedforward"] = 2 * model_cfg["d_model"]
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim
            model_cfg["past_dynamic_dim"] = past_dynamic_dim

        elif self.model_name == "LinearFamily":
            model_cfg["loss"] = "mae"
            model_cfg["target_dim"] = target_dim
            model_cfg["model_type"] = trial.suggest_categorical(
                "model_type", ["linear", "dlinear", "nlinear"]
            )
            model_cfg["individual"] = trial.suggest_categorical(
                "individual", [True, False]
            )
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8, 12])
                * self.ds_config.prediction_length
            )

        elif self.model_name == "DeepTime":
            model_cfg["loss"] = "mae"
            model_cfg["target_dim"] = target_dim
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8, 12])
                * self.ds_config.prediction_length
            )
            model_cfg["d_model"] = trial.suggest_categorical(
                "d_model", [256, 512, 1024]
            )
            model_cfg["num_layers"] = trial.suggest_categorical(
                "num_layers", [3, 5, 7, 9]
            )
            model_cfg["num_fourier_feats"] = (
                trial.suggest_categorical("num_fourier_feats_multiplier", [4, 8, 16])
                * model_cfg["d_model"]
            )
            time_features = trial.suggest_categorical("time_features", [True, False])
            model_cfg["time_features"] = None if time_features else []

        elif self.model_name == "TimeGrad":
            model_cfg["target_dim"] = target_dim
            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8, 12])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim

            model_cfg["num_layers"] = trial.suggest_int("num_layers", 1, 4)
            model_cfg["hidden_size"] = trial.suggest_int("hidden_size", 20, 80, step=5)

        elif self.model_name == "DeepVAR":
            if target_dim > 1:
                distr_output = trial.suggest_categorical(
                    "distr_output", ["independent", "multivariate"]
                )
                model_cfg["distr_output"] = (
                    IndependentStudentTOutput(target_dim)
                    if distr_output == "independent"
                    else MultivariateStudentTOutput(target_dim)
                )

            model_cfg["context_length"] = (
                trial.suggest_categorical("context_length", [1, 2, 4, 8, 12])
                * self.ds_config.prediction_length
            )
            model_cfg["static_cardinalities"] = static_cardinalities
            model_cfg["static_dim"] = static_dim

            model_cfg["num_layers"] = trial.suggest_int("num_layers", 1, 4)
            model_cfg["hidden_size"] = trial.suggest_int("hidden_size", 20, 80, step=5)

        elif self.model_name.startswith("NBEATS"):
            assert target_dim == 1
            if self.model_name == "NBEATSGeneric":
                model_cfg["model_type"] = "generic"
            elif self.model_name == "NBEATSInterpretable":
                model_cfg["model_type"] = "interpretable"
            elif self.model_name == "NBEATSEnsemble":
                assert self.test, "NBEATSEnsemble is only supported for testing"
                model_cfg["checkpoints"] = []
            else:
                raise ValueError(f"Model {self.model_name} not supported")

        else:
            raise ValueError(f"Model {self.model_name} not supported")

        return model_cfg

    def __call__(self, trial: optuna.Trial) -> float:
        model_cfg = self.get_params(trial)
        trial.set_user_attr("epochs", self.epochs)

        forecaster = self.forecaster_cls(**model_cfg)

        training_data_loader = create_training_data_loader(
            forecaster,
            self.dataset,
            OffsetSplitter(self.train_offset),
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_workers=self.num_workers,
        )

        validation_data_loader = create_validation_data_loader(
            forecaster,
            self.dataset,
            OffsetSplitter(self.validation_offset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, mode="min"
        )
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=10,
            enable_checkpointing=False,
            logger=False,
            callbacks=[
                early_stopping,
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
        )
        trainer.fit(
            model=forecaster,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
        )

        stopped_epoch = early_stopping.stopped_epoch or self.epochs
        trial.set_user_attr("epochs", stopped_epoch - early_stopping.wait_count)
        return early_stopping.best_score

    def evaluate(self, trial: optuna.trial.FrozenTrial):
        model_cfg = self.get_params(trial)
        forecaster = self.forecaster_cls(**model_cfg)

        training_data_loader = create_training_data_loader(
            forecaster,
            self.dataset,
            OffsetSplitter(self.train_offset),
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
            num_workers=self.num_workers,
        )

        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="train_loss", mode="min", verbose=True, dirpath=self.output_dir
        )

        trainer = pl.Trainer(
            max_epochs=trial.user_attrs["epochs"],
            accelerator="auto",
            devices="auto",
            gradient_clip_val=10,
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(
                    monitor="train_loss", patience=10, mode="min"
                ),
            ],
        )
        trainer.fit(
            model=forecaster,
            train_dataloaders=training_data_loader,
        )

        forecaster = forecaster.load_from_checkpoint(checkpoint.best_model_path)
        predictor = create_predictor(forecaster.to("cuda"), batch_size=self.batch_size)

        process = ProcessDataEntry(
            self.ds_config.freq,
            one_dim_target=self.ds_config.univariate,
            use_timestamp=False,
        )
        dataset = Map(process, self.dataset)
        _, test_template = split(dataset, offset=self.train_offset)
        test_data = test_template.generate_instances(
            self.ds_config.prediction_length,
            windows=self.ds_config.rolling_evaluations,
            distance=self.ds_config.stride,
        )

        metrics, agg_metrics = get_metrics(univariate=self.ds_config.univariate)
        results = evaluate_model(
            predictor,
            test_data=test_data,
            metrics=metrics,
            agg_metrics=agg_metrics,
            axis=None,
            seasonality=1,
        )
        results = {k: v[0] for k, v in results.to_dict("list").items()}
        trainer.logger.log_metrics(results)
        print(results)


@hydra.main(version_base="1.1", config_path="conf/", config_name="benchmark_exp")
def main(cfg: DictConfig):
    hydra_cfg: dict[str, any] = OmegaConf.to_container(HydraConfig.get())
    output_dir = hydra_cfg["runtime"]["output_dir"]

    dataset_name = cfg["dataset_name"]
    data_path = cfg["data_path"]
    model_name = cfg["model_name"]
    seed = cfg["seed"]
    test = cfg["test"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    num_batches_per_epoch = cfg["num_batches_per_epoch"]
    num_workers = cfg["num_workers"]
    storage = cfg["storage"]
    n_trials = cfg["n_trials"]
    max_trials = cfg["max_trials"]

    experiment = Experiment(
        dataset_name=dataset_name,
        data_path=data_path,
        output_dir=output_dir,
        model_name=model_name,
        test=test,
        batch_size=batch_size,
        epochs=epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        num_workers=num_workers,
    )

    study_name = f"{dataset_name}_{model_name}"
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
    )

    if test:
        torch.manual_seed(seed)
        np.random.seed(seed)

        experiment.evaluate(study.best_trial)
    else:
        states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
        trials = study.get_trials(deepcopy=False, states=states)
        if len(trials) >= max_trials:
            exit()

        study.optimize(
            experiment,
            n_trials=n_trials,
            callbacks=[MaxTrialsCallback(max_trials, states=states)],
            gc_after_trial=True,
        )


if __name__ == "__main__":
    main()
