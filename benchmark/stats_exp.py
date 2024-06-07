import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import ray
from datasets import (
    load_dataset,
    load_dataset_builder,
    Dataset as HuggingFaceDataset,
)
from gluonts.dataset import Dataset as GluonTSDataset
from gluonts.dataset.common import _FileDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.ev import SumQuantileLoss
from gluonts.ext.statsforecast import (
    StatsForecastPredictor,
    NaivePredictor,
    AutoARIMAPredictor,
    AutoETSPredictor,
    AutoThetaPredictor,
)
from gluonts.time_feature import get_seasonality
from gluonts.transform import (
    AddObservedValuesIndicator,
    MissingValueImputation,
    DummyValueImputation,
)

from util.evaluation import evaluate_forecasts_raw, get_metrics as _get_metrics
from benchmark.model.multivariate_stats_predictors import (
    MultivariateNaivePredictor,
    VARPredictor,
)


def get_metrics(univariate: bool = True):
    umetrics, mmetrics = _get_metrics(univariate=univariate)
    if not univariate:
        mmetrics += [
            (
                SumQuantileLoss(q=q),
                np.sum
            ) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ]
    return umetrics, mmetrics


class LastValueImputation(MissingValueImputation):
    """
    This class replaces each missing value with the last value that was not
    missing.
    (If the first values are missing, they are replaced by the closest non
    missing value.)
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
        if values.ndim == 1:
            values = np.expand_dims(values, axis=0)

        mask = np.isnan(values)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        values[mask] = values[np.nonzero(mask)[0], idx[mask]]

        if values.ndim == 1:
            values = np.squeeze(values)
        # in case we need to replace nan at the start of the array
        mask = np.isnan(values)
        values[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), values[~mask]
        )

        return values


@ray.remote
def forecast_and_evaluate(
    model_name: str,
    model: StatsForecastPredictor,
    dataset: HuggingFaceDataset,
    idx: int,
    fallback_model: NaivePredictor,
    prediction_length: int,
    rolling_evaluations: int,
    stride: int,
    freq: str,
    univariate: bool,
) -> Tuple[str, dict]:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Running {model_name} on time series {idx}")

    dataset = dataset.select([idx])

    offset = prediction_length + stride * (rolling_evaluations - 1)
    dataset = _FileDataset(
        dataset, freq=freq, one_dim_target=univariate
    )

    _, test_template = split(dataset, offset=-offset)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=rolling_evaluations,
        distance=stride,
    )

    transform = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
        imputation_method=LastValueImputation(),
    )

    try:
        test_input = cast(GluonTSDataset, transform(test_data.input, is_train=False))
        forecast = model.predict(test_input)
        umetrics, mmetrics = get_metrics(univariate=univariate)
        metrics_per_ts = evaluate_forecasts_raw(
            forecasts=forecast,
            test_data=test_data,
            metrics=umetrics,
            agg_metrics=mmetrics,
            axis=0,
            seasonality=1
        )
        if isinstance(model, VARPredictor) and metrics_per_ts["sum_absolute_error"].sum() >  metrics_per_ts["sum_absolute_label"].sum():
            raise Exception("VAR model CMI")
    except Exception as e:
        logging.warning(e)

        test_input = cast(GluonTSDataset, transform(test_data.input, is_train=False))
        forecast = fallback_model.predict(test_input)
        umetrics, mmetrics = get_metrics(univariate=univariate)
        metrics_per_ts = evaluate_forecasts_raw(
            forecasts=forecast,
            test_data=test_data,
            metrics=umetrics,
            agg_metrics=mmetrics,
            axis=0,
            seasonality=1
        )

    return model_name, metrics_per_ts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Salesforce/cloudops_tsf",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["naive", "auto_arima", "auto_ets", "auto_theta"],
    )
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--split_idx", type=int, default=0)
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/stats_exp",
        help="output file path",
    )

    args = parser.parse_args()

    # load huggingface dataset
    dataset = load_dataset(
        path=args.data_path,
        name=args.dataset_name,
        split="train_test",
    )
    builder = load_dataset_builder(
        path=args.data_path,
        name=args.dataset_name,
    )

    univariate = builder.config.univariate
    prediction_length = builder.config.prediction_length
    freq = builder.config.freq
    rolling_evaluations = builder.config.rolling_evaluations
    stride = builder.config.stride
    season_length = get_seasonality(freq)

    # Predictors
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    models = {
        "naive": NaivePredictor(prediction_length, quantile_levels=quantile_levels),
        "auto_arima": AutoARIMAPredictor(
            prediction_length,
            season_length=season_length if season_length <= 200 else 1,
            quantile_levels=quantile_levels,
        ),
        "auto_ets": AutoETSPredictor(
            prediction_length,
            damped=True,
            season_length=season_length if season_length <= 24 else 1,
            quantile_levels=quantile_levels,
        ),
        "auto_theta": AutoThetaPredictor(
            prediction_length,
            season_length=season_length,
            quantile_levels=quantile_levels,
        ),
        "multivariate_naive": MultivariateNaivePredictor(
            prediction_length, quantile_levels=quantile_levels
        ),
        "var": VARPredictor(
            prediction_length,
            num_samples=100,
        ),
    }
    models = {k: v for k, v in models.items() if k in args.models}

    if univariate:
        fallback_model = NaivePredictor(
            prediction_length, quantile_levels=quantile_levels
        )
    else:
        fallback_model = MultivariateNaivePredictor(
            prediction_length, quantile_levels=quantile_levels
        )

    # Forecast and Evaluate
    save_path = Path(args.save_path) / args.dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    start = args.split_idx * len(dataset) // args.splits
    end = (args.split_idx + 1) * len(dataset) // args.splits

    ray.init(ignore_reinit_error=True)
    models = {k: ray.put(v) for k, v in models.items()}
    dataset = ray.put(dataset)
    results_ref = []
    for idx in range(start, end):
        for mname, model in models.items():
            rref = forecast_and_evaluate.remote(
                model_name=mname,
                model=model,
                dataset=dataset,
                idx=idx,
                fallback_model=fallback_model,
                prediction_length=prediction_length,
                rolling_evaluations=rolling_evaluations,
                stride=stride,
                freq=freq,
                univariate=univariate,
            )
            results_ref.append(rref)

    results = ray.get(results_ref)
    grouped_results = {mname: [] for mname in models.keys()}

    for mname, metrics_per_ts in results:
        grouped_results[mname].append(metrics_per_ts)

    for mname, result in grouped_results.items():
        with open(save_path / f"{mname}_{args.split_idx}.pkl", "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
