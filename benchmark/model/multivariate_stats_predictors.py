from typing import Optional, Type, List, Callable

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import (
    SampleForecast,
    QuantileForecast as _QuantileForecast,
)
from gluonts.model.predictor import RepresentablePredictor
from gluonts.ext.statsforecast import ModelConfig
from gluonts.ext.naive_2._predictor import naive_2, get_seasonality
from statsmodels.tsa.api import VAR
from statsforecast.models import Naive, SeasonalNaive


class QuantileForecast(_QuantileForecast):
    def copy_aggregate(self, agg_fun: Callable) -> "QuantileForecast":
        if len(self.forecast_array.shape) == 2:
            forecast_array = self.forecast_array
        else:
            # Aggregate over target dimension axis
            forecast_array = agg_fun(self.forecast_array, axis=2)
        return QuantileForecast(
            forecast_arrays=forecast_array,
            forecast_keys=self.forecast_keys,
            start_date=self.start_date,
            item_id=self.item_id,
            info=self.info,
        )


class VARPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsmodels`_ package.
    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsmodel`` model type to use.
    .. _statsmodels: https://github.com/statsmodels/statsmodels
    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    num_samples
        Number of samples to draw from the model.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsmodels``
        for details.
    """

    ModelType: Type = VAR

    @validated()
    def __init__(
        self,
        prediction_length: int,
        num_samples: Optional[int] = 100,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model_params = model_params
        self.num_samples = num_samples

    def predict_item(self, entry: DataEntry) -> SampleForecast:
        target = entry["target"].T
        result = self.ModelType(target, **self.model_params).fit(ic="aic")
        mean = result.forecast(target, steps=self.prediction_length)
        cov = result.forecast_cov(self.prediction_length)
        predictions = MultivariateNormal(
            torch.as_tensor(mean), torch.as_tensor(cov)
        ).sample((self.num_samples,))
        predictions = predictions.numpy()

        return SampleForecast(
            samples=predictions,
            start_date=forecast_start(entry),
            item_id=entry.get("item_id"),
            info=entry.get("info"),
        )


class MultivariateStatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.
    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsforecast`` model type to use.
    .. _statsforecast: https://github.com/Nixtla/statsforecast
    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsforecast``
        for details.
    """

    ModelType: Type

    @validated()
    def __init__(
        self,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.model = self.ModelType(**model_params)
        self.config = ModelConfig(quantile_levels=quantile_levels)

    def predict_item(self, entry: DataEntry) -> QuantileForecast:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals

        forecast_arrays = []
        for target in entry["target"]:
            prediction = self.model.forecast(
                y=target,
                h=self.prediction_length,
                **kwargs,
            )

            dim_forecast_arrays = np.stack(
                [prediction[k] for k in self.config.statsforecast_keys], axis=0
            )
            forecast_arrays.append(dim_forecast_arrays)

        return QuantileForecast(
            forecast_arrays=np.stack(forecast_arrays, axis=-1),  # N, T, C
            forecast_keys=self.config.forecast_keys,
            start_date=forecast_start(entry),
            item_id=entry.get("item_id"),
            info=entry.get("info"),
        )


class MultivariateNaivePredictor(MultivariateStatsForecastPredictor):
    ModelType = Naive


class MultivariateSeasonalNaivePredictor(MultivariateStatsForecastPredictor):
    ModelType = SeasonalNaive


class MultivariateNaive2Predictor(RepresentablePredictor):
    """
    Naïve 2 forecaster as described in the M4 Competition Guide:
    https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-
    Guide.pdf.
    The python analogue implementation to:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R#L118
    Parameters
    ----------
    freq
        Frequency of the input data
    prediction_length
        Number of time points to predict
    season_length
        Length of the seasonality pattern of the input data
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        freq: Optional[str] = None,
        season_length: Optional[int] = None,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        assert (
            season_length is None or season_length > 0
        ), "The value of `season_length` should be > 0"
        assert freq is not None or season_length is not None, (
            "Either the frequency or season length of the time series "
            "has to be specified. "
        )

        self.freq = freq
        self.prediction_length = prediction_length
        self.season_length = (
            season_length if season_length is not None else get_seasonality(freq)
        )

    def predict_item(self, item: DataEntry) -> SampleForecast:
        item_id = item.get("item_id", None)
        forecast_start_time = forecast_start(item)

        samples = []
        for past_ts_data in item["target"]:
            assert (
                len(past_ts_data) >= 1
            ), "all time series should have at least one data point"

            prediction = naive_2(past_ts_data, self.prediction_length, self.freq)

            samples.append(np.array([prediction]))

        return SampleForecast(
            samples=np.stack(samples, axis=-1),
            start_date=forecast_start_time,
            item_id=item_id,
        )
