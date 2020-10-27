import math
import netCDF
import pickle
import pandas as pd
import numpy as np
import pathlib
from os import path
from math import sqrt
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import islice
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.trainer import Trainer
from sklearn.metrics import mean_squared_error
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from gluonts.evaluation.backtest import make_evaluation_predictions

def mean_absolute_percentage_error(y_true, y_pred):
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def modelTest(test_ds, num_samples, data, train_stop, test_stop):
    filename = pathlib.Path("trained_model.sav")
    # load the model from disk
    predictor = pickle.load(open(filename, 'rb'))

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=num_samples,  # number of sample paths we want for evaluation
    )


    def plot_forecasts(tss, forecasts, past_length, num_plots):

        for target, forecast in islice(zip(tss, forecasts), num_plots):

            ax = target[-past_length:][0].plot(figsize=(14, 10), linewidth=2)
            forecast.copy_dim(0).plot(color='g')
            plt.grid(which='both')
            plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
            plt.title("Forecasting Reco time series")
            plt.xlabel("Timestamp")
            plt.ylabel('Reco')
            plt.show()


    forecasts = list(forecast_it)
    tss = list(ts_it)

    y_pred = []

    for i in range(num_samples):
        y_pred.append(forecasts[0].samples[i].transpose()[0].tolist())

    y_pred = np.array(y_pred)
    y_true = data[train_stop: test_stop]

    mape = mean_absolute_percentage_error(y_true, np.mean(y_pred, axis=0))
    rmse = sqrt(mean_squared_error(y_true, np.mean(y_pred, axis=0)))

    plot_forecasts(tss, forecasts, past_length=35, num_plots=4)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter([pd.DataFrame((tss[0][:][0]))]), iter([forecasts[0].copy_dim(0)]), num_series=len(test_ds))
    print("Performance metrices", agg_metrics)

    return rmse, mape