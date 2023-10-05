import math
import random
import netCDF
import pickle
import pandas as pd
import numpy as np
import mxnet as mx
import pathlib
import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr
from os import path
from math import sqrt
from fitter import Fitter
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import islice
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.trainer import Trainer
from model_test import modelTest
from sklearn.metrics import mean_squared_error
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from gluonts.evaluation.backtest import make_evaluation_predictions


# np.random.seed(0)
mx.random.seed(0)


class Counterfactuals:

    def __init__(self, model_path, train_vars, test_vars, target, category):

        self.model_path = model_path
        self.train_vars = train_vars
        self.test_vars = test_vars
        self.target = target
        self.cat = category

    def generate(self):

        # Parameters
        freq = 'D'
        dim = 1
        epochs = 150
        win_size = 1

        training_length = 500
        prediction_length = 10
        num_samples = 10

        start = 0
        train_stop = start + training_length
        test_stop = train_stop + prediction_length
        # *********************************************************

        train_ds = ListDataset(
            [
                {'start': "01/01/1961 00:00:00",
                 'target': [self.train_vars[0][start: train_stop]],
                 'dynamic_feat': [self.train_vars[1][start: train_stop], self.train_vars[2][start: train_stop],
                                  self.train_vars[3][start: train_stop]],
                 'cat': [1]
                 },
                {'start': "01/01/1961 00:00:00",
                 'target': [self.train_vars[1][start: train_stop]],
                 'dynamic_feat': [self.train_vars[0][start: train_stop], self.train_vars[2][start: train_stop],
                                  self.train_vars[3][start: train_stop]],
                 'cat': [2]
                 },
                {'start': "01/01/1961 00:00:00",
                 'target': [self.train_vars[2][start: train_stop]],
                 'dynamic_feat': [self.train_vars[0][start: train_stop], self.train_vars[1][start: train_stop],
                                  self.train_vars[3][start: train_stop]],
                 'cat': [3]
                 }
                ,
                {'start': "01/01/1961 00:00:00",
                 'target': [self.train_vars[3][start: train_stop]],
                 'dynamic_feat': [self.train_vars[0][start: train_stop], self.train_vars[1][start: train_stop],
                                  self.train_vars[2][start: train_stop]],
                 'cat': [4]
                 }
            ],
            freq=freq,
            one_dim_target=False
        )

        # create estimator
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length=prediction_length,
            freq=freq,
            num_layers=4,
            num_cells=40,
            dropout_rate=0.05,
            trainer=Trainer(
                ctx="cpu",
                epochs=epochs,
                hybridize=False,
                batch_size=32
            ),
            distr_output=MultivariateGaussianOutput(dim=dim)
        )

        filename = pathlib.Path(self.model_path)
        if not filename.exists():
            print("Training forecasting model....")
            predictor = estimator.train(train_ds)
            # save the model to disk
            pickle.dump(predictor, open(filename, 'wb'))

        mselist = []
        mapelist = []
        start_slide = start
        counterfactual = []
        for i in range(100):
            train_stop = start_slide + training_length
            test_stop = train_stop + prediction_length
            test_ds = ListDataset(
                [
                    {'start': "01/01/1961 00:00:00",
                     'target': [self.target[start_slide: test_stop]],
                     'dynamic_feat': [self.test_vars[0][start_slide: test_stop],
                                      self.test_vars[1][start_slide: test_stop],
                                      self.test_vars[2][start_slide: test_stop]],
                     'cat': [self.cat]
                     }
                ],
                freq=freq,
                one_dim_target=False
            )
            start_slide = start_slide + 10
            idx = 0
            mse, mape, y_pred = modelTest(self.model_path, test_ds, num_samples, self.target, idx, train_stop, test_stop, 888)
            mselist.append(mse)
            mapelist.append(mape)
            counterfactual.extend(y_pred)

        return counterfactual


if __name__ == '__main__':

    def normalize(var):
        nvar = (np.array(var) - np.mean(var)) / np.std(var)
        return nvar


    def deseasonalize(var, interval):
        deseasonalize_data = []
        for i in range(interval, len(var)):
            value = var[i] - var[i - interval]
            deseasonalize_data.append(value)
        return deseasonalize_data

    def down_sample(data, win_size):
        agg_data = []
        monthly_data = []
        for i in range(len(data)):
            monthly_data.append(data[i])
            if (i % win_size) == 0:
                agg_data.append(sum(monthly_data) / win_size)
                monthly_data = []
        return agg_data

    def SNR(s, n):
        Ps = np.sqrt(np.mean(np.array(s) ** 2))
        Pn = np.sqrt(np.mean(np.array(n) ** 2))
        SNR = Ps / Pn
        return 10 * math.log(SNR, 10)

    def mean_absolute_percentage_error(y_true, y_pred):

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    win_size = 1
    interval = 100
    # LOad synthetic data
    syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data_seasonal.csv")
    rg = normalize(deseasonalize(down_sample(np.array(syndata['Rg']), win_size), interval))
    temp = normalize(deseasonalize(down_sample(np.array(syndata['T']), win_size), interval + 7))
    gpp = normalize(deseasonalize(down_sample(np.array(syndata['GPP']), win_size), interval + 7))
    reco = normalize(deseasonalize(down_sample(np.array(syndata['Reco']), win_size), interval + 11))

    # # LOad synthetic data *************************
    # syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv")
    # xts = normalize(down_sample(np.array(syndata['Xts']), win_size))
    # yts = normalize(down_sample(np.array(syndata['Yts']), win_size))
    # zts = normalize(down_sample(np.array(syndata['Zts']), win_size))

    path = "models/counterfactual_model.sav"
    train_vars = [rg, temp, gpp, reco]
    test_vars = [rg, gpp, reco]
    target = temp
    category = 2
    obj = Counterfactuals(path, train_vars, test_vars, target, category)
    counterfactuals = obj.generate()

    print(counterfactuals)
    plt.plot(counterfactuals)
    plt.plot(target[500: 1500])
    plt.show()
    print("Displaying from MAIN")




