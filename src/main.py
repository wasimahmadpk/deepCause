import math
# import netCDF
import pickle
import random
import pathlib
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from netCDF4 import Dataset
from itertools import islice
from datetime import datetime
from deepcause import deepCause
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from riverdata import RiverData
from scipy.special import stdtr
from model_test import modelTest
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

np.random.seed(1)
mx.random.seed(2)


def normalize(var):
    nvar = (np.array(var) - np.mean(var)) / np.std(var)
    return nvar


def deseasonalize(var, interval):

    deseasonalize_data = []
    for i in range(interval, len(var)):
        value = var[i] - var[i - interval]
        deseasonalize_data.append(value)
    return deseasonalize_data


def down_sample(data, win_size, partition=None):
    agg_data = []
    daily_data = []
    for i in range(len(data)):
        daily_data.append(data[i])

        if (i % win_size) == 0:

            if partition == None:
                agg_data.append(sum(daily_data) / win_size)
                daily_data = []
            elif partition == 'gpp':
                agg_data.append(sum(daily_data[24: 30]) / 6)
                daily_data = []
            elif partition == 'reco':
                agg_data.append(sum(daily_data[40: 48]) / 8)
                daily_data = []
    return agg_data


def SNR(s, n):
    Ps = np.sqrt(np.mean(np.array(s) ** 2))
    Pn = np.sqrt(np.mean(np.array(n) ** 2))
    SNR = Ps / Pn
    return 10 * math.log(SNR, 10)


def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))


def running_avg_effect(y, yint):

    rae = 0
    for i in range(len(y)):
        ace = 1/((params.get("train_len") + 1 + i) - params.get("train_len")) * (rae + (y[i] - yint[i]))
    return rae



# Synthetic data
freq = '30min'
epochs = 50
win_size = 1

training_length = 555
prediction_length = 15
num_samples = 10
# *********************************************************

# "Load fluxnet-2006 data"
# nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
# nc_fid = Dataset(nc_f, 'r')   # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
# nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);
#
# # Extract data from NetCDF file
# vpd = normalize(nc_fid.variables['VPD_f'][:].ravel().data)  # extract/copy the data

# # *********************************************************
# # "Load fluxnet 2015 data for grassland IT-Mbo site-Half hourly data"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# print(fluxnet.columns)
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']


# LOad synthetic data *************************
df = pd.read_csv("/home/ahmad/Projects/deepCause/datasets/ncdata/synthetic_data.csv")

original_data = []
train_data = []
columns = df.columns
dim = len(df.columns)
print(f"Dimension {dim} and Columns: {df.columns}")

for col in df:
    # print("Col1:", len(df[col]))
    original_data.append(df[col])
    # original_data.append(normalize(down_sample(df[col], win_size)))

original_data = np.array(original_data)

# dataobj = RiverData()
# data = dataobj.get_data()
# xts = data['Kempten']
# yts = data['Dillingen']
# zts = data['Lenggries']

train_ds = ListDataset(
    [
        {'start': "01/01/1961 00:00:00",
         'target': original_data[:, 0: training_length].tolist()
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
    num_layers=7,
    num_cells=70,
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=48
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

# model_path = "models/trained_model_eco22Dec.sav"
model_path = "/home/ahmad/Projects/deepCause/models/trained_model_synth.sav"
filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))



# Generate Knockoffs
category = 3
data_actual = np.array(original_data[:, :]).transpose()
n = len(original_data[:, 0])
obj = Knockoffs()
knockoffs = obj.GenKnockoffs(n, dim, data_actual)


# Ground truth causal graph for synthetic time series
prior_graph = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
               [1, 1, 1, 1]])

params = {
    'num_samples': num_samples,
    'col': columns,
    'pred_len': prediction_length,
    'train_len': training_length,
    'prior_graph': prior_graph,
    'dim': dim,
    'freq': freq
    }

deepCause(original_data, knockoffs, model_path, params)
