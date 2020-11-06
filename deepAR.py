import math
import random
import netCDF
import pickle
import pandas as pd
import numpy as np
import pathlib
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


def normalize(var):
    nvar = (np.array(var) - np.mean(var)) / np.std(var)
    return nvar


def down_sample(data, win_size):
    agg_data = []
    monthly_data = []
    for i in range(len(data)):
        monthly_data.append(data[i])
        if (i % win_size) == 0:
            agg_data.append(sum(monthly_data)/win_size)
            monthly_data = []
    return agg_data


def SNR(s, n):
        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        return 10*math.log(((Ps-Pn)/Pn), 10)


def mean_absolute_percentage_error(y_true, y_pred):
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Parameters
freq = 'D'
dim = 4
epochs = 100
win_size = 24

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Code updated at: ", current_time)

training_length = 950
prediction_length = 50
num_samples = 24

start = 33
train_stop = start + training_length
test_stop = train_stop + prediction_length
# *********************************************************

# "Load meteriological data (DWD)"
# col_names = ['temperature', 'sunshine', 'altitude', 'precipitation', 'longitude', 'latitude']
# dwd = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/DWD/DWD_labels.csv", sep=';')
# print(dwd.head())
# temp = dwd['temperature']
# sunshine = dwd['sunshine']
# alt = dwd['altitude']
# ppt = dwd['precipitation']
# long = dwd['longitude']
# lat = dwd['latitude']

# "Load fluxnet 2015 data for grassland IT-Mbo site"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']
# ovpd = fluxnet['VPD_F']
# oppt = fluxnet['P_F']
# ogpp = fluxnet['GPP_DT_VUT_50']
# oreco = fluxnet['RECO_NT_VUT_50']
# plt.hist(oppt, 1000)
# plt.show()

# LOad synthetic data
syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv")
ats = down_sample(np.array(syndata['Rg']), win_size)
bts = down_sample(np.array(syndata['T']), win_size)
cts = down_sample(np.array(syndata['GPP']), win_size)
dts = down_sample(np.array(syndata['Reco']), win_size)

# # ************ Normalize the features *************
# rg = normalize(down_sample(org, win_size))
# temp = normalize(down_sample(otemp, win_size))
# gpp = normalize(down_sample(ogpp, win_size))
# reco = normalize(down_sample(oreco, win_size))
# vpd = normalize(down_sample(ovpd, win_size))
# ppt = normalize(down_sample(oppt, win_size))
#
# # **********Fit distribution **********************
# dfname = pathlib.Path("tseries.dist")
# if not dfname.exists():
#     print('fitting distribution')
#     fdist = Fitter(alt)
#     fdist.fit()
#     with open(dfname, 'wb') as f:
#         pickle.dump(fdist, f)

# with open(dfname, 'rb') as f:
#     fdist = pickle.load(f)
#
# dist_pdf = fdist.fitted_pdf.get(list(fdist.get_best().keys())[0])
# print(fdist.get_best().get(list(fdist.get_best().keys())[0]))
# print('Best Fit: ', list(fdist.get_best().keys())[0])
# # fdist.summary()
# plt.plot(dist_pdf)
# plt.show()
#

train_ds = ListDataset(
    [
         {'start': "01/01/1961 00:00:00",
          'target': [ats[start: train_stop], bts[start: train_stop],
                     cts[start: train_stop], dts[start: train_stop]]
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
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=32
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

filename = pathlib.Path("trained_model.sav")
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))


# intervene = random.choices(np.linspace(fdist.get_best().get(list(fdist.get_best().keys())[0])[2],
#                                        fdist.get_best().get(list(fdist.get_best().keys())[0])[2] + fdist.get_best().get(list(fdist.get_best().keys())[0])[3],
#                                        len(dist_pdf.tolist())), weights=tuple(dist_pdf),
#                                        k=len(temp))
# intervene = np.empty(len(ats))
# intervene.fill(np.mean(dts))

intervene = np.random.choice(cts, len(dts)) + np.random.normal(0, 0.1, len(dts))

# corr1 = np.corrcoef(temp, intervene)
# corr2 = np.corrcoef(gpp, intervene)
# corr3 = np.corrcoef(reco, intervene)
# corr4 = np.corrcoef(rg, intervene)
# corr5 = np.corrcoef(ppt, intervene)
# corr6 = np.corrcoef(vpd, intervene)

# print("Correlation Coefficient (temp, intervene): ", corr1)
# print("Correlation Coefficient (gpp, intervene): ", corr2)
# print("Correlation Coefficient (reco, intervene): ", corr3)
# print("Correlation Coefficient (rg, intervene): ", corr4)
# print("Correlation Coefficient (ppt, intervene): ", corr5)
# print("Correlation Coefficient (vpd, intervene): ", corr6)

# print("SNR (Temperature)", SNR(temp, intervene))
# print("SNR (GPP)", SNR(gpp, intervene))
# print("SNR (Reco)", SNR(reco, intervene))
# print("SNR (RG)", SNR(rg, intervene))
# print("SNR (PPT)", SNR(ppt, intervene))
# print("SNR (VPD)", SNR(vpd, intervene))

# test model
rmselist = []
mapelist = []

for i in range(10):

    start = start + 5
    train_stop = start + training_length
    test_stop = train_stop + prediction_length

    test_ds = ListDataset(
        [
            {'start': "01/01/1961 00:00:00",
             'target': [ats[start: test_stop], bts[start: test_stop],
                        cts[start: test_stop], dts[start: test_stop]]
             }
        ],
        freq=freq,
        one_dim_target=False
    )

    rmse, mape = modelTest(test_ds, num_samples, ats, train_stop, test_stop, i)
    rmselist.append(rmse)
    mapelist.append(mape)

rmse = np.mean(rmselist)
mape = np.mean(mapelist)

print(f"RMSE: {rmse}, MAPE:{mape}%")
print("Causal strength: ", math.log(rmse/2435.0911), 2)
