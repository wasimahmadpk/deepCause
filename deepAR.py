import math
import netCDF
import pickle
import random
import pathlib
import numpy as np
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
from counterfactuals import Counterfactuals
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


def avg_causal_effect(y, yint):

    ace = 0
    for i in range(len(y)):
        ace = 1/(len(y) - i) * (ace + (y[i] - yint[i])) + ace
    return ace


# # Parameters for fluxnet2006
# freq = '30min'
# epochs = 50
#
# training_length = 720  # data for 15 days
# prediction_length = 48  # data for 1 days
#
# start = 29000
# train_stop = start + training_length
# test_stop = train_stop + prediction_length

# # Parameters for River discharge data
# freq = 'D'
# dim = 3
# epochs = 150
# win_size = 1

# Parameters for ecosystem data
freq = 'D'
dim = 5
epochs = 150
win_size = 48

prediction_length = 14
num_samples = 50

# # Synthetic data
# freq = 'D'
# epochs = 150
# win_size = 1
#
# prediction_length = 15
# num_samples = 50
# # *********************************************************

# "Load fluxnet-2006 data"
# nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
# nc_fid = Dataset(nc_f, 'r')   # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
# nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);
#
# # Extract data from NetCDF file
# vpd = nc_fid.variables['VPD_f'][:].ravel().data  # extract/copy the data
# temp = nc_fid.variables['Tair_f'][:].ravel().data
# rg = nc_fid.variables['Rg_f'][:].ravel().data
# swc1 = nc_fid.variables['SWC1_f'][:].ravel().data
# nee = nc_fid.variables['NEE_f'][:].ravel().data
# gpp = nc_fid.variables['GPP_f'][:].ravel().data
# reco = nc_fid.variables['Reco'][:].ravel().data
# le = nc_fid.variables['LE_f'][:].ravel().data
# h = nc_fid.variables['H_f'][:].ravel().data
# time = nc_fid.variables['time'][:].ravel().data

# //////////////////////////////////////////////////
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

# # # "Load fluxnet 2015 data for grassland IT-Mbo site"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']
# ovpd = fluxnet['VPD_F']
# # oppt = fluxnet['P_F']
# nee = fluxnet['NEE_VUT_50']
# ogpp = fluxnet['GPP_NT_VUT_50']
# oreco = fluxnet['RECO_NT_VUT_50']

# print(fluxnet.head(5))
# print(fluxnet.columns)

interval = 100
# LOad synthetic data *************************
df = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv")
# rg = normalize(down_sample(np.array(df['rg']), win_size))
# temp = normalize(down_sample(np.array(df['temp']), win_size))
# gpp = normalize(down_sample(np.array(df['gpp']), win_size))
# reco = normalize(down_sample(np.array(df['reco']), win_size))

# # LOad FLUXNET2015 data *************************
# rg = normalize(down_sample(org, win_size))
# temp = normalize(down_sample(otemp, win_size))
# gpp = normalize(down_sample(nee, win_size, partition='gpp'))
# reco = normalize(down_sample(nee, win_size, partition='reco'))
# # ppt = normalize(down_sample(oppt, win_size))
# vpd = normalize(down_sample(ovpd, win_size))
# print("Length:", len(rg))
# plt.plot(gpp)
# plt.plot(reco)
# plt.show()


# data = {'Rg': rg, 'T': temp, 'GPP': gpp, 'Reco': reco, 'VPD': vpd}
# ecodata = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco', 'VPD'])

original_data = []
dim = len(df.columns)
# print("Dimension: ", dim)
for col in df:
    # print("Col1:", col)
    original_data.append(df[col])
    # original_data.append(normalize(down_sample(ecodata[col], win_size)))

# xts = normalize(down_sample(df['Xts'], win_size))
# yts = normalize(down_sample(df['Yts'], win_size))
# zts = normalize(down_sample(df['Zts'], win_size))
# rts = normalize(down_sample(df['Rts'], win_size))


# dataobj = RiverData()
# data = dataobj.get_data()
# xts = data['Kempten']
# yts = data['Dillingen']
# zts = data['Lenggries']

xdata = []
ydata = []
zdata = []
rdata = []

ts_len = 75
processed_data_train = []
processed_data_test = []

for col in df.columns:
    data_batches = []
    data_batches_train = []
    initval = 0
    summer_val = 150

    for i in range(ts_len, len(original_data[0])):

        # if i % ts_len == 0:
        if i == summer_val:

            data_batches.append(list(df[col][summer_val: summer_val+ts_len]))
            data_batches_train.append(list(df[col][summer_val: summer_val+ts_len - prediction_length]))
            # data_batches.append(list(ecodata[col][initval: i]))
            # data_batches_train.append(list(ecodata[col][initval: i - prediction_length]))
            # print(len(data_batches_train[:][0]))
            # xdata.append(list(xts[initval: i]))
            # ydata.append(list(yts[initval: i]))
            # zdata.append(list(zts[initval: i]))
            # rdata.append(list(rts[initval: i]))

            # xdata.append(list(xts.iloc[initval: i].values))
            # ydata.append(list(yts.iloc[initval: i].values))
            # zdata.append(list(zts.iloc[initval: i].values))
            # rdata.append(list(reco.iloc[initval: i].values))

            # initval = initval + ts_len
            summer_val = summer_val + 365

    processed_data_train.append(data_batches_train)
    processed_data_test.append(data_batches)

# print("train", processed_data_train[0][0][:])
# print("test", len(processed_data_test[0][0][:]))
# print("train+test", len(processed_data_train), len(processed_data_test))

# # Plot data
# fig = plt.figure()
# ax1 = fig.add_subplot(411)
# ax1.plot(rg[515:615])
# ax1.set_ylabel('Rg')
#
# ax2 = fig.add_subplot(412)
# ax2.plot(temp[515:615])
# ax2.set_ylabel("Temp")
#
# ax3 = fig.add_subplot(413)
# ax3.plot(gpp[515:615])
# ax3.set_ylabel("GPP")
#
# ax4 = fig.add_subplot(414)
# ax4.plot(reco[515:615])
# ax4.set_ylabel("Reco")
# plt.show()
#
# # Plot data
# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax1.hist(rg)
# ax1.set_ylabel('Xt')
#
# ax2 = fig.add_subplot(222)
# ax2.hist(temp)
# ax2.set_ylabel("Yt")
#
# ax3 = fig.add_subplot(223)
# ax3.hist(gpp)
# ax3.set_ylabel("Zt")
#
# ax4 = fig.add_subplot(224)
# ax4.hist(reco)
# ax4.set_ylabel("Rt")
# plt.show()

train_ds = ListDataset(
    [
        {'start': "01/01/1961 00:00:00",
         'target': list(time_series_train),
         'cat': [i]
         }
        # for i, (xts, yts, zts, rts) in enumerate(zip(xdata, ydata, zdata, rdata))
        for i, time_series_train in enumerate(zip(*processed_data_train))
    ],
    freq=freq,
    one_dim_target=False
)

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length,
    freq=freq,
    num_layers=5,
    num_cells=50,
    dropout_rate=0.075,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=24
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)
model_path = "models/trained_model_eco.sav"
filename = pathlib.Path(model_path)
if not filename.exists():
    print("Training forecasting model....")
    predictor = estimator.train(train_ds)
    # save the model to disk
    pickle.dump(predictor, open(filename, 'wb'))


# # Test the model
#
# path = "models/counterfactual_model.sav"
# train_vars = [rg, temp, gpp, reco]
# test_vars = [temp, gpp, reco]
# target = rg
# category = 1
# obj = Counterfactuals(path, train_vars, test_vars, target, category)
# counterfactuals = obj.generate()
# np.random.shuffle(counterfactuals)

# plt.plot(counterfactuals)
# plt.plot(target[500: 1500])
# plt.show()

# Generate Knockoffs
category = 4
# target = xts
data_actual = np.array(original_data).transpose()
n = len(original_data[:][0])
obj = Knockoffs()
knockoffs = obj.GenKnockoffs(n, dim, data_actual)
counterfactuals = np.array(knockoffs[:, category-1])
# print("Deep Knockoffs: \n", counterfactuals)

# plt.plot(np.arange(0, len(counterfactuals)), original_data[3][: len(counterfactuals)], counterfactuals)
# plt.show()

# corr = np.corrcoef(counterfactuals, target[0: len(counterfactuals)])
# print(f"Correlation Coefficient (Variable, Counterfactual): {corr}")


params = {
    'num_samples': num_samples,
    'pred_len': prediction_length,
    'ts_len': ts_len,
    'dim': dim,
    'freq': freq
    }


deepCause(original_data, processed_data_test, knockoffs, model_path, params)
# # num_samples, target, idx-1, prediction_length, i, False, 0
# mulvareco = [xts[0:ts_len], yts[0:ts_len], zts[0:ts_len], rts[0: ts_len]]
#
# mean = [np.zeros(len(mulvareco[ts])) + np.mean(mulvareco[ts]) for ts in range(len(mulvareco))]
# indist = [np.random.normal(np.mean(mulvareco[ts]), np.var(mulvareco[ts]), len(mulvareco[ts])) for ts in range(len(mulvareco))]
# outdist = np.random.normal(0, 0.001, ts_len)
#
# interventionlist = [mean, counterfactuals, outdist]
# heuristic_itn_types = ['Mean', 'In-dist', 'Out-dist']
# css_list = []
# css_list_new = []
# css_score_new = []
# mselol = []
# mapelol = []
# acelol = []
# mselolint = []
# mapelolint = []
# ts_number = category-1
#
#
# for m in range(len(interventionlist)):  # apply all interventions len(interventionlist)
#
#     if m > 0:
#         intervene = interventionlist[m]
#     else:
#         intervene = interventionlist[m][ts_number]
#
#     mselist = []
#     mselistint = []
#     acelist = []
#     mapelist = []
#     mapelistint = []
#     css_score = []
#     diff = []
#     for i, (xts, yts, zts, rts) in enumerate(zip(xdata, ydata, zdata, rdata)):
#         test_ds = ListDataset(
#             [
#                 {'start': "01/01/1961 00:00:00",
#                  'target': [xts, yts, zts, rts],
#                  'cat': [i]
#                  }
#             ],
#             freq=freq,
#             one_dim_target=False
#         )
#         # rg[0:-50] + list(intervene[-50:])
#         test_dsint = ListDataset(
#             [
#                 {'start': "01/01/1961 00:00:00",
#                  'target': [intervene, yts, zts, rts],
#                  'cat': [i]
#                  }
#             ],
#             freq=freq,
#             one_dim_target=False
#         )
#
#         idx = 4
#         target = rts
#         model_path = "models/trained_model_cl10.sav"
#         mse, mape, ypred = modelTest(model_path, test_ds, num_samples, target, idx-1, prediction_length, i, False, 0)
#         mseint, mapeint, ypredint = modelTest(model_path, test_dsint, num_samples, target, idx-1, prediction_length, i, True, m)
#
#         mselist.append(mse)
#         mapelist.append(mape)
#         mselistint.append(mseint)
#         mapelistint.append(mapeint)
#         acelist.append(avg_causal_effect(np.array(ypred), np.array(ypredint)))
#
#         target_before = np.array(target[:ts_len] + ypred)
#         target_after = np.array(target[:ts_len] + ypredint)
#
#         diff.append(abs(target_after - target_before))
#
#     mse = np.mean(mselist)
#     mape = np.mean(mapelist)
#     mselol.append(mselist)
#     mapelol.append(mapelist)
#     acelol.append(acelist)
#     # print(f"MSE: {mselist}, MAPE: {mape}%")
#     # print(f"ACE: {acelist}")
#
#     mse = np.mean(mselistint)
#     mape = np.mean(mapelistint)
#
#     mselolint.append(mselistint)
#     mapelolint.append(mapelistint)
#     # print(f"MSE: {mselistint}, MAPE: {mape}%")
#     # avg_diff = np.mean(diff, axis=0)
#     # plt.plot(avg_diff)
#     # plt.show()
#
#     for k in range(len(mselist)):
#         css_score.append(np.log(mapelistint[k] / mapelist[k]))
#         # css_score.append(np.log(mselistint[k] / mselist[k]))
#         # css_score.append(abs(mselistint[k] - mselist[k]))
#
#     # css_score = [abs(x) if x < 0 else x for x in css_score]
#     css_list.append(css_score)
#     # plt.hist(css_score)
#     # plt.show()
#     print("CSS: ", css_score)
#     # print("Before Intervention: ", mselist)
#     # print("After Intervention: ", mselistint)
#
# # print(f"MSE(Mean): {list(np.mean(mselol, axis=0))}")
# for z in range(len(heuristic_itn_types)):
#     print(f"Average Causal Strength using {heuristic_itn_types[z]} Intervention: {np.mean(css_list[z])}")
#     # print(f"Average Causal Strength using {heuristic_itn_types[z]} Intervention: {np.mean(np.array(mselolint[z]) - np.array(mselol[z]))}")
#     # print("CSS: ", css_score)
#     # t, p = ttest_ind(np.array(mselolint[z]), np.array(mselol[z]), equal_var=False)
#     # t, p = ttest_1samp(np.array(mselolint[z]) - np.array(mselol[z]), popmean=0.0)
#     t, p = ttest_1samp(css_list[z], popmean=0.0)
#     # t, p = ttest_ind(mselolint[z], mselol[z], equal_var=False)
#     # plt.hist(mselolint[z])
#     # plt.hist(mselol[z])
#     # plt.show()
#     print(f'Test statistic: {t}, p-value: {p}')
#     if p < 0.05:
#         print("Null hypothesis is rejected")
#     else:
#         print("Fail to reject null hypothesis")
#
#     # print(f"Average Causal Impact using {heuristic_itn_types[z]} Intervention: {np.mean(acelol[z])}")
#     # t, p = ttest_1samp(acelol[z], 0)
#     # print(f'Test statistic: {t}, p-value: {p}')
#     # if p < 0.05:
#     #     print("Null hypothesis is rejected")
#     # else:
#     #     print("Fail to reject null hypothesis")
#
