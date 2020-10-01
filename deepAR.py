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
dim = 5
epochs = 2
win_size = 48

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Code updated at: ", current_time)

training_length = 85  # round((2880)/win_size)  # data for 2 month (Jun-July-Aug*)
prediction_length = 5  # round((144)/win_size)  # data for 2*2 days (last 3 days of Aug)
num_samples = 15

start = round(7200/win_size)
train_stop = start + training_length
test_stop = train_stop + prediction_length
# ******************************************************************

# "Load fluxnet 2015 data for grassland IT-Mbo site"
# fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
# org = fluxnet['SW_IN_F']
# otemp = fluxnet['TA_F']
# ovpd = fluxnet['VPD_F']
# oppt = fluxnet['P_F']
# ogpp = fluxnet['GPP_DT_VUT_50']
# oreco = fluxnet['RECO_NT_VUT_50']

# LOad synthetic data

syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data.csv")
org = syndata['Rg']
oreco = syndata['Reco']
otemp = syndata['T']
ogpp = syndata['GPP']

# ************ Normalize the features *************
rg = normalize(down_sample(org, win_size))
temp = normalize(down_sample(otemp, win_size))
# vpd = normalize(down_sample(ovpd, win_size))
# ppt = normalize(down_sample(oppt, win_size))
gpp = normalize(down_sample(ogpp, win_size))
reco = normalize(down_sample(oreco, win_size))
intervene = np.random.normal(0, 1, len(reco))


corr1 = np.corrcoef(temp, intervene)
corr2 = np.corrcoef(gpp, intervene)
corr3 = np.corrcoef(reco, intervene)
corr4 = np.corrcoef(rg, intervene)
# corr5 = np.corrcoef(ppt, intervene)
# corr6 = np.corrcoef(vpd, intervene)

print("Correlation Coefficient (temp, intervene): ", corr1)
print("Correlation Coefficient (gpp, intervene): ", corr2)
print("Correlation Coefficient (reco, intervene): ", corr3)
print("Correlation Coefficient (rg, intervene): ", corr4)
# print("Correlation Coefficient (ppt, intervene): ", corr5)
# print("Correlation Coefficient (vpd, intervene): ", corr6)

# print("SNR (Temperature)", SNR(temp, intervene))
# print("SNR (GPP)", SNR(gpp, intervene))
# print("SNR (Reco)", SNR(reco, intervene))
# print("SNR (RG)", SNR(rg, intervene))
# print("SNR (PPT)", SNR(ppt, intervene))
# print("SNR (VPD)", SNR(vpd, intervene))

train_ds = ListDataset(
    [
         {'start': "06/01/2003 00:00:00", 
          'target': [reco[start:train_stop],
                     gpp[start:train_stop], temp[start:train_stop],
                     rg[start:train_stop]],
          'dynamic_feat':[gpp[start:train_stop], temp[start:train_stop],
                     rg[start:train_stop]]}
    ],
    freq=freq,
    one_dim_target=False
)

test_ds = ListDataset(
    [
        {'start': "06/01/2003 00:00:00", 
         'target': [reco[start:test_stop],
                    gpp[start:test_stop], temp[start:test_stop],
                    rg[start:test_stop]],
         'dynamic_feat':[gpp[start:test_stop], temp[start:test_stop],
                    rg[start:test_stop]]}
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
    dropout_rate=0.05,
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
y_true = reco[train_stop:train_stop+prediction_length]
mape = mean_absolute_percentage_error(y_true, np.mean(y_pred, axis=0))

print("Y actual:", y_true)
print("Y pred:", y_pred)
print("Y pred mean:", np.mean(y_pred, axis=0))

rmse = sqrt(mean_squared_error(y_true, np.mean(y_pred, axis=0)))
print(f"RMSE: {rmse}, MAPE:{mape} %")
print("Causal strength: ", math.log(rmse/0.1670), 2)

plot_forecasts(tss, forecasts, past_length=14, num_plots=4)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter([pd.DataFrame((tss[0][:][0]))]), iter([forecasts[0].copy_dim(0)]), num_series=len(test_ds))
print("Intervention on Temperature")
print("Performance metrices", agg_metrics)
