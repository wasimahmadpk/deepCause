import pandas as pd
import numpy as np
import netCDF
import pickle
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
from gluonts.trainer import Trainer
from sklearn.metrics import mean_squared_error
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput
from gluonts.evaluation.backtest import make_evaluation_predictions


def normalize(var):
    nvar = (var - np.mean(var)) / np.std(var)
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


def mean_absolute_percentage_error(y_true, y_pred):
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Parameters
freq = 'D'
dim = 6
epochs = 100
win_size = 48

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Code updated at: ", current_time)

training_length = 87  # round((2880)/win_size)  # data for 2 month (Jun-July-Aug*)
prediction_length = 3  # round((144)/win_size)  # data for 2*2 days (last 3 days of Aug)

start = round(7200/win_size)
train_stop = start + training_length
test_stop = train_stop + prediction_length
# ******************************************************************

"Load fluxnet 2015 data for grassland IT-Mbo site"
fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
org = fluxnet['SW_IN_F']
otemp = fluxnet['TA_F']
ovpd = fluxnet['VPD_F']
oppt = fluxnet['P_F']
ogpp = fluxnet['GPP_DT_VUT_50']
oreco = fluxnet['RECO_NT_VUT_50']

# ************ Normalize the features *************
rg = down_sample(normalize(org), win_size)
temp = down_sample(normalize(otemp), win_size)
vpd = down_sample(normalize(ovpd), win_size)
ppt = down_sample(normalize(oppt), win_size)
gpp = down_sample(normalize(ogpp), win_size)
reco = down_sample(normalize(oreco), win_size)
intervene = np.random.normal(0.0001, 0.001, len(reco))


train_ds = ListDataset(
    [
         {'start': "06/01/2003 00:00:00", 
          'target': [reco[start:train_stop], rg[start:train_stop], 
                     gpp[start:train_stop], temp[start:train_stop], 
                     ppt[start:train_stop], vpd[start:train_stop]]}                 
          #'dynamic_feat':[temp[start:train_stop], 
          #                gpp[start:train_stop], rg[start:train_stop],
           #               ppt[start:train_stop], vpd[start:train_stop]]}
    ],
    freq=freq,
    one_dim_target=False
)

test_ds = ListDataset(
    [
        {'start': "06/01/2003 00:00:00", 
         'target': [reco[start:test_stop], rg[start:test_stop], 
                    gpp[start:test_stop], temp[start:test_stop], 
                    ppt[start:test_stop], vpd[start:test_stop]]}         
         #'dynamic_feat':[temp[start:test_stop], 
          #               gpp[start:test_stop], rg[start:test_stop],
           #              ppt[start:test_stop], vpd[start:test_stop]]}
    ],
    freq=freq,
    one_dim_target=False
)

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length+10,
    freq=freq,
    num_layers=6,
    num_cells=60,
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=False,
        batch_size=24
    ),
    distr_output= MultivariateGaussianOutput(dim=dim)
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
    num_samples=prediction_length,  # number of sample paths we want for evaluation
)


def plot_forecasts(tss, forecasts, past_length, num_plots):
    counter = 0
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(14, 10), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.title("Forecasting " + titles[counter] + " time series")
        plt.xlabel("Timestamp")
        plt.ylabel(titles[counter])
        plt.show()
        counter += 1


forecasts = list(forecast_it)
tss = list(ts_it)
titles = ['Reco', 'Temperature', 'Rg', 'GPP']

y_true = reco[train_stop:train_stop+prediction_length]
y_pred = forecasts[0].samples.transpose()[0]
mape = mean_absolute_percentage_error(y_true, y_pred)

print("Y actual:", y_true)
print("Y pred:", y_pred)
print("Y pred mean:", np.mean(y_pred, axis=0))



rmse = sqrt(mean_squared_error(np.array(y_true), np.mean(y_pred, axis=0)))

print(f"RMSE: {rmse}, MAPE:{mape}%")


#plot_forecasts(tss, forecasts, past_length=21, num_plots=4)

#evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

#agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
#print("Intervention on Temperature")
#print("Performance metrices", agg_metrics)
