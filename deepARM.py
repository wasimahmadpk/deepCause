import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from gluonts.evaluation import Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from netCDF4 import Dataset
import netCDF

from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

from gluonts.evaluation.backtest import make_evaluation_predictions

# Parameters
prediction_length = 172
freq = '30min'
epochs = 100
start = 45000
train_stop = start + 672
test_stop = train_stop + prediction_length

# ******************************************************************
"Load NC data"
nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);

# Extract data from NetCDF file
vpd_f = nc_fid.variables['VPD_f'][:].ravel().data  # extract/copy the data
tair_f = nc_fid.variables['Tair_f'][:].ravel().data
rg_f = nc_fid.variables['Rg_f'][:].ravel().data
swc1_f = nc_fid.variables['SWC1_f'][:].ravel().data
nee_f = nc_fid.variables['NEE_f'][:].ravel().data
gpp_f = nc_fid.variables['GPP_f'][:].ravel().data
reco = nc_fid.variables['Reco'][:].ravel().data
le_f = nc_fid.variables['LE_f'][:].ravel().data
h_f = nc_fid.variables['H_f'][:].ravel().data
time = nc_fid.variables['time'][:].ravel().data
hour = nc_fid.variables['hour'][:].ravel().data
day = nc_fid.variables['day'][:].ravel().data
month = nc_fid.variables['month'][:].ravel().data
year = nc_fid.variables['year'][:].ravel().data
# ******************************************************************


train_ds = ListDataset(
    [
        {'start': "01/01/2006 00:00:00", 'target': reco[start:train_stop], 'cat': [0], 'dynamic_feat':[tair_f[start:train_stop], rg_f[start:train_stop], gpp_f[start:train_stop]]},
        {'start': "01/01/2006 00:00:00", 'target': tair_f[start:train_stop], 'cat': [1], 'dynamic_feat':[reco[start:train_stop], rg_f[start:train_stop], gpp_f[start:train_stop]]},
        {'start': "01/01/2006 00:00:00", 'target': rg_f[start:train_stop], 'cat': [2], 'dynamic_feat':[reco[start:train_stop], tair_f[start:train_stop], gpp_f[start:train_stop]]},
        {'start': "01/01/2006 00:00:00", 'target': gpp_f[start:train_stop], 'cat': [3], 'dynamic_feat':[reco[start:train_stop], tair_f[start:train_stop], rg_f[start:train_stop]]}
    ],
    freq=freq
)

test_ds = ListDataset(
    [
        {'start': "01/01/2006 00:00:00", 'target': reco[start:test_stop], 'cat': [0], 'dynamic_feat':[tair_f[start:test_stop], rg_f[start:test_stop], gpp_f[start:train_stop]]}
    ],
    freq=freq
)

# {'start': "01/01/2006 00:00:00", 'target': reco[start:test_stop], 'cat': [0], 'dynamic_feat':[tair_f[start:test_stop], rg_f[start:test_stop], gpp_f[start:train_stop]]},
# {'start': "01/01/2006 00:00:00", 'target': tair_f[start:test_stop], 'cat': [1], 'dynamic_feat':[reco[start:test_stop], rg_f[start:test_stop], gpp_f[start:train_stop]]},
#         {'start': "01/01/2006 00:00:00", 'target': rg_f[start:test_stop], 'cat': [2], 'dynamic_feat':[reco[start:test_stop], tair_f[start:train_stop], gpp_f[start:train_stop]]},
#         {'start': "01/01/2006 00:00:00", 'target': gpp_f[start:test_stop], 'cat': [3], 'dynamic_feat':[reco[start:test_stop], tair_f[start:train_stop], rg_f[start:train_stop]]}

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length,
    freq=freq,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=True
    )
)

predictor = estimator.train(train_ds)
print("Training complete:)")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=172,  # number of sample paths we want for evaluation
)


def plot_forecasts(tss, forecasts, past_length, num_plots):
    counter = 0
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
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
plot_forecasts(tss, forecasts, past_length=600, num_plots=4)


evaluator = Evaluator(quantiles=[0.5], seasonality=2006)

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
print("Performance metrices", agg_metrics)
