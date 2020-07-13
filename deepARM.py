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
    
        
# Generate data 

N = 20  # number of time series
T = 1000  # number of timesteps
dim = 2 # dimension of the observations
prediction_length = 150
freq = '1D'


#******************************************************************
"Load NC data"
nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r')   # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
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
#******************************************************************


train_ds = ListDataset(
    [
        {'target': reco[:15000], 'start': 0, 'dynamic_feat': tair_f[:15000]}
    ],
    freq=freq
)



test_ds = ListDataset(
    [
        {'target': reco[:15150], 'start': 0, 'dynamic_feat': tair_f[:15150]},
],
    freq=freq
)

# Trainer parameters
epochs = 100

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

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=150,  # number of sample paths we want for evaluation
)


def plot_forecasts(tss, forecasts, past_length, num_plots):
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.title("Forecasting Ecosystem Respiration")
        plt.xlabel("Timestamp")
        plt.ylabel("R_Eco")
        plt.show()

forecasts = list(forecast_it)
tss = list(ts_it)

plot_forecasts(tss, forecasts, past_length=500, num_plots=1)


evaluator = Evaluator(quantiles=[0.5], seasonality=2006)

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
print("Performance metrices", agg_metrics)
