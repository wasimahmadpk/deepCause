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


def normalize(var):
    nvar = (var - np.mean(var)) / np.std(var)
    return nvar


# Parameters
freq = '30min'
epochs = 50

training_length = 2880  # data for 2 month (July)
prediction_length = 144  # data for 3 days

start = 8400
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

# ************** NOrmalize the features *************
rg = normalize(org)
temp = normalize(otemp)
vpd = normalize(ovpd)
ppt = normalize(oppt)
gpp = normalize(ogpp)
reco = normalize(oreco)
intervene = np.random.normal(0.0001, 0.001, len(reco))
# ******************************************************************

"Load fluxnet 2012 data"
# nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
# nc_fid = Dataset(nc_f, 'r')
# # Dataset is the class behavior to open the file
# # and create an instance of the ncCDF4 class
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
# hour = nc_fid.variables['hour'][:].ravel().data
# day = nc_fid.variables['day'][:].ravel().data
# month = nc_fid.variables['month'][:].ravel().data
# year = nc_fid.variables['year'][:].ravel().data
# ******************************************************************

train_ds = ListDataset(
    [
         {'start': "07/01/2003 00:00:00", 'target': reco[start:train_stop],
           'dynamic_feat':[intervene[start:train_stop], rg[start:train_stop], ppt[start:train_stop], vpd[start:train_stop]]}
        # {'start': "01/01/2006 00:00:00", 'target': temp[start:train_stop], 'cat': [1],
        #  'dynamic_feat':[reco[start:train_stop], rg[start:train_stop], gpp[start:train_stop]]},
        # {'start': "01/01/2006 00:00:00", 'target': rg[start:train_stop], 'cat': [2],
        #  'dynamic_feat':[reco[start:train_stop], temp[start:train_stop], gpp[start:train_stop]]},
        # {'start': "01/01/2006 00:00:00", 'target': gpp[start:train_stop], 'cat': [3],
        #  'dynamic_feat':[reco[start:train_stop], temp[start:train_stop], rg[start:train_stop]]}
    ],
    freq=freq
)

test_ds = ListDataset(
    [
        {'start': "07/01/2003 00:00:00", 'target': reco[start:test_stop],
         'dynamic_feat':[intervene[start:train_stop], rg[start:train_stop], ppt[start:train_stop], vpd[start:train_stop]]}
        # {'start': "01/01/2006 00:00:00", 'target': temp[start:test_stop], 'cat': [1],
        #  'dynamic_feat': [reco[start:test_stop], rg[start:test_stop], gpp[start:train_stop]]},
        # {'start': "01/01/2006 00:00:00", 'target': rg[start:test_stop], 'cat': [2],
        #  'dynamic_feat': [reco[start:test_stop], temp[start:train_stop], gpp[start:train_stop]]},
        # {'start': "01/01/2006 00:00:00", 'target': gpp[start:test_stop], 'cat': [3],
        #  'dynamic_feat': [reco[start:test_stop], temp[start:train_stop], rg[start:train_stop]]}

    ],
    freq=freq
)

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length,
    freq=freq,
    num_layers=4,
    num_cells=50,
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=True,
        batch_size=96
    )
)

predictor = estimator.train(train_ds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=prediction_length,  # number of sample paths we want for evaluation
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

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
print("Intervene on VPD")
print("Performance metrices", agg_metrics)
