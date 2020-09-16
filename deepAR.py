import pandas as pd
import numpy as np
import pickle
import pathlib
from os import path
from datetime import datetime
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


def down_sample(data, win_size):
    agg_data = []
    monthly_data = []
    for i in range(len(data)):
        monthly_data.append(data[i])
        if (i % win_size) == 0:
            agg_data.append(sum(monthly_data)/win_size)
            monthly_data = []
    return agg_data


# Parameters
freq = 'H'
epochs = 50
win_size = 2

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Code updated at: ", current_time)

training_length = round((2*2880)/win_size)  # data for 2 month (July)
prediction_length = round((2*144)/win_size)  # data for 2*2 days

start = round(8400/win_size)
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

# ************** Normalize the features *************
rg = down_sample(normalize(org), win_size)
temp = down_sample(normalize(otemp), win_size)
vpd = down_sample(normalize(ovpd), win_size)
ppt = down_sample(normalize(oppt), win_size)
gpp = down_sample(normalize(ogpp), win_size)
reco = down_sample(normalize(oreco), win_size)
intervene = np.random.normal(0.0001, 0.001, len(reco))

# *****************************************************

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
           'dynamic_feat':[temp[start:train_stop], gpp[start:train_stop], rg[start:train_stop],
                           ppt[start:train_stop], vpd[start:train_stop]]}
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
         'dynamic_feat':[temp[start:test_stop], gpp[start:test_stop], rg[start:test_stop],
                         ppt[start:test_stop], vpd[start:test_stop]]}
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
    num_layers=5,
    num_cells=30,
    dropout_rate=0.1,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        hybridize=True,
        batch_size=16
    )
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
plot_forecasts(tss, forecasts, past_length=333, num_plots=4)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
print("No Intervention")
print("Performance metrices", agg_metrics)
