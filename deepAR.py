from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from netCDF4 import Dataset
import confidence
import numpy as np
import crps
import pandas as pd
import matplotlib.pyplot as plt
import netCDF


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

# Edf = pd.read_csv('/home/ahmad/PycharmProjects/deepCause/datasets/electricity/electricity.csv', header=0, index_col=0)
data = common.ListDataset([{
    "start": 0,
    "target": reco[:44100]
}],
                          freq="5min")

trainer = Trainer(epochs=10)
estimator = deepar.DeepAREstimator(
    freq="5min", prediction_length=111, trainer=trainer)
predictor = estimator.train(training_data=data)

actual = reco[44100:44211]
prediction = next(predictor.predict(data))
eval = Evaluator()
forecast = prediction.mean

print(prediction.mean)
print("MAPE: ", eval.mape(actual, forecast))
print("MSE: ", eval.mse(actual, forecast))
print("CRPS: ", crps.calc_crps(forecast, actual))

prediction.plot(output_file='/home/ahmad/PycharmProjects/deepCause/plots/graph.png')

actual, lower, upper = confidence.mean_confidence_interval(actual, 0.90)
compare_df = pd.DataFrame({'Actual': actual, 'Upper': upper, 'lower': lower, 'Forecast': forecast})
print("LIne 58")
# plot the two vectors
ax = compare_df.plot(colormap='jet', marker='.', markersize=10, title='Forecasting Electricity Consumption')
ax.set_xlabel("Time frequency")
ax.set_ylabel("Electricity consumption")
fig = ax.get_figure()
fig.savefig("/home/ahmad/PycharmProjects/deepCause/plots/compare.png")
