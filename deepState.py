from gluonts.dataset import common
from gluonts.model import deepstate
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

data = common.ListDataset(
    [{"start": 0, "target": tair_f[:40050]}],
    freq="60min")

trainer = Trainer(epochs=2)
estimator = deepstate.DeepStateEstimator(
    freq="60min", prediction_length=24, trainer=trainer)
predictor = estimator.train(training_data=data)

actual = tair_f[40050:40074]
prediction = next(predictor.predict(data))
eval = Evaluator()
forecast = prediction.mean

print(prediction.mean)
print("MAPE: ", eval.mape(actual, forecast))
print("MSE: ", eval.mse(actual, forecast))
print("CRPS: ", crps.calc_crps(forecast, actual))

plt.title('Forecasting Air temperature-95% PI')
plt.ylabel('T_Air')
plt.xlabel('Hour of the day')
prediction.plot(prediction_intervals=(0, 95), color='g', output_file='/home/ahmad/PycharmProjects/deepCause/plots/graph.png')
plt.plot(actual)
actual, lower, upper = confidence.mean_confidence_interval(actual, 0.95)
compare_df = pd.DataFrame({'Actual': actual, 'Upper bound': upper, 'Lower bound': lower, 'Forecast': forecast})

# plot the two vectors
ax = compare_df.plot(colormap='jet', marker='.', markersize=10, title='Forecasting Air Temperature-95% CI')
ax.set_xlabel("Hour of the day")
ax.set_ylabel("T_Air")
plt.show()
fig = ax.get_figure()
fig.savefig("/home/ahmad/PycharmProjects/deepCause/plots/compare.png")
