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
nc_f = 'datasets/ncdata/DE_Hai2000_2012.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r')   # Dataset is the class behavior to open the file                         # and create an instance of the ncCDF4 class
nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);

# Extract data from NetCDF file
vpd_f = nc_fid.variables['VPD_F'][:]  # extract/copy the data
tair_f = nc_fid.variables['TA_F'][:]
# rg_f = nc_fid.variables['Rg_f'][:]
swc1_f = nc_fid.variables['SWC_F_MDS_1'][:]
nee_f = nc_fid.variables['NEE_CUT_REF'][:]
gpp_f = nc_fid.variables['GPP_NT_VUT_REF'][:]
reco = nc_fid.variables['RECO_DT_VUT_REF'][:]
le_f = nc_fid.variables['H_F_MDS'][:]
h_f = nc_fid.variables['H_F_MDS'][:]
time = nc_fid.variables['time'][:]

"Load electricity dataset"
Edf = pd.read_csv('datasets/electricity/electricity.csv', header=0, index_col=0)
data = common.ListDataset([{
    "start": Edf.index[0],
    "target": Edf.MT_003[:44100]
}],
                          freq="5min")

trainer = Trainer(epochs=10)
estimator = deepar.DeepAREstimator(
    freq="5min", prediction_length=24, trainer=trainer)
predictor = estimator.train(training_data=data)

actual = Edf.MT_003[44100:44124].values.tolist()
prediction = next(predictor.predict(data))
eval = Evaluator()
forecast = prediction.mean

print(prediction.mean)
print("MAPE: ", eval.mape(actual, forecast))
print("MSE: ", eval.mse(actual, forecast))
print("CRPS: ", crps.calc_crps(forecast, actual))

prediction.plot(output_file='plots/graph.png')

actual, lower, upper = confidence.mean_confidence_interval(actual, 0.90)
compare_df = pd.DataFrame({'Actual': actual, 'Upper': upper, 'lower': lower, 'Forecast': forecast})

# plot the two vectors
ax = compare_df.plot(colormap='jet', marker='.', markersize=10, title='Forecasting Electricity Consumption')
ax.set_xlabel("Time frequency")
ax.set_ylabel("Electricity consumption")
fig = ax.get_figure()
fig.savefig("plots/compare.png")
