from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from itertools import islice
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from netCDF4 import Dataset
import netCDF

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
#******************************************************************



#df = pd.DataFrame(reco, header=0, index_col=0)

#df[:200].plot(figsize=(12, 5), linewidth=2)
#plt.grid()
#plt.legend(["observations"])
#plt.show()

training_data = ListDataset(
    [{"start": 0, "target": tair_f[0:40000]}],
    freq = "1D"
)

estimator = DeepAREstimator(freq="1D", 
                            prediction_length=500, 
                            trainer=Trainer(epochs=10))

predictor = estimator.train(training_data=training_data)


test_data = ListDataset(
    [
        {"start": 0, "target": tair_f[0:40500]},
        {"start": 0, "target": taif_f[0:41000]},
        {"start": 0, "target": tair_f[0:41500]}
    ],
    freq = "1D"
)


def plot_forecasts(tss, forecasts, past_length, num_plots):
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.title("Forecasting Air Temperature")
        plt.xlabel("Timestamp")
        plt.ylabel("T_Air")
        plt.show()

forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_samples=500)
forecasts = list(forecast_it)
tss = list(ts_it)
plot_forecasts(tss, forecasts, past_length=3000, num_plots=3)


evaluator = Evaluator(quantiles=[0.5], seasonality=2006)

agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
print("Performance metrices", agg_metrics)
