import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.datetools import dates_from_str

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

# mdata = sm.datasets.macrodata.load_pandas().data
# # prepare the dates index
# dates = mdata[['year', 'quarter']].astype(int).astype(str)
# quarterly = dates["year"] + "Q" + dates["quarter"]
# quarterly = dates_from_str(quarterly)
# mdata = mdata[['realgdp','realcons','realinv']]
# mdata.index = pandas.DatetimeIndex(quarterly)
# data = np.log(mdata).diff().dropna()
win_size = 1

"Load average energy consumpation data (hourly)"
syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv", sep=',')
ats = down_sample(np.array(syndata['Rg']), win_size)
bts = down_sample(np.array(syndata['T']), win_size)
cts = down_sample(np.array(syndata['GPP']), win_size)
dts = down_sample(np.array(syndata['Reco']), win_size)

col_list = ['Rg', 'T', 'GPP', 'Reco']
data = pd.DataFrame({'Rg': ats, 'T': bts, 'GPP': cts, 'Reco': dts}, columns=col_list)
print(data.head())

# "Load fluxnet data"
# col_list = ['SW_IN_F', 'TA_F', 'VPD_F', 'P_F', 'GPP_DT_VUT_50', 'RECO_NT_VUT_50']
# data = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv", usecols=col_list)
# print(data.head())

# make a VAR model
model = VAR(data)
results = model.fit(2)
# print(results.summary())
for i in range(len(col_list)):
    for j in range(len(col_list)):
        print(results.test_causality(col_list[j], col_list[i], kind='f'))