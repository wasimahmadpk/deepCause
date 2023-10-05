import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 12
test = 'ssr_chi2test'


def normalize(var):
    nvar = (np.array(var) - np.mean(var)) / np.std(var)
    return nvar


def down_sample(data, win_size):
    agg_data = []
    monthly_data = []
    for i in range(len(data)):
        monthly_data.append(data[i])
        if (i % win_size) == 0:
            agg_data.append(sum(monthly_data) / win_size)
            monthly_data = []
    return agg_data


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


if __name__ == '__main__':


    "Load average energy consumpation data (hourly)"
    win_size = 24
    syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv", sep=',')
    Xts = down_sample(np.array(syndata['Rg']), win_size)
    Yts = down_sample(np.array(syndata['T']), win_size)
    Zts = down_sample(np.array(syndata['GPP']), win_size)
    # dts = down_sample(np.array(syndata['Reco']), win_size)

    data = pd.DataFrame({'X': Xts, 'Y': Yts, 'Z': Zts}, columns=['X', 'Y', 'Z'])
    print(data.head())

    # # Load DWD dataset
    # data = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/DWD/DWD_labels.csv", sep=';')
    # print(data.head())

    # # Load fluxnet 2015 data
    # col_list = ['SW_IN_F', 'TA_F', 'VPD_F', 'P_F', 'GPP_DT_VUT_50', 'RECO_NT_VUT_50']
    # data = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv", usecols=col_list)
    # print(data.head())

    causal_df = grangers_causation_matrix(data, variables=data.columns)
    causal_df[causal_df > 0.05] = 'No'
    print(causal_df)