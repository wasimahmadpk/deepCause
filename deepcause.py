import math
import netCDF
import pickle
import random
import pathlib
import numpy as np
import numpy as np
import mxnet as mx
import pandas as pd
from os import path
from math import sqrt
from netCDF4 import Dataset
from itertools import islice
from datetime import datetime
import matplotlib.pyplot as plt
from knockoffs import Knockoffs
from riverdata import RiverData
from scipy.special import stdtr
from model_test import modelTest
from gluonts.trainer import Trainer
from sklearn.metrics import f1_score
from gluonts.evaluation import Evaluator
from counterfactuals import Counterfactuals
from sklearn.metrics import mean_squared_error
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepar._network import DeepARTrainingNetwork
from gluonts.evaluation.backtest import make_evaluation_predictions
from scipy.stats import ttest_ind, ttest_ind_from_stats, ttest_1samp
from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

np.random.seed(1)
mx.random.seed(2)


def deepCause(odata, pdata, knockoffs, model, params):

    filename = pathlib.Path(model)
    if not filename.exists():
        print("Training forecasting model....")
        predictor = estimator.train(train_ds)
        # save the model to disk
        pickle.dump(predictor, open(filename, 'wb'))

    conf_mat = []
    conf_mat_mean = []
    conf_mat_indist = []
    conf_mat_outdist = []

    for i in range(len(odata)):

        int_var = odata[i]
        int_var_name = "X_" + str(i + 1) + ""
        causal_decision = []
        mean_cause = []
        indist_cause = []
        outdist_cause = []
        # knockoff_sample = np.array(knockoffs[333:params.get('ts_len') + 333, i])
        knockoff_sample = random.sample(list(knockoffs[:, i]), params.get('ts_len'))
        for j in range(len(odata)):

            pred_var = odata[j]
            pred_var_name = "X_" + str(j + 1) + ""

            # print("Deep Knockoffs: \n", counterfactuals)

            # plt.plot(np.arange(0, len(counterfactuals)), target[: len(counterfactuals)], counterfactuals)
            # plt.show()

            # corr = np.corrcoef(knockoff_sample, odata[j][0: len(knockoff_sample)])
            # print(f"Correlation Coefficient (Variable, Counterfactual): {corr}")

            mean = [np.zeros(len(knockoff_sample)) + np.mean(odata[ts]) for ts in range(len(odata))]
            outdist = np.random.normal(0, 0.001, params.get('ts_len'))

            interventionlist = [mean[:][0:params.get('ts_len')], knockoff_sample, outdist]
            heuristic_itn_types = ['Mean', 'In-dist', 'Out-dist']
            css_list = []
            css_list_new = []
            css_score_new = []
            mselol = []
            mapelol = []
            acelol = []
            mselolint = []
            mapelolint = []

            for m in range(len(interventionlist)):  # apply all interventions len(interventionlist)

                if m > 0:
                    intervene = interventionlist[m]
                else:
                    intervene = interventionlist[m][j]

                mselist = []
                mselistint = []
                acelist = []
                mapelist = []
                mapelistint = []
                css_score = []
                diff = []
                for iter, ls in enumerate(zip(*pdata)):
                    time_series = list(ls).copy()
                    test_ds = ListDataset(
                        [
                            {'start': "01/01/1961 00:00:00",
                             'target': time_series,
                             'cat': [iter]
                             }
                        ],
                        freq=params.get('freq'),
                        one_dim_target=False
                    )
                    # rg[0:-50] + list(intervene[-50:])
                    int_time_series = list(ls).copy()
                    int_time_series[i] = intervene

                    test_dsint = ListDataset(
                        [
                            {'start': "01/01/1961 00:00:00",
                             'target': int_time_series,
                             'cat': [iter]
                             }
                        ],
                        freq=params.get('freq'),
                        one_dim_target=False
                    )

                    mse, mape, ypred = modelTest(model, test_ds, params.get('num_samples'), time_series[j], j,
                                                 params.get('pred_len'), iter, False, 0)
                    mseint, mapeint, ypredint = modelTest(model, test_dsint, params.get('num_samples'), time_series[j], j,
                                                          params.get('pred_len'), iter, True, m)

                    mselist.append(mse)
                    mapelist.append(mape)
                    mselistint.append(mseint)
                    mapelistint.append(mapeint)
                    # acelist.append(avg_causal_effect(np.array(ypred), np.array(ypredint)))

                    target_before = np.array(time_series[j][:params.get('ts_len')] + ypred)
                    target_after = np.array(time_series[j][:params.get('ts_len')] + ypredint)

                    diff.append(abs(target_after - target_before))

                mse = np.mean(mselist)
                mape = np.mean(mapelist)
                mselol.append(mselist)
                mapelol.append(mapelist)
                # acelol.append(acelist)
                # print(f"MSE: {mselist}, MAPE: {mape}%")
                # print(f"ACE: {acelist}")

                mse = np.mean(mselistint)
                mape = np.mean(mapelistint)

                mselolint.append(mselistint)
                mapelolint.append(mapelistint)
                # print(f"MSE: {mselistint}, MAPE: {mape}%")
                # avg_diff = np.mean(diff, axis=0)
                # plt.plot(avg_diff)
                # plt.show()

                for k in range(len(mselist)):
                    css_score.append(np.log(mapelistint[k] / mapelist[k]))
                    # css_score.append(np.log(mselistint[k] / mselist[k]))
                    # css_score.append(abs(mselistint[k] - mselist[k]))

                # css_score = [abs(x) if x < 0 else x for x in css_score]
                css_list.append(css_score)
                # plt.hist(css_score)
                # plt.show()
                # print("CSS: ", css_score)
                # print("Before Intervention: ", mselist)
                # print("After Intervention: ", mselistint)

            # print(f"MSE(Mean): {list(np.mean(mselol, axis=0))}")
            print(f"Time series: X_{i+1} ----------> X_{j+1}")
            print("-------------------------------------------------------------------------")
            for z in range(len(heuristic_itn_types)):
                print(f"Average Causal Strength using {heuristic_itn_types[z]} Intervention: {np.mean(css_list[z])}")
                # print(f"Average Causal Strength using {heuristic_itn_types[z]} Intervention: {np.mean(np.array(mselolint[z]) - np.array(mselol[z]))}")
                # print("CSS: ", css_score)
                # t, p = ttest_ind(np.array(mselolint[z]), np.array(mselol[z]), equal_var=False)
                # t, p = ttest_1samp(np.array(mselolint[z]) - np.array(mselol[z]), popmean=0.0)
                t, p = ttest_1samp(css_list[z], popmean=0.0)
                # t, p = ttest_ind(mselolint[z], mselol[z], equal_var=False)
                # plt.hist(mselolint[z])
                # plt.hist(mselol[z])
                # plt.show()
                print(f'Test statistic: {t}, p-value: {p}')
                if p < 0.075:
                    print("Null hypothesis is rejected")
                    if i != j:
                        causal_decision.append(1)
                else:
                    print("Fail to reject null hypothesis")
                    if i != j:
                        causal_decision.append(0)
            if i != j:
                mean_cause.append(causal_decision[0])
                indist_cause.append(causal_decision[1])
                outdist_cause.append(causal_decision[2])
                causal_decision = []
                # print(f"Average Causal Impact using {heuristic_itn_types[z]} Intervention: {np.mean(acelol[z])}")
                # t, p = ttest_1samp(acelol[z], 0)
                # print(f'Test statistic: {t}, p-value: {p}')
                # if p < 0.05:
                #     print("Null hypothesis is rejected")
                # else:
                #     print("Fail to reject null hypothesis")
            print("-------------******--------------*******-------------*******-------------")


        conf_mat_mean = conf_mat_mean + mean_cause
        conf_mat_indist = conf_mat_indist + indist_cause
        conf_mat_outdist = conf_mat_outdist + outdist_cause
        mean_cause, indist_cause, outdist_cause = [], [], []


    conf_mat.append(conf_mat_mean)
    conf_mat.append(conf_mat_indist)
    conf_mat.append(conf_mat_outdist)

    print("Confusion Matrix:", conf_mat)
    # true_conf_mat = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    true_conf_mat = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for ss in range(len(conf_mat)):

        fscore = round(f1_score(true_conf_mat, conf_mat[ss], average='micro'), 2)
        print(f"F-score {heuristic_itn_types[ss]} Intervention: {fscore}")




