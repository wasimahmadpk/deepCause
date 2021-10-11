import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF
import math


class SyntheticDataset:

    def __init__(self, xts, time_steps, Tref, C, Tao, ey, ez, er):

        self.time_steps = time_steps
        self.Xts = xts
        self.C = C
        self.Tao = Tao
        self.ey = ey
        self.ez = ez
        self.er = er
        self.Yts = list(np.zeros(15))
        self.Zts = list(np.zeros(15))
        self.Rts = list(np.zeros(15))

    def normalize(self, var):
        nvar = (np.array(var) - np.mean(var)) / np.std(var)
        return nvar

    def down_sample(self, data, win_size):
        agg_data = []
        monthly_data = []
        for i in range(len(data)):
            monthly_data.append(data[i])
            if (i % win_size) == 0:
                agg_data.append(sum(monthly_data) / win_size)
                monthly_data = []
        return agg_data

    def generate_data(self):

        for t in range(15, self.time_steps):
            self.Yts.append(C.get('c1')*self.Xts[t-Tao.get('t1')] + ey[t])
            self.Zts.append(C.get('c2')**((self.Xts[t-Tao.get('t2')])/2 + ez[t]))
            self.Rts.append(C.get('c3')*self.Yts[t-Tao.get('t3')] + C.get('c4')*self.Zts[t-Tao.get('t4')] + er[t])
        return self.Xts, self.Yts, self.Zts, self.Rts

    def SNR(self, s, n):

        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        SNR = Ps/Pn
        return 10*math.log(SNR, 10)        # 10*math.log(((Ps-Pn)/Pn), 10)


if __name__ == '__main__':

    xts = np.random.normal(0, 0.5, 30015)
    t = np.linspace(0, 20, 30015)
    season = np.cos(2 * np.pi * 10 * t)
    xtss = xts + np.abs(season)
    stateone = np.random.normal(0.3, 0.3, 1000)
    anom_idxone = random.sample(list(range(30015)), 1000)

    # statetwo = np.random.normal(1.25, 0.15, 2000)
    anom_idxtwo = random.sample(list(range(30015)), 1000)

    for s in range(len(stateone)):
        xtss[anom_idxone[s]] = stateone[s]
        # xtss[anom_idxtwo[s]] = statetwo[s]

    time_steps, Tref = round(len(xts)), 15
    ey = np.random.normal(0, 0.10, time_steps)
    ez = np.random.normal(0, 0.11, time_steps)
    er = np.random.normal(0, 0.15, time_steps)

    C = {'c1': 0.95, 'c2': 0.75, 'c3': 1.25, 'c4': 0.90, 'c5': 0.80}          # c2:1.75, c5:1.85
    Tao = {'t1': 2, 't2': 3, 't3': 4, 't4': 5, 't5': 6, 't6': 7}
    data_obj = SyntheticDataset(xtss, time_steps, Tref, C, Tao, ey, ez, er)
    Xts, Yts, Zts, Rts = data_obj.generate_data()

    corr1 = np.corrcoef(ey, ez)

    print("Correlation Coefficient (ey, ez): ", corr1)

    # print("SNR (Temperature)", data_obj.SNR(Yts, ez))

    data = {'Xts': Xts[15:], 'Yts': Yts[15:], 'Zts': Zts[15:], 'Rts': Rts[15:]}
    df = pd.DataFrame(data, columns=['Xts', 'Yts', 'Zts', 'Rts'])
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv', index_label=False, header=True)

    # data_obj = SyntheticDataset(xtss, time_steps, Tref, C, Tao, ey, ez, er)
    # Xts, Yts, Zts, Rts = data_obj.generate_data()
    #
    # corr1 = np.corrcoef(ey, ez)
    #
    # print("Correlation Coefficient (ey, ez): ", corr1)
    #
    # # print("SNR (Temperature)", data_obj.SNR(Yts, ez))
    #
    # data = {'Xts': Xts[15:], 'Yts': Yts[15:], 'Zts': Zts[15:], 'Rts': Rts[15:]}
    # df = pd.DataFrame(data, columns=['Xts', 'Yts', 'Zts', 'Rts'])
    # df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data_seasonal.csv', index_label=False,
    #           header=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax1.plot(Xts[15:250])
    ax1.set_ylabel('Xts')

    ax2 = fig.add_subplot(412)
    ax2.plot(Yts[15:250])
    ax2.set_ylabel("Yts")

    ax3 = fig.add_subplot(413)
    ax3.plot(Zts[15:250])
    ax3.set_ylabel("Zts")

    # ax3 = fig.add_subplot(414)
    # ax3.plot(Rts[15:100])
    # ax3.set_ylabel("Rts")
    plt.show()