import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF
import math


class SyntheticDataset:

    def __init__(self, root, time_steps, Tref, C, Tao, ey, ez, er):

        self.time_steps = time_steps

        self.root = root
        self.C = C
        self.Tao = Tao
        self.ey = ey
        self.ez = ez
        self.er = er
        self.X1 = list(np.zeros(15))
        self.X2 = list(np.zeros(15))
        self.X3 = list(np.zeros(15))
        self.X4 = list(np.zeros(15))
        self.X5 = list(np.zeros(15))
        self.X6 = list(np.zeros(15))
        self.X7 = list(np.zeros(15))
        self.X8 = list(np.zeros(15))
        self.X9 = list(np.zeros(15))
        self.X10 = list(np.zeros(15))

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

        # self.Yts.append(C.get('c1') * self.Xts[t - Tao.get('t1')] + ey[t])
        # self.Zts.append(C.get('c2') ** ((self.Xts[t - Tao.get('t2')]) / 2 + ez[t]))
        # self.Rts.append(C.get('c3') * self.Yts[t - Tao.get('t3')] + C.get('c4') * self.Zts[t - Tao.get('t4')] + er[t])

        for t in range(15, self.time_steps):

            self.X1.append(C.get('c1')*self.root[t-Tao.get('t2')] + ey[t])
            self.X2.append(C.get('c2')*self.X1[t-Tao.get('t3')] + ez[t])
            
        return self.X1, self.X2

    def SNR(self, s, n):

        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        SNR = Ps/Pn
        return 10*math.log(SNR, 10)        # 10*math.log(((Ps-Pn)/Pn), 10)


if __name__ == '__main__':

    root = np.random.normal(0, 1, 10015)
    t = np.linspace(0, 20, 10015)
    season = np.cos(2 * np.pi * 10 * t)
    roots = root + np.abs(season)
    stateone = np.random.normal(1.5, 5, 1000)
    anom_idxone = random.sample(list(range(10015)), 1000)

    # statetwo = np.random.normal(1.25, 0.15, 2000)
    anom_idxtwo = random.sample(list(range(10015)), 1000)

    for s in range(len(stateone)):
        roots[anom_idxone[s]] = stateone[s]
        # xtss[anom_idxtwo[s]] = statetwo[s]

    time_steps, Tref = round(len(root)), 15
    ey = np.random.normal(0.5, 0.35, time_steps)
    ez = np.random.normal(0.3, 0.45, time_steps)
    er = np.random.normal(0.1, 0.10, time_steps)

    C = {'c1': 0.50, 'c2': 0.95, 'c3': 0.75, 'c4': 0.90, 'c5': 0.80}          # c2:1.75, c5:1.85
    Tao = {'t1': 1, 't2': 3, 't3': 5, 't4': 4, 't5': 5, 't6': 6}
    data_obj = SyntheticDataset(roots, time_steps, Tref, C, Tao, ey, ez, er)
    X1, X2 = data_obj.generate_data()

    corr1 = np.corrcoef(ey, ez)

    print("Correlation Coefficient (ey, ez): ", corr1)
    # print("SNR (Temperature)", data_obj.SNR(Yts, ez))

    data = {'Z1': X1[15:], 'Z2': X2[15:]}
    df = pd.DataFrame(data, columns=['Z1', 'Z2'])
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data_lnos.csv', index_label=False, header=True)
    print(df.head(15000))

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
    ax1.plot(X1[15:150])
    ax1.set_ylabel('X1')

    ax2 = fig.add_subplot(412)
    ax2.plot(X2[15:150])
    ax2.set_ylabel("X2")

    # ax3 = fig.add_subplot(413)
    # ax3.plot(X8[15:150])
    # ax3.set_ylabel("X3")

    # ax4 = fig.add_subplot(414)
    # ax4.plot(X10[15:150])
    # ax4.set_ylabel("X4")
    plt.show()