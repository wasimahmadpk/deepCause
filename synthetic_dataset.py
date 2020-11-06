import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF
import math


class SyntheticDataset:

    def __init__(self, Rg, time_steps, Tref, C, Tao, et, egpp, ereco):

        self.time_steps = time_steps
        self.Rg = Rg
        self.C = C
        self.Tao = Tao
        self.et = et
        self.egpp = egpp
        self.ereco = ereco
        self.T, self.Gpp, self.Reco = list(np.zeros(15)), list(np.zeros(15)), list(np.zeros(15))

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
            self.T.append(C.get('c1')*self.Rg[t-Tao.get('t1')])
            self.Gpp.append(C.get('c2')*self.T[t-Tao.get('t2')])
            self.Reco.append(C.get('c3')**((self.T[t-Tao.get('t3')])/10) + C.get('c4')*self.Gpp[t-Tao.get('t4')])
        return self.Rg, self.T, self.Gpp, self.Reco

    def SNR(self, s, n):

        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        SNR = Ps/Pn
        return 10*math.log(SNR, 10)        # 10*math.log(((Ps-Pn)/Pn), 10)


if __name__ == '__main__':

    # rg = np.random.normal(9, 3, 1005)

    "Load average energy consumpation data (hourly)"
    path = '/home/ahmad/PycharmProjects/deepCause/datasets/AEC Hourly/AEP_hourly.csv'  # Your filename
    energy = pd.read_csv(path, sep=';')
    print(energy.head())
    rg = energy['AEP_MW']

    time_steps, Tref = round(len(rg)), 15
    et = np.random.normal(2.75, 3, time_steps)
    egpp = np.random.normal(3, 5, time_steps)
    ereco = np.random.normal(2.5, 7, time_steps)

    C = {'c1': 0.75, 'c2': .9, 'c3': 0.99, 'c4': 0.77, 'c5': .88}          # c2:1.75, c5:1.85
    Tao = {'t1': 5, 't2': 7, 't3': 9, 't4': 11, 't5': 12, 't6': 13}
    data_obj = SyntheticDataset(rg, time_steps, Tref, C, Tao, et, egpp, ereco)
    rg, temp, gpp, reco = data_obj.generate_data()

    data = {'Rg': rg[15:], 'T': temp[15:], 'GPP': gpp[15:], 'Reco': reco[15:]}
    df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv', index_label=False, header=True)

    corr1 = np.corrcoef(et, egpp)
    corr2 = np.corrcoef(et, ereco)
    corr3 = np.corrcoef(ereco, egpp)

    print("Correlation Coefficient (et, egpp): ", corr1)
    print("Correlation Coefficient (et, ereco): ", corr2)
    print("Correlation Coefficient (ereco, egpp): ", corr3)

    print("SNR (Temperature)", data_obj.SNR(temp, et))
    print("SNR (GPP)", data_obj.SNR(gpp, egpp))
    print("SNR (Reco)", data_obj.SNR(reco, ereco))

    temp = temp + et
    gpp = gpp + egpp
    reco = reco + ereco


    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax1.plot(rg[15:1555])
    ax1.set_ylabel('Xts')

    ax2 = fig.add_subplot(412)
    ax2.plot(temp[15:1555])
    ax2.set_ylabel("Yts")

    ax3 = fig.add_subplot(413)
    ax3.plot(gpp[15:1555])
    ax3.set_ylabel("Zts")

    ax4 = fig.add_subplot(414)
    ax4.plot(reco[15:1555])
    ax4.set_ylabel("Rts")
    plt.show()