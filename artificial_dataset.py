import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF
import math


class ArtificialDataset:

    def __init__(self, Rg, time_steps, Tref, C, Tao, et, egpp, ereco):

        self.time_steps = time_steps
        self.Rg = Rg
        self.Tref = Tref
        self.C = C
        self.Tao = Tao
        self.et = et
        self.egpp = egpp
        self.ereco = ereco
        self.T, self.Gpp, self.Reco = list(np.zeros(10)), list(np.zeros(10)), list(np.zeros(10))

    def generate_data(self):

        for t in range(10, self.time_steps-10):
            self.T.append(C.get('c1')*self.T[t-Tao.get('t1')] + C.get('c2')*self.Rg[t-Tao.get('t2')] + et[t])
            self.Gpp.append(C.get('c3')*self.Rg[t-Tao.get('t3')]*self.T[t-Tao.get('t4')] + egpp[t])
            self.Reco.append(C.get('c4')*self.Gpp[t-Tao.get('t5')]*C.get('c5')**((self.T[t-Tao.get('t6')]-Tref)/10) + ereco[t])
        return self.Rg, self.T, self.Gpp, self.Reco

    def SNR(self, s, n):

        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        return 10*math.log(((Ps-Pn)/Pn), 10)


if __name__ == '__main__':

    "Load fluxnet data"
    nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
    nc_fid = Dataset(nc_f, 'r')
    nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);

    # Extract data from NetCDF file
    rg = nc_fid.variables['Rg_f'][:].ravel().data
    nrg = (rg - np.mean(rg))/np.std(rg)

    time_steps, Tref = len(rg), 15
    et = np.random.normal(0.0001, 0.001, time_steps)
    egpp = np.random.normal(0.00015, 0.0015, time_steps)
    ereco = np.random.normal(0.00025, 0.0005, time_steps)

    C = {'c1': 0.25, 'c2': 0.5, 'c3': 0.75, 'c4': 1, 'c5': 2}
    Tao = {'t1': 1, 't2': 2, 't3': 3, 't4': 4, 't5': 5, 't6': 6}
    data_obj = ArtificialDataset(nrg, time_steps, Tref, C, Tao, et, egpp, ereco)
    rg, tair, gpp, reco = data_obj.generate_data()

    corr1 = np.corrcoef(et, egpp)
    corr2 = np.corrcoef(et, ereco)
    corr3 = np.corrcoef(ereco, egpp)

    print("Correlation Coefficient (et, egpp): ", corr1)
    print("Correlation Coefficient (et, ereco): ", corr2)
    print("Correlation Coefficient (ereco, egpp): ", corr3)

    print("SNR (Temperature)", data_obj.SNR(tair, et))
    print("SNR (GPP)", data_obj.SNR(gpp, egpp))
    print("SNR (Reco)", data_obj.SNR(reco, ereco))

    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax1.plot(reco[0:55555])
    ax1.set_ylabel('Reco')

    ax2 = fig.add_subplot(412)
    ax2.plot(tair[0:55555])
    ax2.set_ylabel("Temp")

    ax3 = fig.add_subplot(413)
    ax3.plot(gpp[0:55555])
    ax3.set_ylabel("GPP")

    ax4 = fig.add_subplot(414)
    ax4.plot(rg[0:55555])
    ax4.set_ylabel("Rg")