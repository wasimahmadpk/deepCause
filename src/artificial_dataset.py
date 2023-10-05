import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.T, self.Gpp, self.Reco = list(np.zeros(25)), list(np.zeros(25)), list(np.zeros(25))

    def generate_data(self):

        for t in range(25, self.time_steps-25):

            self.T.append(C.get('c1')*self.T[t-Tao.get('t1')] + C.get('c2')*self.Rg[t-Tao.get('t2')] + et[t])
            self.Gpp.append(C.get('c3')*self.Rg[t-Tao.get('t3')]*self.T[t-Tao.get('t4')] + egpp[t])
            self.Reco.append(C.get('c4')*self.Gpp[t-Tao.get('t5')]*C.get('c5')**((self.T[t-Tao.get('t6')]-Tref)/10) + ereco[t])
        return self.Rg, self.T, self.Gpp, self.Reco

    def SNR(self, s, n):

        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        SNR = Ps/Pn
        return 10*math.log(SNR, 10)        # 10*math.log(((Ps-Pn)/Pn), 10)


if __name__ == '__main__':

    "Load fluxnet 2015 data for grassland IT-Mbo site"
    fluxnet = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/fluxnet2015/FLX_IT-MBo_FLUXNET2015_SUBSET_2003-2013_1-4/FLX_IT-MBo_FLUXNET2015_SUBSET_HH_2003-2013_1-4.csv")
    rg = fluxnet['SW_IN_F']
    # nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
    # nc_fid = Dataset(nc_f, 'r')
    # nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);
    #
    # # Extract data from NetCDF file
    # rg = nc_fid.variables['SW_IN_F_MDS'][:].ravel().data
    print("Rg length: ", len(rg))
    nrgs = (rg - np.mean(rg)) / np.std(rg)
    # t = np.linspace(0, 10000, 10000)
    # season = np.cos(2*np.pi*150*t)
    # nrg = np.random.normal(0, 0.50, 10000)
    # nrgs = nrg + np.abs(season)

    time_steps, Tref = len(nrgs), 15
    et = np.random.normal(0, 0.10, time_steps)
    egpp = np.random.normal(0, 0.15, time_steps)
    ereco = np.random.normal(0, 0.05, time_steps)

    C = {'c1': 0.25, 'c2': 0.5, 'c3': 0.10, 'c4': 0.75, 'c5': 0.2}          # c2:1.75, c5:1.85
    Tao = {'t1': 5, 't2': 10, 't3': 15, 't4': 20, 't5': 15, 't6': 10}
    data_obj = ArtificialDataset(nrgs, time_steps, Tref, C, Tao, et, egpp, ereco)
    rg, tair, gpp, reco = data_obj.generate_data()

    data = {'Rg': rg[25:], 'T': tair, 'GPP': gpp, 'Reco': reco}
    df = pd.DataFrame(data, columns=['Rg', 'T', 'GPP', 'Reco'])
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data.csv', index_label=False, header=True)

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
    ax1.plot(reco[0:30000])
    ax1.set_ylabel('Reco')

    ax2 = fig.add_subplot(412)
    ax2.plot(tair[0:30000])
    ax2.set_ylabel("Temp")

    ax3 = fig.add_subplot(413)
    ax3.plot(gpp[0:30000])
    ax3.set_ylabel("GPP")

    ax4 = fig.add_subplot(414)
    ax4.plot(rg[0:30000])
    ax4.set_ylabel("Rg")
    plt.show()