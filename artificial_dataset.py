import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import netCDF

class artificial_dataset():

    def __init__(self, Rg, time_steps, Tref, C, Tao, et, egpp, ereco):

        self.time_steps = time_steps
        self.Rg = Rg
        self.Tref = Tref
        self.C = C
        self.Tao = Tao
        self.et = et
        self.egpp = egpp
        self.ereco = ereco
        # c1, c2, c3, c4, c5 = 0.3, 0.4, 0.5, 0.6, 1.5
        self.T, self.Gpp, self.Reco = rg[0:10], rg[0:10], rg[0:10]

    def generate_data(self):

        for t in range(self.time_steps-6):
            print(t)
            self.T.append(C.get('c1')*self.T[(t+6)-Tao.get('t1')] + C.get('c2')*Rg[(t+6)-Tao.get('t2')] + et)
            self.Gpp.append(C.get('c3')*Rg[(t+6)-Tao.get('t3')]*self.T[(t+6)-Tao.get('t4')] + egpp)
            self.Reco.append(C.get('c4')*self.Gpp[(t+6)-Tao.get('t5')]*C.get('c5')**(self.T[(t+6)-Tao.get('t6')]-Tref) + ereco)
        return Rg, self.T, self.Gpp, self.Reco


if __name__ == '__main__':

    "Load fluxnet data"
    nc_f = '/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/DE-Hai.2000.2006.hourly.nc'  # Your filename
    nc_fid = Dataset(nc_f, 'r')
    nc_attrs, nc_dims, nc_vars = netCDF.ncdump(nc_fid);

    # Extract data from NetCDF file
    rg = list(nc_fid.variables['Rg_f'][:].ravel().data)

    time_steps, Tref = len(rg), 15
    Rg = rg
    et, egpp, ereco = 1, 2, 5
    C = {'c1': 0.2, 'c2': 0.4, 'c3': 0.6, 'c4': 0.4, 'c5': 1.5}
    Tao = {'t1': 1, 't2': 3, 't3': 5, 't4': 7, 't5': 9, 't6': 11}
    data_obj = artificial_dataset(Rg, time_steps, Tref, C, Tao, et, egpp, ereco)
    rg, tair, gpp, reco = data_obj.generate_data()