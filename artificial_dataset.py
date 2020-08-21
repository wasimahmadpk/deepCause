import numpy as np

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
        self.T, self.Gpp, self.Reco = [1, 2, 4, 3, 3, 4, 5, 6], [1, 2, 3, 3,9, 4, 5, 6], [1, 2, 3, 3, 3, 4, 5, 6]

    def generate_data(self):

        for t in range(self.time_steps-6):
            print(t)
            self.T.append(C.get('c1')*self.T[(t+6)-Tao.get('t1')] + C.get('c2')*Rg[(t+6)-Tao.get('t2')] + et)
            self.Gpp.append(C.get('c3')*Rg[(t+6)-Tao.get('t3')]*self.T[(t+6)-Tao.get('t4')] + egpp)
            self.Reco.append(C.get('c4')*self.Gpp[(t+6)-Tao.get('t5')]*C.get('c5')**(self.T[(t+6)-Tao.get('t6')]-Tref) + ereco)
        return Rg, self.T, self.Gpp, self.Reco

if __name__ == '__main__':
    # c1, c2, c3, c4, c5 = 0.3, 0.4, 0.5, 0.6, 1.5
    time_steps, Tref = 50, 29
    Rg = np.ones(time_steps+2).tolist()
    et, egpp, ereco = 1, 2, 3
    C = {'c1': 0.1, 'c2': 0.2, 'c3': 0.3, 'c4': 0.4, 'c5': 1}
    Tao = {'t1': 1, 't2': 2, 't3': 3, 't4': 4, 't5': 5, 't6': 6}
    data_obj = artificial_dataset(Rg, time_steps, Tref, C, Tao, et, egpp, ereco)
    rg, tair, gpp, reco = data_obj.generate_data()