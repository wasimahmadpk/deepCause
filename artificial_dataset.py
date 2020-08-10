def artificial_dataset(Rg, time_steps, Tref, C, Tao, et, egpp, ereco):
    Rg = Rg
    # c1, c2, c3, c4, c5 = 0.3, 0.4, 0.5, 0.6, 1.5
    T, Gpp, Reco = [], [], []
    def generate_data(Rg):
        for t in time_steps:
            T.append(C.c1*T[t-Tao.t1] + C.c2*Rg(t-Tao.t2) + et)
            Gpp.append(C.c3*Rg(t-Tao.t3)*T(t-Tao.t4) + egpp)
            Reco.append(C.c4*Gpp(t-Tao.t5)*C.c5^(T(t-Tao.t6)-Tref) + ereco)
            return Rg, T, Gpp, Reco
