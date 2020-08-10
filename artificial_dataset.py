def artificial_dataset(Rg):
    Rg = Rg

    def generate_data(Rg):
        T = Rg * Rg
        Gpp = Rg.T
        Reco = Gpp * Rg * T
        return Rg, T, Gpp, Reco
