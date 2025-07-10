from lab_framework import Manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat

# this file determines the angle chi of a given state sin(chi/2)|H>|alpha>+(e^i*gamma)cos(chi/2)|V>|alpha_perp>
# from the density matrix


if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    DATE = "07032025" #please update
    STATE = "ha_negpi_3_vd"
    chis_theo = np.linspace(0.001, np.pi/2, 6)
    chis_actual = []
    
    fileName = f"chi_data_from_tomo"
    directory = f'ria_ha_negpi_3_vd_trial1'

    for chi in chis_theo:
        chi_theo = np.rad2deg(chi)
        datas = Manager.load_data(f'{directory}/tomo_data_ha_negpi_3_vd_{chi_theo}_{DATE}_1.csv')

        datas = datas.set_index("note")

        # for hdva state: 
        chi_act = 2*unp.arctan(unp.sqrt(datas.at['VA', 'C4']/datas.at['HD', 'C4']))
        chis_actual.append(chi_act)

    df = pd.DataFrame({'Chi Theory': chis_theo, 'Chi Actual': chis_actual})

    pd.DataFrame(df).to_csv(f'{directory}/{fileName}.csv')

