from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat

# this file determines the angle gamma of a given state sin(theta)|H>|alpha>+(e^i*gamma)cos(theta)|V>|alpha_perp>
# from the density matrix in the same manner the gamma_determination file does


if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    DATE = "07032025" #please update
    STATE = "ha_negpi_3_vd"
    chis = np.linspace(0.001, np.pi/2, 6)
    gammas = []
    
    fileName = f"gamma_data_from_tomo"
    directory = f'ria_ha_negpi_3_vd_trial1'

    for chi in chis:
        chi_act = np.rad2deg(chi)
        #bases for hdva state: 'DR', 'AR', 'DL', 'AL', 'RR', 'LL', 'RL', 'LR'
        #bases for hrvl state: 'DD', 'DA', 'AD', 'AA', 'RD', 'RA', 'LD', 'LA'
        datas = Manager.load_data(f'{directory}/tomo_data_ha_negpi_3_vd_{chi_act}_{DATE}_1.csv')

        datas = datas.set_index("note")

        # for hdva state: 
        gamma = unp.arctan2(-(datas.at['DR', 'C4']+datas.at['AL', 'C4']-datas.at['DL', 'C4']-datas.at['AR', 'C4']), (datas.at['RR', 'C4']+datas.at['LL', 'C4']-datas.at['LR', 'C4']-datas.at['RL', 'C4']))
        gammas.append(gamma)

    df = pd.DataFrame({'chi': chis, 'gamma': gammas})

    # for hrvl state:
    # datas['gamma'] = unp.arctan2((datas['DD']+datas['AA']-datas['DA']-datas['AD']), -(datas['RD']+datas['LA']-datas['RA']-datas['LD']))
    pd.DataFrame(df).to_csv(f'{directory}/{fileName}.csv')

    df = Manager.load_data(f'{directory}/{fileName}.csv')

    chis, gammas = df['chi'], df['gamma'].apply(unc.ufloat_fromstr)

    params = analysis.fit('line', chis, gammas)
    analysis.plot_func('line', params, chis, color='blue')
    analysis.plot_errorbar(chis, gammas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('Chi')
    plt.ylabel('Gamma (rad)')
    plt.savefig(f'{directory}/{fileName}.png', dpi=600)
    plt.show()

