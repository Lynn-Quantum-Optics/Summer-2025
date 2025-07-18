import analysis2_delete as analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat

# this file determines the angle gamma of a given state sin(theta)|H>|alpha>+(e^i*gamma)cos(theta)|V>|alpha_perp>
# from the density matrix in the same manner the gamma_determination file does

def reformat_float_to_ufloat(df_:pd.DataFrame) -> pd.DataFrame:
        ''' Reformat a dataframe with floats to contain ufloats where applicable.

        Each column "X" that has a corresponding column "X_SEM" will be collapsed to the single column "X" containing ufloats.

        Parameters
        ----------
        df_ : pd.DataFrame
            The dataframe to reformat.
            
        Returns
        -------
        pd.DataFrame
            The reformatted dataframe.
        '''
        # create a copy of the dataframe to work with
        df = df_.copy()
        # loop through columns to see which should be reformatted
        to_reformat = [c for c in df.columns if c+'_SEM' in df.columns]
        # loop through columns to reformat
        for c in to_reformat:
            # recast to object type
            df[c] = df[c].astype(object)
            # create the ufloats
            for i in range(len(df)):
                df.at[i, c] = ufloat(df[c][i], df[c+'_SEM'][i])
        # drop the sem columns
        df.drop(columns=[c+'_SEM' for c in to_reformat], inplace=True)
        # return the new dataframe
        return df

def load_data(file_path:str) -> pd.DataFrame:
    ''' Load data saved by this class directly into a pandas dataframe.

    Parameters
    ----------
    file_path : str
        The path to the csv data file to load.
    
    Returns
    -------
    pd.DataFrame
        The data loaded from the file.
    '''
    # start by just loading the data
    df = pd.read_csv(file_path)
    # return a reformatted version with ufloats
    return reformat_float_to_ufloat(df)


if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    DATE = "07162025" #please update
    STATE = "ha_negpi_3_vd"
    TRIAL = 4

    chis = np.linspace(0.001, np.pi/2, 6)[1:]
    gammas = []
    
    fileName = f"gamma_data_from_tomo"
    directory = f'ria_{STATE}_trial{TRIAL}'

    for chi in chis:
        chi_act = np.rad2deg(chi)
        #bases for hdva state: 'DR', 'AR', 'DL', 'AL', 'RR', 'LL', 'RL', 'LR'
        #bases for hrvl state: 'DD', 'DA', 'AD', 'AA', 'RD', 'RA', 'LD', 'LA'
        datas = load_data(f'{directory}/tomo_data_{STATE}_{chi_act}_{DATE}_{TRIAL}.csv')

        datas = datas.set_index("note")

        if STATE == 'ha_negpi_3_vd':
            # for havd states:
            gamma = unp.arctan2((datas.at['DR', 'C4']+datas.at['AL', 'C4']-datas.at['DL', 'C4']-datas.at['AR', 'C4']), -(datas.at['RR', 'C4']+datas.at['LL', 'C4']-datas.at['LR', 'C4']-datas.at['RL', 'C4']))
        elif STATE == 'hd_negpi_3_va':
            # for hdva state: 
            gamma = unp.arctan2(-(datas.at['DR', 'C4']+datas.at['AL', 'C4']-datas.at['DL', 'C4']-datas.at['AR', 'C4']), (datas.at['RR', 'C4']+datas.at['LL', 'C4']-datas.at['LR', 'C4']-datas.at['RL', 'C4']))
        elif STATE == 'hr_negpi_6_vl':
            # for hrvl state:
            gamma = unp.arctan2((datas.at['DD', 'C4']+datas.at['AA', 'C4']-datas.at['DA', 'C4']-datas.at['AD', 'C4']), -(datas.at['RD', 'C4']+datas.at['LA', 'C4']-datas.at['RA', 'C4']-datas.at['LD', 'C4']))
        
        gammas.append(gamma)

    df = pd.DataFrame({'chi': chis, 'gamma': gammas})


    pd.DataFrame(df).to_csv(f'{directory}/{fileName}.csv')

    df = load_data(f'{directory}/{fileName}.csv')

    chis, gammas = df['chi'], df['gamma'].apply(unc.ufloat_fromstr)

    params = analysis.fit('line', chis, gammas)
    analysis.plot_func('line', params, chis, color='blue')
    analysis.plot_errorbar(chis, gammas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('Chi')
    plt.ylabel('Gamma (rad)')
    plt.savefig(f'{directory}/{fileName}.png', dpi=600)
    plt.show()

