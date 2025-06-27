#from lab_framework import Manager, analysis
import numpy as np
import scipy.optimize as opt # type: ignore
from analysis_old import *
import pandas as pd
import analysis2_delete as analysis # delete
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat

# Use this file to load UVHWP sweep data generated in the full tomography.
# This file will plot the data and generate the same chi values used in the tomography.

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
    basisName = 'hd_negpi_3_va'
    mpName = "ria"
    date = "06252025"
    TRIAL = 9
    CHI_PARAMS = [0.001, np.pi/2, 6]

    # manually perform sweep of UVHWP for various chi values (chi/2=eta)
    chi_vals = np.linspace(*CHI_PARAMS)

    # obtain the first round of data and switch to a new output file
    data1 = pd.read_csv(f"{mpName}_{basisName}_trial{TRIAL}/UVHWP_balance_sweep1 original.csv")
    data2 = pd.read_csv(f'{mpName}_{basisName}_trial{TRIAL}/UVHWP_balance_sweep2 original.csv')

    args1, unc1 = fit('sin2_sq', data1.C_UV_HWP, data1.C4, data1.C4_SEM)
    args2, unc2 = fit('sin2_sq', data2.C_UV_HWP, data2.C4, data2.C4_SEM)

    for chi in chi_vals:
        # Calculate the UVHWP angle we want for a given CHI value
        desired_ratio = (np.cos(chi/2) / np.sin(chi/2))**2
        def min_me(x_:np.ndarray, args1_:tuple, args2_:tuple):
            ''' Function want to minimize'''
            return (sin2_sq(x_, *args1_) / sin2_sq(x_, *args2_) - desired_ratio)**2
        x_min, x_max = np.min(data1.C_UV_HWP), np.max(data1.C_UV_HWP)
        UVHWP_angle = opt.brute(min_me, args=(args1, args2), ranges=((x_min, x_max),))

        # might need to retune this if there are multiple roots. I'm only assuming one root
        print(UVHWP_angle)

    df1 = load_data(f"{mpName}_{basisName}_trial{TRIAL}/UVHWP_balance_sweep1 original.csv")
    df2 = load_data(f"{mpName}_{basisName}_trial{TRIAL}/UVHWP_balance_sweep2 original.csv")


    angles1, gammas1 = df1['C_UV_HWP'], df1['C4']
    angles2, gammas2 = df2['C_UV_HWP'], df2['C4']

    thetas = unp.arctan(unp.sqrt((gammas2)/(gammas1)))

    params = analysis.fit('line', angles1, thetas)
    analysis.plot_func('line', params, angles1, color='blue')
    analysis.plot_errorbar(angles1, thetas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Theta Parameter (rad)')
    plt.savefig(f'{mpName}_{basisName}_trial{TRIAL}/UVHWP_plot_{TRIAL}.png', dpi=600)
    plt.show()

    x = analysis.find_value('line', params, np.pi/4, angles1)
    print(f'Pi/4 at {x}')
