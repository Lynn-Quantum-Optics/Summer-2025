'''
Author: Ria Haapala
Last Update: 6/23/2025

This file is has been compiled using Lev Gruber's testing witnesses file for mixed states.

For a given state of the form cos(chi/2)*|alpha>|beta> + e^i*gamma sin(chi/2)|alpha_perp>|beta_perp>, 
this file will calculate all W_3 and W_5 witness values over the specified range using the legacy
rho methods.
'''

# Imports
import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import sympy as sp
from scipy.optimize import curve_fit

from uncertainties import ufloat
from uncertainties import unumpy as unp

# import Isabel & Brayden's files
import states_and_witnesses as sw
import operations as op
from process_expt import parse_W_ls

####### Helper methods for rho creation
def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)


def get_theo_rho(state, chi):
    '''
    Calculates the density matrix (rho) for a given set of paramters (eta, chi) for Stuart's states
    
    Parameters:
    state (string): Which state we want
    eta (float): The parameter eta.
    chi (float): The parameter chi.
    
    Returns:
    numpy.ndarray: The density matrix (rho)
    '''
    # Define kets and bell states in vector form 
    H = ket([1,0])
    V = ket([0,1])
    R = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
    L = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
    D = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
    A = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])
    
    if state == 'hr_negpi_6_vl':
        phi = np.cos(chi/2) * np.kron(H, R) + np.exp(-1j * np.pi/6) * np.sin(chi/2) * np.kron(V, L)
    
    if state == 'hd_negpi_3_va':
        phi = np.cos(chi/2) * np.kron(H, D) + np.exp(-1j * np.pi/3) * np.sin(chi/2) * np.kron(V, A)
    
    # create rho and return it
    rho = phi @ phi.conj().T
    return rho
    
def create_noise(rho, power):
    '''
    Adds noise of order power to a density matrix rho
    
    Parameters:
    rho: NxN density matrix
    power: integer multiple of 10
    
    Returns:
    noisy_rho: rho with noise
    '''
    
    # get size of matrix
    n, _ = rho.shape
    
    # iterature over matrix and add some random noise to each elemnent
    for i in range(n):
        for j in range(n):
            rando = random.random() / (10 ** power)
            rho[i,j] += rando
    noisy_rho = rho
    
    return noisy_rho

######## Helper methods for analyzing density matrices
def analyze_rho(rho_actual, id='id'):
    '''; 
    __
    inputs:
        filenames: list of filenames to analyze
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
    __
    returns: df with:
        - W theory (adjusted for purity) and W expt and W unc
        - W' theory (adjusted for purity) and W' expt and W' unc
    '''
    df = pd.DataFrame()
    W_T_params, W_T_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho_actual)

    data = {
            'W3': {},
            'W5': {
                't1': {},
                't2': {},
                't3': {}
            }
        }

    # Theoretical data
    data = parse_W_ls(W_T_params, W_T_vals, False, data, "T")

    print("\n------\nW3s\n------")
    print("\nAll theoretical W3s:", W_T_vals[:6])
    print("\nTheoretical W3 min:", data['W3']['min_T'])
    # Initialize W3 objects with the experimental and theoretical rhos
    W3_T_obj = sw.W3(rho=rho_actual)
    # Save W3's min name and params
    W3_idx_T = data['W3']['name_T']
    W3_param_T = data['W3']['param_T']

    print("\n------\nW5 t1\n------")
    print("\nAll theoretical W5s:", W_T_vals[6:9])
    print("\nTheoretical W5 triplet 1 min:", data['W5']['t1']['min_T'])
    # Initialize W5 objects with the experimental and theoretical rhos
    W5_T_obj = sw.W5(rho=rho_actual)
    # Save W5 t1's experimental min name and params
    W5t1_idx_T = data['W5']['t1']['name_T']
    W5t1_params_T = data['W5']['t1']['params_T']

    print("\n------\nW5 t2\n------")
    print("\nAll theoretical W5s:", W_T_vals[9:12])
    print("\nTheoretical W5 triplet 2 min:", data['W5']['t2']['min_T'])
    # Save W5 t2's experimental min name and params
    W5t2_idx_T = data['W5']['t2']['name_T']
    W5t2_params_T = data['W5']['t2']['params_T']

    print("\n------\nW5 t3\n------")
    print("\nAll theoretical W5s:", W_T_vals[12:15])
    print("\nTheoretical W5 triplet 3 min:", data['W5']['t3']['min_T'])
    # Save W5 t3's experimental min name and params
    W5t3_idx_T = data['W5']['t3']['name_T']
    W5t3_params_T = data['W5']['t3']['params_T']

    # Flatten the dictionary and make it into a dataframe
    flat_data = {}
    def flatten(d, parent_key='', sep='_'):
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key, sep=sep)
            else:
                flat_data[new_key] = v
    flatten(data)

    # Pull out min, uncertainties, and names of Ws
    min_data = {}
    for k, v in flat_data.items():
        if "param" not in k:
            min_data[k] = v

    # Save df
    print('saving dataframe...')
    df.to_csv(join(f'ria_temp_analysis_{id}.csv'))



def plot_all(dfname):
    print("plotting...")
    # read in df
    df = pd.read_csv(join(dfname))
    fig, ax = plt.subplots(figsize = (8, 8))
    chi = df['chi'].to_numpy()

    # extract witness values
    W3_min_T = df['W3_min_T'].to_numpy()

    W5_min_T = df[['W5_t1_min_T', 'W5_t2_min_T', 'W5_t3_min_T']].min(axis=1).to_numpy()

    # plot curves for T and AT
    def sinsq(x, a, b, c, d):
        return a*np.sin(b*np.deg2rad(x) + c)**2 + d

    """
    NOTE: popt is a list of optimal values for chi so that the sum of the
            squared residuals of f(xdata, *popt) - ydata is minimized, while pcov is a
            matrix representing the estimated approximate covariance of popt
    """
    popt_W3_T, pcov_W3_T = curve_fit(sinsq, chi, W3_min_T, maxfev = 10000)
    popt_W5_T, pcov_W5_T = curve_fit(sinsq, chi, W5_min_T, maxfev = 10000)
    
    chi_ls = np.linspace(min(chi), max(chi), 1000)

    ax.plot(chi_ls, sinsq(chi_ls, *popt_W3_T), label='$W_T^3$', color='navy')
 
    ax.plot(chi_ls, sinsq(chi_ls, *popt_W5_T), label="$W_T^5$", color='crimson')

    ax.set_ylabel('Witness value', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(ncol=2, fontsize=20)
    ax.set_xlabel('$\chi$ (deg)', fontsize=20)
    ax.axhline(y=0, color='black')

    plt.tight_layout()
    plt.savefig(join(f'ria_plotted_data_{id}.pdf'))
    plt.show()


####### Main Functions

if __name__ == '__main__':
    
    #######  Choose your run parameters
    
    chis = np.linspace(0.001, np.pi/2, 6)
    state = 'hd_negpi_3_va'
    id = state
    
    rho_actuals = []
    rho_actuals = []

    # Obtain the density matrix for each state
    for chi in chis:
        rho_actuals.append(get_theo_rho(id, chi))
        
    # analyze rho files
    analyze_rho(rho_actuals, id)
    
    plot_all(f'ria_temp_analysis_{id}.csv')




