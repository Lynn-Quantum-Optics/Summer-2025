'''
AuthorL Lev Gruber
Last Update: 5/17/2024

This file uses code from Oscar's process_expt_richard.py in Summer 2023 to brute force
calculate eta and chi values (for the state cos(eta)PHI+ + sin(eta)e^i*chi PHI-) in which W is positive
and W' is negative. 

It does so by:
1. Creating theoretical density matrices at a variety of eta values and 1-2 chi values
2. Computing both W and W' witnesses for each density matrix
3. Pushing values with +W and -W' into one df, and values with +W, +W' into another. 

In the future, I hope to add functionality to compute V witnesses for states for which W and W' are positive.
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

from uncertainties import ufloat
from uncertainties import unumpy as unp

# Original code adapted from Oscar Scholin in Summer 2023
from legacy_sample_rho import *
from legacy_rho_methods import *

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
def analyze_rho(rho_actual, verbose = False, id='id'):
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
    
    # calculate W and W' theory
    W_T_ls = compute_witnesses(rho = rho_actual, return_all = True) # theory #, return_all = True, return_params = True
    # parse lists
    W_3s = W_T_ls[:6]
    W_5_t1s = W_T_ls[6:9]
    W_5_t2s = W_T_ls[9:12]
    W_5_t3s = W_T_ls[12:15]

    W_min = min(W_T_ls[:6])
    Wp_t1 = min(W_T_ls[6:9])
    Wp_t2 = min(W_T_ls[9:12])
    Wp_t3 = min(W_T_ls[12:15])
    
    if verbose:
        # If verbose, return not just the 4 values but also the names of the minimal witnesses!
        # Define dictionary to get name of
        all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
        index_names = {i: name for i, name in enumerate(all_W)}

        # Get which W/W' were minimized
        W_min_name = index_names.get(W_T_ls.index(W_min), 'Unknown')
        Wp1_min_name = index_names.get(W_T_ls.index(Wp_t1), 'Unknown')
        Wp2_min_name = index_names.get(W_T_ls.index(Wp_t2), 'Unknown')
        Wp3_min_name = index_names.get(W_T_ls.index(Wp_t3), 'Unknown')
        
        # Return the params as well, add in if return_params = True in compute_witness call
        #W_param = W_params[0]
        #Wp1_param = W_params[1]
        #Wp2_param = W_params[2]
        #Wp3_param = W_params[3]
        #add to returns , W_param, Wp1_param, Wp2_param, Wp3_param
        
        # Find names from dictionary and return them and their values
        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_3s, W_5_t1s, W_5_t2s, W_5_t3s
    else:
         return W_min, Wp_t1, Wp_t2, Wp_t3

def plot_all(name, etas):
    '''
    Plots data from main
    
    Parameters:
    names (list): which csv names to read
    
    W (array): Minimum W array
    min_Wp (array): Minimum W prime array
    eta (float): eta value 
    chi (array): chi array
    name (string): name for file output (of state)
    '''
    fig, ax = plt.subplots()
    etas_nice = []
    for i, eta in enumerate(etas):
        # Read in CSV
        data = pd.read_csv(f'pure_and_mixed_states/{name}_{etas[i]}.csv')
        # Extract data
        W = data['W']
        min_Wp = data['min_W_prime']
        chi = data['chi']

        # Plot data
        etas_nice.append(round(np.degrees(eta)))
        ax.plot(chi, W, label = f'$W, \eta = {etas_nice[i]}$')
        ax.plot(chi, min_Wp, label = f'$W\prime, \eta = {etas_nice[i]}$')
  
    name = f'{name}_{etas_nice[0]}'  #_{etas_nice[1]}_{etas_nice[2]}_{etas_nice[3]}
    print(name)
    ax.axhline(0, color='black', linewidth=0.5) 
    ax.set_title(f'State {name}', fontsize=12)
    ax.set_ylabel('Witness value', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    ax.legend(fontsize=8, loc = 'upper right')
    ax.set_xlabel('$\chi$', fontsize=12)
    plt.savefig(f'pure_and_mixed_states/{name}.pdf')

####### Main Functions

if __name__ == '__main__':
    
    #######  Choose your run parameters
    
    #points = 25 # number of points to take
    #chis = np.linspace(0.001, np.pi/2, points)
    num_etas = 1  # the number of etas you choose below
    etas = [np.pi/4] #np.pi/12, np.pi/6, np.pi/4 , np.pi/3
    chis = np.linspace(0.001, np.pi/2, 6)
    plot = False
    state = 'hr_negpi_6_vl'

    ####### Witness each rho and save it
    
    ### Save witness values to above lists of lists
    for chi in chis:
        rho = get_theo_rho(state, chi)
        # find the minimum witness expectation value of the 3 w primes and of the W.
        # Add in to get which W was minimum: W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp1_param, Wp2_param, Wp3_param,  verbose = True
        WM, WP1, WP2, WP3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_3s, W_5_t1s, W_5_t2s, W_5_t3s = analyze_rho(rho, verbose=True) 
        min_wp = min(WP1, WP2, WP3)
        print('My Ws were:', min_wp, 'at chi:', chi)
        print('Minimum W was:', W_min_name, 'with value', WM)
        print('Minimum W prime 1 was:', Wp1_min_name, 'with value', WP1)
        print('Minimum W prime 2 was:', Wp2_min_name, 'with value', WP2)
        print('Minimum W prime 3 was:', Wp3_min_name, 'with value', WP3)
        print('My W3s were:', W_3s)
        print('My W_5 triplets 1s were:', W_5_t1s)
        print('My W_5 triplets 2s were:', W_5_t2s)
        print('My W_5 triplets 3s were:', W_5_t3s)
    
    # wp8 is def the min for this one (hd_negpi_3_va) :)
    #all other w_5s are near zero, w_3s are above zero




