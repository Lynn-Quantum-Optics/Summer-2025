"""
Authors: Lev G., Isabel G.
Last updated: 5/28/2025

This file reads and processes experimentally collected density matrices using functionality from
states_and_witnesses.py and operations.py, so make sure to either copy those files to your directory
or update the path variables to import them. This file no longer depends on rho_methods.py or
sample_rho.py.

To run this file, you don't need to copy or edit it. Simply run it and fill in the user inputs as
prompted in your command line.
"""

print("initializing...")

# Silence TensorFlow warnings that make it hard to read outputs of this file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from scipy.optimize import curve_fit
import scipy.linalg as la

from uncertainties import ufloat
from uncertainties import unumpy as unp

# import Isabel & Brayden's files
import states_and_witnesses as sw
import operations as op

def get_rho_from_file(filename, verbose=True, angles=None):
    '''Function to read in experimental density matrix from file. For trials > 14. N.b. up to trial 23,
    angles were not saved (but recorded in lab_notebook markdown file). Also note that in trials 20
    (E0 eta = 45), 21 (blueo of E0 (eta = 45, chi = 0, 18)), 22 (E0 eta = 60), and 23
    (E0 eta = 60, chi = -90), there was a sign error in the phi phase in the Jones matrices,
    so will recalculate the correct density matrix; ** the one saved in the file as the theoretical
    density matrix is incorrect **
    --
    Parameters
        filename : str, Name of file to read in
        verbose : bool, Whether to print out results
        angles: list, List of angles used in the experiment. If not None, will assume angles provided in
        the data file.
    '''
    def split_filename():
            ''' Splits up the file name and identifies the trial number, eta, and chi values'''

            # split filename
            split_filename = filename.split('(')
            split_filename = split_filename[1].split(')')
            
            # get trial number
            trial = int(split_filename[0].split('-')[2].split(')')[0])
            chi = float(split_filename[0].split('-')[1])

            return trial, chi

    # read in data
    rho, unc, Su, un_proj, un_proj_unc, chi, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
    trial, chi = split_filename()
    
    # print results
    if verbose:
        print('angles\n---')
        print(angles)
        print('measured rho\n---')
        print(rho)
        print('uncertainty \n---')
        print(unc)
        print('fidelity', fidelity)
        print('purity', purity)

        print('trace of measured rho:', np.trace(rho))
        print('eigenvalues of measured rho:', np.linalg.eigvals(rho))

    return trial, rho, unc, Su, fidelity, purity, chi, angles, un_proj, un_proj_unc

def adjust_rho(rho, expt_purity, state = 'E0'):
    ''' Adjusts theo density matrix to account for experimental impurity
        Multiplies unwanted experimental impurities (top right bottom right block) by expt purity to account for 
        non-entangled particles in our system '''
    if state == 'E0':
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if i < 3:
                    if j < 3:
                        pass
                if i > 2:
                    if j > 2:
                        pass
                else:
                    rho[i][j] = expt_purity * rho[i][j]
    return rho

def get_fidelity(rho1, rho2):
    '''Compute fidelity of 2 density matrices'''
    try:
        fidelity = np.real((np.trace(la.sqrtm(la.sqrtm(rho1)@rho2@la.sqrtm(rho1))))**2)
        return fidelity
    except:
        print('error computing fidelity!')
        print('rho1', rho1)
        print('rho2', rho2)
        return 1e-5

def parse_W_ls(W_params, W_vals):
            """
            A function to parse the lists of outputs from minimize_witnesses.
            Inputs:
                W_params: a list of the parameters used to minimize each witness.
                W_vals: a list of the minimum expectation value of each witness.
            """
            W_names = ['W3_1','W3_2', 'W3_3', 'W3_4', 'W3_5', 'W3_6', 'W5_1', 'W5_2', 'W5_3', 'W5_4', 'W5_5', 'W5_6', 'W5_7', 'W5_8', 'W5_9']

            # Map the names of the W3s to their minimum expectation values
            W3_vals_dict = dict(zip(W_names[:6], W_vals[:6]))
            W3_params_dict = dict(zip(W_names[:6], W_params[:6]))

            # Search the dictionary for the minimum W3 and save its name, expec. value, and minimization param
            W3_name = min(W3_vals_dict, key=W3_vals_dict.get)
            W3_min = W3_vals_dict[W3_name]
            W3_param = W3_params_dict[W3_name]

            # Creating W5 dictionaries
            # Triplet 1
            W5t1_vals_dict = dict(zip(W_names[6:9], W_vals[6:9]))
            W5t1_params_dict = dict(zip(W_names[6:9], W_params[6:9]))

            #Triplet 2
            W5t2_vals_dict = dict(zip(W_names[9:12], W_vals[9:12]))
            W5t2_params_dict = dict(zip(W_names[9:12], W_params[9:12]))

            #Triplet 3
            W5t3_vals_dict = dict(zip(W_names[12:15], W_vals[12:15]))
            W5t3_params_dict = dict(zip(W_names[12:15], W_params[12:15]))

            # Finding the minimum for each W5 triplet
            W5_t1_name = min(W5t1_vals_dict, key=W5t1_vals_dict.get)
            W5_t1_min = W5t1_vals_dict[W5_t1_name]
            W5_t1_param = W5t1_params_dict[W5_t1_name]

            W5_t2_name = min(W5t2_vals_dict, key=W5t2_vals_dict.get)
            W5_t2_min = W5t2_vals_dict[W5_t2_name]
            W5_t2_param = W5t2_params_dict[W5_t2_name]

            W5_t3_name = min(W5t3_vals_dict, key=W5t3_vals_dict.get)
            W5_t3_min = W5t3_vals_dict[W5_t3_name]
            W5_t3_param = W5t3_params_dict[W5_t3_name]

            return W3_min, W5_t1_min, W5_t2_min, W5_t3_min, W3_name, W5_t1_name, W5_t2_name, W5_t3_name, W3_param, W5_t1_param, W5_t2_param, W5_t3_param

def analyze_rhos(filenames, rho_actuals, id='id'):
    '''Extending get_rho_from_file to include multiple files; 
    __
    inputs:
        filenames: list of filenames to analyze
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
    __
    returns: df with:
        - trial number
        - eta (if they exist)
        - chi (if they exist)
        - fidelity
        - purity
        - W theory (adjusted for purity) and W expt and W unc
        - W' theory (adjusted for purity) and W' expt and W' unc
    '''
    # initialize df
    df = pd.DataFrame()
    eta = 45.0 # TODO: reformat for no eta with new rho methods file

    for i, file in tqdm(enumerate(filenames)):
        trial, rho, unc, Su, fidelity, purity, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
        print('Purity is:', purity)
        print('Fidelity is:', fidelity)
        rho_actual = rho_actuals[i]
        
        print('Theoretical rho is:')
        print(np.round(rho_actual, 4))
        print('Experimental rho is:')
        print(np.round(rho, 3))
        
        # calculate W and W' theory
        W_T_params, W_T_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho_actual)
        print("Theory Ws computed")
        W_AT_params, W_AT_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=adjust_rho(rho_actual, [eta, chi], 0.95))
        print("Adjusted theory Ws computed")
        # calculate W and W' expt
        # TODO: assertion error when giving rho & counts, just giving counts causes index error
        W_E_params, W_E_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho)
        # W_E_params, W_E_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho, counts=unp.uarray(un_proj, un_proj_unc))
        print("Experimental Ws computed")
        
        ## PARSE LISTS ##
        # Theoretical data
        W_min_T, Wp_t1_T, Wp_t2_T, Wp_t3_T, W_name_T, Wp1_name_T, Wp2_name_T, Wp3_name_T, W_param_T, Wp1_param_T, Wp2_param_T, Wp3_param_T = parse_W_ls(W_T_params, W_T_vals)
        
        # Adjusted theory: params are not returned
        W_min_AT, Wp_t1_AT, Wp_t2_AT, Wp_t3_AT, W_name_AT, Wp1_name_AT, Wp2_name_AT, Wp3_name_AT, _, _, _, _ = parse_W_ls(W_AT_params, W_AT_vals)
        
        print('The minimized first triplet W5 is:', Wp1_name_T)
        print('The minimized second triplet W5 is:', Wp2_name_T)
        print('The minimized third triplet W5 is:', Wp3_name_T)

        W_expt_ls = list(parse_W_ls(W_E_params, W_E_vals))

        # using propagated uncertainty
        W_min_expt = unp.nominal_values(W_expt_ls[0])
        W_min_unc = unp.std_devs(W_expt_ls[0])
        Wp_t1_expt = unp.nominal_values(W_expt_ls[1])
        Wp_t1_unc = unp.std_devs(W_expt_ls[1])
        Wp_t2_expt = unp.nominal_values(W_expt_ls[2])
        Wp_t2_unc = unp.std_devs(W_expt_ls[2])
        Wp_t3_expt = unp.nominal_values(W_expt_ls[3])
        Wp_t3_unc = unp.std_devs(W_expt_ls[3])
        W_name_expt = W_expt_ls[4]
        Wp1_name_expt = W_expt_ls[5]
        Wp2_name_expt = W_expt_ls[6]
        Wp3_name_expt = W_expt_ls[7]
        W_param_expt = W_expt_ls[8]
        Wp1_param_expt = W_expt_ls[9]
        Wp2_param_expt = W_expt_ls[10]
        Wp3_param_expt = W_expt_ls[11]

        if eta is not None and chi is not None:
            adj_fidelity = get_fidelity(adjust_rho(rho_actual, [eta, chi], purity), rho)

            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'eta':eta, 'chi':chi, 'fidelity':fidelity, 'purity':purity, 'AT_fidelity':adj_fidelity,
            'W_min_T': W_min_T, 'Wp_t1_T':Wp_t1_T, 'Wp_t2_T':Wp_t2_T, 'Wp_t3_T':Wp_t3_T,'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

        else:
            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

    # save df
    print('saving dataframe...')
    df.to_csv(join(DATA_PATH, f'analysis_{id}.csv'))

def make_plots_E0(dfname):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    dfname: str, name of df to read in
    num_plots: int, number of separate plots to make (based on eta)
    '''
    print("plotting...")
    id = dfname.split('.')[0].split('_')[-1] # extract identifier from dfname

    # read in df
    df = pd.read_csv(join(DATA_PATH, dfname))
    eta_vals = df['eta'].unique()

    # preset plot sizes
    if len(eta_vals) == 1:
        fig, ax = plt.subplots(figsize = (8, 8))
        # get df for each eta
        df_eta = df
        purity_eta = df_eta['purity'].to_numpy()
        fidelity_eta = df_eta['fidelity'].to_numpy()
        chi_eta = df_eta['chi'].to_numpy()
        adj_fidelity = df_eta['AT_fidelity'].to_numpy()

        # # do purity and fidelity plots
        # ax[1,i].scatter(chi_eta, purity_eta, label='Purity', color='gold')
        # ax[1,i].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')

        # # plot adjusted theory purity
        # ax[1,i].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

        # extract witness values
        W_min_T = df_eta['W_min_T'].to_numpy()
        W_min_AT = df_eta['W_min_AT'].to_numpy()
        W_min_expt = df_eta['W_min_expt'].to_numpy()
        W_min_unc = df_eta['W_min_unc'].to_numpy()

        Wp_T = df_eta[['Wp_t1_T', 'Wp_t2_T', 'Wp_t3_T']].min(axis=1).to_numpy()
        Wp_AT = df_eta[['Wp_t1_AT', 'Wp_t2_AT', 'Wp_t3_AT']].min(axis=1).to_numpy()
        Wp_expt = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].min(axis=1).to_numpy()
        Wp_expt_min = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].idxmin(axis=1)
        Wp_unc = np.where(Wp_expt_min == 'Wp_t1_expt', df_eta['Wp_t1_unc'], np.where(Wp_expt_min == 'Wp_t2_expt', df_eta['Wp_t2_unc'], df_eta['Wp_t3_unc']))

        # plot curves for T and AT
        def sinsq(x, a, b, c, d):
            return a*np.sin(b*np.deg2rad(x) + c)**2 + d

        popt_W_T_eta, pcov_W_T_eta = curve_fit(sinsq, chi_eta, W_min_T, maxfev = 10000)
        popt_W_AT_eta, pcov_W_AT_eta = curve_fit(sinsq, chi_eta, W_min_AT, maxfev = 10000)
        #print('popt_W are:', popt_W_AT_eta) 
        popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(sinsq, chi_eta, Wp_T, maxfev = 10000)
        popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(sinsq, chi_eta, Wp_AT, maxfev = 10000)
        
        chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
        ax.errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue', markersize=10)

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
        ax.errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon', markersize=10)
        #ax.set_title(f'$\eta = 45\degree$', fontsize=18)
        ax.set_ylabel('Witness value', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(ncol=2, fontsize=20)
        ax.set_xlabel('$\chi$ (deg)', fontsize=20)
        ax.axhline(y=0, color='black')
        # ax[1,i].set_ylabel('Value', fontsize=31)
        # ax[1,i].legend()
    else:
        if len(eta_vals) == 2:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        elif len(eta_vals) == 3:
            fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex=True)
        
        for i, eta in enumerate(eta_vals):
            # get df for each eta
            df_eta = df[df['eta'] == eta]
            purity_eta = df_eta['purity'].to_numpy()
            fidelity_eta = df_eta['fidelity'].to_numpy()
            chi_eta = df_eta['chi'].to_numpy()
            adj_fidelity = df_eta['AT_fidelity'].to_numpy()

            # # do purity and fidelity plots
            # ax[1,i].scatter(chi_eta, purity_eta, label='Purity', color='gold')
            # ax[1,i].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')

            # # plot adjusted theory purity
            # ax[1,i].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

            # extract witness values
            W_min_T = df_eta['W_min_T'].to_numpy()
            W_min_AT = df_eta['W_min_AT'].to_numpy()
            W_min_expt = df_eta['W_min_expt'].to_numpy()
            W_min_unc = df_eta['W_min_unc'].to_numpy()

            Wp_T = df_eta[['Wp_t1_T', 'Wp_t2_T', 'Wp_t3_T']].min(axis=1).to_numpy()
            Wp_AT = df_eta[['Wp_t1_AT', 'Wp_t2_AT', 'Wp_t3_AT']].min(axis=1).to_numpy()
            Wp_expt = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].min(axis=1).to_numpy()
            Wp_expt_min = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].idxmin(axis=1)
            Wp_unc = np.where(Wp_expt_min == 'Wp_t1_expt', df_eta['Wp_t1_unc'], np.where(Wp_expt_min == 'Wp_t2_expt', df_eta['Wp_t2_unc'], df_eta['Wp_t3_unc']))

            # plot curves for T and AT
            def sinsq(x, a, b, c, d):
                return a*np.sin(b*np.deg2rad(x) + c)**2 + d
            popt_W_T_eta, pcov_W_T_eta = curve_fit(sinsq, chi_eta, W_min_T)
            popt_W_AT_eta, pcov_W_AT_eta = curve_fit(sinsq, chi_eta, W_min_AT)

            popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(sinsq, chi_eta, Wp_T)
            popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(sinsq, chi_eta, Wp_AT)

            chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
            ax[i].errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue')

            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
            ax[i].errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon')

            ax[i].set_title('$\eta = 30\degree$', fontsize=33)
            ax[i].set_ylabel('Witness value', fontsize=31)
            ax[i].tick_params(axis='both', which='major', labelsize=25)
            ax[i].legend(ncol=2, fontsize=25)
            ax[i].set_xlabel('$\chi$', fontsize=31)
            # ax[1,i].set_ylabel('Value', fontsize=31)
            # ax[1,i].legend()
            
    plt.suptitle('Entangled State Witnessed by 2nd W\' Triplet', fontsize=25)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'{id}.pdf'))
    plt.show()

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

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


def get_theo_rho(state, chi):
    '''
    Calculates the density matrix (rho) for a given set of parameters (eta, chi) for Stuart's states
    
    Parameters:
    state (string): The name of the state we are analyzing
    chi (float): The parameter chi
    
    Returns:
    rho (numpy.ndarray): The density matrix
    '''
    # Define kets and bell states in vector form 
    H = ket([1,0])
    V = ket([0,1])
    R = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
    L = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
    D = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
    A = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])
    
    PHI_PLUS = (np.kron(H,H) + np.kron(V,V))/np.sqrt(2)
    PHI_MINUS = (np.kron(H,H) - np.kron(V,V))/np.sqrt(2)
    PSI_PLUS = (np.kron(H,V) + np.kron(V,H))/np.sqrt(2)
    PSI_MINUS = (np.kron(H,V) - np.kron(V,H))/np.sqrt(2)

    
    ## The following state(s) are an attempt to find new positive W negative W prime states.
    if state == 'HR_VL':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,R) + (1 - np.exp(1j*chi))/2 * np.kron(V,L)
    
    if state == 'HR_iVL':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,R) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,L)
    
    if state == 'HL_VR':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,L) + (1 - np.exp(1j*chi))/2 * np.kron(V,R)
        
    if state == 'HL_iVR':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,L) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,R)
        
    if state == 'HD_VA':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,D) + (1 - np.exp(1j*chi))/2 * np.kron(V,A)
    
    if state == 'HD_iVA':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,D) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,A)
        
    if state == 'HA_VD':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,A) + (1 - np.exp(1j*chi))/2 * np.kron(V,D)
    
    if state == 'HA_iVD':
        phi = (1 + np.exp(1j*chi))/2 * np.kron(H,A) + 1j*(1 - np.exp(1j*chi))/2 * np.kron(V,D)
        
    if state == 'cosHL_sinVR':
        phi = np.cos(chi/2) * np.kron(H, L) + np.sin(chi/2) * np.kron(V,R)    
    
    if state == 'cosHR_minussinVL':
        phi = np.cos(chi/2) * np.kron(H, R) - np.sin(chi/2) * np.kron(V,L) # no i shows in this form
        
    if state == 'cosHL_minussinVR': 
        phi = np.cos(chi/2) * np.kron(H, L) - np.sin(chi/2) * np.kron(V,R)
        
    if state == 'cosHD_minussinVA':
        phi = np.cos(chi/2) * np.kron(H,D) - np.sin(chi/2) * np.kron(V,A) # no i shows in this form
        
    if state == 'cosHD_sinVA':
        phi = np.cos(chi/2) * np.kron(H,D) + np.sin(chi/2) * np.kron(V,A)
        
    if state == 'cosHA_minussinVD':
        phi = np.cos(chi/2) * np.kron(H,A) - np.sin(chi/2) * np.kron(V,D)
        
    if state == 'cosHA_minusisinVD':
        phi = np.cos(chi/2) * np.kron(H,A) - 1j * np.sin(chi/2) * np.kron(V,D)
            
    if state == 'cosHR_minusisinVL':
        phi = np.cos(chi/2) * np.kron(H, R) - 1j * np.sin(chi/2) * np.kron(V,L) 
        
    if state == 'cosHL_minusisinVR':
        phi = np.cos(chi/2) * np.kron(H, L) - 1j * np.sin(chi/2) * np.kron(V,R) 
        
    if state =='cosHA_minusiphasesinVD':
        phi = np.cos(chi/2) * np.kron(H, A) - np.exp(1j * 1.311) * np.sin(chi/2) * np.kron(V,D)
    
    if state == 'hd_negpi_3_va':
        phi = np.cos(chi/2) * np.kron(H, D) + np.exp(-1j * np.pi/3) * np.sin(chi/2) * np.kron(V, A)

    if state =='cosHA_minusphasesinVD':
        phi = np.cos(chi/2) * np.kron(H, A) + np.exp(-1j * 1.27) * np.sin(chi/2) * np.kron(V,D)
    
    # create rho and return it
    rho = phi @ phi.conj().T
    return rho

if __name__ == '__main__':
    # set path & other user input variables
    current_path = dirname(abspath(__file__))
    DATA_PATH = input('Input the path to the lowest-level directory that your data file is in: ')
    TRIAL = int(input("Trial number: "))
    STATE_ID = input("State name: ")

    chis_range = input("Are you analyzing the full range of chi values? [y/n]: ")
    if chis_range.lower() == "y":
        chis = np.linspace(0.001, np.pi/2, 6)
    else:
        chis_str = input("Which chi value do you want to test (must be in radians; e.g. 'np.pi/2')?\nType nothing and hit ENTER to assign a default value of pi/2 radians: ")
        if chis_str == "":
            chis = [np.pi/2]
        else:
            chis = [eval(chis_str)]

    rho_actuals = []
    filenames = []
    rho_actuals = []

    # Obtain the density matrix for each state
    for chi in chis:
        rho_actuals.append(get_theo_rho(STATE_ID, chi))
        filenames.append(f"rho_({STATE_ID}-{np.rad2deg(chi)}-{TRIAL}).npy")

    # analyze rho files
    analyze_rhos(filenames, rho_actuals, id=STATE_ID)
    make_plots_E0(f'analysis_{STATE_ID}.csv')