"""
Authors: Lev G., Ria H., Isabel G.
Last updated: 6/24/2025

This file reads and processes experimentally collected density matrices using functionality from
states_and_witnesses.py and operations.py, so make sure to either copy those files to your directory
or update the path variables to import them. This file no longer depends on rho_methods.py or
sample_rho.py.

To run this file, simply use "run process_expt.py" and fill in the user inputs when prompted in
your command line. You can also feed in a .txt file as input if you run with the following
command format: "run process_expt.py <path_to_text_file>". Some of these text files already exist!
Check the data folder for a file called process_input.txt. Make sure to include the folder name in
the file path.

This file can be run on data files with the naming format "rho_(state_name-chi-trial).npy".
"""

print("initializing...")

# Silence TensorFlow warnings that make it hard to read outputs of this file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
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
    '''Function to read in experimental density matrix from file.
    --
    Parameters:
        filename : str, Name of file to read in
        verbose : bool, Whether to print out results
        angles: list, List of angles used in the experiment. If not None, will assume angles provided in
        the data file.
    '''
    def split_filename():
            ''' Splits up the file name and identifies the trial number, and chi values'''

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

def gen_mixed_state(state_list, state_prob, chi):
    '''
    Generates a density matrix for a given mixed state
    
    Parameters:
        state_list: list of names of states that compose the mixed state
        state_prob: list of respective probabilities of each state
        chi: chi value
    
    Returns:
        rho: the density matrix of the mixed state
    '''
    
    # get individual rho's per state in state_list, taking probability into account
    individual_rhos = []
    for i, state in enumerate(state_list):
        individual_rhos.append(state_prob[i] * get_theo_rho(state, chi))
    # sum all matrices in individual rhos
    rho = np.sum(individual_rhos, axis = 0)
    
    return rho

def adjust_rho(rho, expt_purity):
    ''' Adjusts theo density matrix to account for experimental impurity
        Multiplies unwanted experimental impurities (top right and bottom left blocks) by
        expt purity to account for non-entangled particles in our system '''
    adj_rho = rho.copy()
    adj_rho[:2, 2:] = adj_rho[:2, 2:] * expt_purity
    adj_rho[2:, :2] = adj_rho[2:, :2] * expt_purity
    return adj_rho

def get_fidelity(rho1, rho2):
    '''Compute fidelity of 2 density matrices'''
    try:
        fidelity = np.real((np.trace(la.sqrtm(la.sqrtm(rho1) @ rho2 @ la.sqrtm(rho1))))**2)
        return fidelity
    except:
        print('error computing fidelity!')
        print('rho1', rho1)
        print('rho2', rho2)
        return 1e-5

def parse_W_ls(W_params, W_vals, do_W7s_W8s, data_dict, intype, W_unc=None):
    """
    A function to parse the lists of outputs from minimize_witnesses.
    Parameters:
        W_params: a list of the parameters used to minimize each witness.
        W_vals: a list of the minimum expectation value of each witness.
        do_W7s_W8s: a boolean indicating whether or not W7s and W8s were calculated.
        data_dict: the nested dictionary to add the data to.
        intype: a label for the type of data being input, i.e. "T" for theory, "AT" for
                adjusted theory, or "E" for experiment
        W_unc (optional): a list of experimental uncertainties for the expectation values.
    """

    W_names = []
    for i in range(1, 7):
        W_names.append(f'W3_{i}')
    for i in range(1, 10):
        W_names.append(f'W5_{i}')
    for i in range(1, 109):
        W_names.append(f'W7_{i}')
    for i in range(1, 37):
        W_names.append(f'W8_{i}')

    # Map the names of all witnesses to their minimization params
    W_params_dict = dict(zip(W_names, W_params))

    ########
    ## W3s
    ########
    # Map the names of the W3s to their minimum expectation values
    W3_vals_dict = dict(zip(W_names[:6], W_vals[:6]))
    W3_min_name = min(W3_vals_dict, key=W3_vals_dict.get)

    # Search the dictionary for the minimum W3 and save its name,
    # expec. value, and minimization param
    data_dict['W3']['name_' + intype] = W3_min_name.split("_")[1]
    data_dict['W3']['min_' + intype] = W3_vals_dict[W3_min_name]
    data_dict['W3']['param_' + intype] = W_params_dict[W3_min_name]

    ########
    ## W5s
    ########
    # Triplet 1
    W5_t1_vals_dict = dict(zip(W_names[6:9], W_vals[6:9]))
    W5_t1_min_name = min(W5_t1_vals_dict, key=W5_t1_vals_dict.get)
    data_dict['W5']['t1']['name_' + intype] = W5_t1_min_name.split("_")[1]
    data_dict['W5']['t1']['min_' + intype] = W5_t1_vals_dict[W5_t1_min_name]
    data_dict['W5']['t1']['params_' + intype] = W_params_dict[W5_t1_min_name]

    #Triplet 2
    W5_t2_vals_dict = dict(zip(W_names[9:12], W_vals[9:12]))
    W5_t2_min_name = min(W5_t2_vals_dict, key=W5_t2_vals_dict.get)
    data_dict['W5']['t2']['name_' + intype] = W5_t2_min_name.split("_")[1]
    data_dict['W5']['t2']['min_' + intype] = W5_t2_vals_dict[W5_t2_min_name]
    data_dict['W5']['t2']['params_' + intype] = W_params_dict[W5_t2_min_name]

    #Triplet 3
    W5_t3_vals_dict = dict(zip(W_names[12:15], W_vals[12:15]))
    W5_t3_min_name = min(W5_t3_vals_dict, key=W5_t3_vals_dict.get)
    data_dict['W5']['t3']['name_' + intype] = W5_t3_min_name.split("_")[1]
    data_dict['W5']['t3']['min_' + intype] = W5_t3_vals_dict[W5_t3_min_name]
    data_dict['W5']['t3']['params_' + intype] = W_params_dict[W5_t3_min_name]

    # Handle uncertainties for experimental data
    if W_unc is not None:
        W_unc_dict = dict(zip(W_names, W_unc))
        data_dict['W3']['unc_' + intype] = W_unc_dict[W3_min_name]

        data_dict['W5']['t1']['unc_' + intype] = W_unc_dict[W5_t1_min_name]
        data_dict['W5']['t2']['unc_' + intype] = W_unc_dict[W5_t2_min_name]
        data_dict['W5']['t3']['unc_' + intype] = W_unc_dict[W5_t3_min_name]
    
    # If we calculated W7s and W8s, create their dictionaries
    if do_W7s_W8s:
        ########
        ## W7s
        ########
        # set 1: no XY, YX
        W7_s1_vals_dict = dict(zip(W_names[15:27], W_vals[15:27]))
        W7_s1_min_name = min(W7_s1_vals_dict, key=W7_s1_vals_dict.get)
        data_dict['W7']['no_XY_YX']['name_' + intype] = W7_s1_min_name.split("_")[1]
        data_dict['W7']['no_XY_YX']['min_' + intype] = W7_s1_vals_dict[W7_s1_min_name]
        data_dict['W7']['no_XY_YX']['params_' + intype] = W_params_dict[W7_s1_min_name]

        # set 2: no XY, YZ
        W7_s2_vals_dict = dict(zip(W_names[27:39], W_vals[27:39]))
        W7_s2_min_name = min(W7_s2_vals_dict, key=W7_s2_vals_dict.get)
        data_dict['W7']['no_XY_YZ']['name_' + intype] = W7_s2_min_name.split("_")[1]
        data_dict['W7']['no_XY_YZ']['min_' + intype] = W7_s2_vals_dict[W7_s2_min_name]
        data_dict['W7']['no_XY_YZ']['params_' + intype] = W_params_dict[W7_s2_min_name]

        # set 3: no XY, ZX
        W7_s3_vals_dict = dict(zip(W_names[39:51], W_vals[39:51]))
        W7_s3_min_name = min(W7_s3_vals_dict, key=W7_s3_vals_dict.get)
        data_dict['W7']['no_XY_ZX']['name_' + intype] = W7_s3_min_name.split("_")[1]
        data_dict['W7']['no_XY_ZX']['min_' + intype] = W7_s3_vals_dict[W7_s3_min_name]
        data_dict['W7']['no_XY_ZX']['params_' + intype] = W_params_dict[W7_s3_min_name]

        # set 4: no XZ, ZX
        W7_s4_vals_dict = dict(zip(W_names[51:63], W_vals[51:63]))
        W7_s4_min_name = min(W7_s4_vals_dict, key=W7_s4_vals_dict.get)
        data_dict['W7']['no_XZ_ZX']['name_' + intype] = W7_s4_min_name.split("_")[1]
        data_dict['W7']['no_XZ_ZX']['min_' + intype] = W7_s4_vals_dict[W7_s4_min_name]
        data_dict['W7']['no_XZ_ZX']['params_' + intype] = W_params_dict[W7_s4_min_name]

        # set 5: no XZ, ZY
        W7_s5_vals_dict = dict(zip(W_names[63:75], W_vals[63:75]))
        W7_s5_min_name = min(W7_s5_vals_dict, key=W7_s5_vals_dict.get)
        data_dict['W7']['no_XZ_ZY']['name_' + intype] = W7_s5_min_name.split("_")[1]
        data_dict['W7']['no_XZ_ZY']['min_' + intype] = W7_s5_vals_dict[W7_s5_min_name]
        data_dict['W7']['no_XZ_ZY']['params_' + intype] = W_params_dict[W7_s5_min_name]

        # set 6: no XZ, YX
        W7_s6_vals_dict = dict(zip(W_names[75:87], W_vals[75:87]))
        W7_s6_min_name = min(W7_s6_vals_dict, key=W7_s6_vals_dict.get)
        data_dict['W7']['no_XZ_YX']['name_' + intype] = W7_s6_min_name.split("_")[1]
        data_dict['W7']['no_XZ_YX']['min_' + intype] = W7_s6_vals_dict[W7_s6_min_name]
        data_dict['W7']['no_XZ_YX']['params_' + intype] = W_params_dict[W7_s6_min_name]

        # set 7: no YX, ZY
        W7_s7_vals_dict = dict(zip(W_names[87:99], W_vals[87:99]))
        W7_s7_min_name = min(W7_s7_vals_dict, key=W7_s7_vals_dict.get)
        data_dict['W7']['no_YX_ZY']['name_' + intype] = W7_s7_min_name.split("_")[1]
        data_dict['W7']['no_YX_ZY']['min_' + intype] = W7_s7_vals_dict[W7_s7_min_name]
        data_dict['W7']['no_YX_ZY']['params_' + intype] = W_params_dict[W7_s7_min_name]

        # set 8: no YZ, ZY
        W7_s8_vals_dict = dict(zip(W_names[99:111], W_vals[99:111]))
        W7_s8_min_name = min(W7_s8_vals_dict, key=W7_s8_vals_dict.get)
        data_dict['W7']['no_YZ_ZY']['name_' + intype] = W7_s8_min_name.split("_")[1]
        data_dict['W7']['no_YZ_ZY']['min_' + intype] = W7_s8_vals_dict[W7_s8_min_name]
        data_dict['W7']['no_YZ_ZY']['params_' + intype] = W_params_dict[W7_s8_min_name]

        # set 9: no YZ, ZX
        W7_s9_vals_dict = dict(zip(W_names[111:123], W_vals[111:123]))
        W7_s9_min_name = min(W7_s9_vals_dict, key=W7_s9_vals_dict.get)
        data_dict['W7']['no_YZ_ZX']['name_' + intype] = W7_s9_min_name.split("_")[1]
        data_dict['W7']['no_YZ_ZX']['min_' + intype] = W7_s9_vals_dict[W7_s9_min_name]
        data_dict['W7']['no_YZ_ZX']['params_' + intype] = W_params_dict[W7_s9_min_name]

        ########
        ## W8s
        ########
        # set 1: no XY
        W8_s1_vals_dict = dict(zip(W_names[123:129], W_vals[123:129]))
        W8_s1_min_name = min(W8_s1_vals_dict, key=W8_s1_vals_dict.get)
        data_dict['W8']['no_XY']['name_' + intype] = W8_s1_min_name.split("_")[1]
        data_dict['W8']['no_XY']['min_' + intype] = W8_s1_vals_dict[W8_s1_min_name]
        data_dict['W8']['no_XY']['params_' + intype] = W_params_dict[W8_s1_min_name]

        # set 2: no YX
        W8_s2_vals_dict = dict(zip(W_names[129:135], W_vals[129:135]))
        W8_s2_min_name = min(W8_s2_vals_dict, key=W8_s2_vals_dict.get)
        data_dict['W8']['no_YX']['name_' + intype] = W8_s2_min_name.split("_")[1]
        data_dict['W8']['no_YX']['min_' + intype] = W8_s2_vals_dict[W8_s2_min_name]
        data_dict['W8']['no_YX']['params_' + intype] = W_params_dict[W8_s2_min_name]

        # set 3: no XZ
        W8_s3_vals_dict = dict(zip(W_names[135:141], W_vals[135:141]))
        W8_s3_min_name = min(W8_s3_vals_dict, key=W8_s3_vals_dict.get)
        data_dict['W8']['no_XZ']['name_' + intype] = W8_s3_min_name.split("_")[1]
        data_dict['W8']['no_XZ']['min_' + intype] = W8_s3_vals_dict[W8_s3_min_name]
        data_dict['W8']['no_XZ']['params_' + intype] = W_params_dict[W8_s3_min_name]

        # set 4: no ZX
        W8_s4_vals_dict = dict(zip(W_names[141:147], W_vals[141:147]))
        W8_s4_min_name = min(W8_s4_vals_dict, key=W8_s4_vals_dict.get)
        data_dict['W8']['no_ZX']['name_' + intype] = W8_s4_min_name.split("_")[1]
        data_dict['W8']['no_ZX']['min_' + intype] = W8_s4_vals_dict[W8_s4_min_name]
        data_dict['W8']['no_ZX']['params_' + intype] = W_params_dict[W8_s4_min_name]

        # set 5: no YZ
        W8_s5_vals_dict = dict(zip(W_names[147:153], W_vals[147:153]))
        W8_s5_min_name = min(W8_s5_vals_dict, key=W8_s5_vals_dict.get)
        data_dict['W8']['no_YZ']['name_' + intype] = W8_s5_min_name.split("_")[1]
        data_dict['W8']['no_YZ']['min_' + intype] = W8_s5_vals_dict[W8_s5_min_name]
        data_dict['W8']['no_YZ']['params_' + intype] = W_params_dict[W8_s5_min_name]

        # set 6: no ZY
        W8_s6_vals_dict = dict(zip(W_names[153:159], W_vals[153:159]))
        W8_s6_min_name = min(W8_s6_vals_dict, key=W8_s6_vals_dict.get)
        data_dict['W8']['no_ZY']['name_' + intype] = W8_s6_min_name.split("_")[1]
        data_dict['W8']['no_ZY']['min_' + intype] = W8_s6_vals_dict[W8_s6_min_name]
        data_dict['W8']['no_ZY']['params_' + intype] = W_params_dict[W8_s6_min_name]

        # Handle uncertainties for experimental data
        if W_unc is not None:
            data_dict['W7']['no_XY_YX']['unc_' + intype] = W_unc_dict[W7_s1_min_name]
            data_dict['W7']['no_XY_YZ']['unc_' + intype] = W_unc_dict[W7_s2_min_name]
            data_dict['W7']['no_XY_ZX']['unc_' + intype] = W_unc_dict[W7_s3_min_name]
            data_dict['W7']['no_XZ_ZX']['unc_' + intype] = W_unc_dict[W7_s4_min_name]
            data_dict['W7']['no_XZ_ZY']['unc_' + intype] = W_unc_dict[W7_s5_min_name]
            data_dict['W7']['no_XZ_YX']['unc_' + intype] = W_unc_dict[W7_s6_min_name]
            data_dict['W7']['no_YX_ZY']['unc_' + intype] = W_unc_dict[W7_s7_min_name]
            data_dict['W7']['no_YZ_ZY']['unc_' + intype] = W_unc_dict[W7_s8_min_name]
            data_dict['W7']['no_YZ_ZX']['unc_' + intype] = W_unc_dict[W7_s9_min_name]

            data_dict['W8']['no_XY']['unc_' + intype] = W_unc_dict[W8_s1_min_name]
            data_dict['W8']['no_YX']['unc_' + intype] = W_unc_dict[W8_s2_min_name]
            data_dict['W8']['no_XZ']['unc_' + intype] = W_unc_dict[W8_s3_min_name]
            data_dict['W8']['no_ZX']['unc_' + intype] = W_unc_dict[W8_s4_min_name]
            data_dict['W8']['no_YZ']['unc_' + intype] = W_unc_dict[W8_s5_min_name]
            data_dict['W8']['no_ZY']['unc_' + intype] = W_unc_dict[W8_s6_min_name]

    return data_dict

def analyze_rhos(filenames, rho_actuals, id='id'):
    '''Extending get_rho_from_file to include multiple files; 
    __
    Parameters:
        filenames: list of filenames to analyze
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
    __
    Returns:
        dataframe:
        - trial number
        - chi (if they exist)
        - fidelity
        - purity
        - W3 theory (adjusted for purity), W3 expt, and W3 unc
        - W5 theory (adjusted for purity), W5 expt, and W5 unc
    '''
    # initialize df
    df = pd.DataFrame()

    for i, file in tqdm(enumerate(filenames)):
        trial, rho, unc, Su, fidelity, purity, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
        print('Purity is:', purity)
        rho_actual = rho_actuals[i]
        #TODO: change back to fidelity from file after getting a new tomography
        fidelity = get_fidelity(rho_actual, rho)
        print('Fidelity is:', fidelity)
        
        print('Theoretical rho is:')
        print(np.round(rho_actual, 4))
        print('Experimental rho is:')
        print(np.round(rho, 3))
        
        #########################
        ## MINIMIZING WITNESSES
        #########################

        # calculate W3 and W5 theory
        # TODO: edit lists of witness classes to calculate W7s and W8s
        print("Minimizing witnesses for theoretical data...")
        W_T_params, W_T_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho_actual)
        print("Minimizing witnesses for adjusted theory data...")
        W_AT_params, W_AT_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=adjust_rho(rho_actual, purity))

        # calculate W3 and W5 expt
        flat_un_proj = un_proj.flatten()
        flat_un_proj_unc = un_proj_unc.flatten()
        print("Minimizing witnesses for experimental data...")
        # NOTE: do not put in uncertainties here
        W_E_params, W_E_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho)

        # check if we calculated W7s and W8s
        do_W7s_W8s = False
        if len(W_E_vals) > 15:
            do_W7s_W8s = True
        
        ##############################
        ## CALCULATING UNCERTAINTIES
        ##############################

        # NOTE: i is indexed from one because it represents a witness subscript
        W_E_unc = []
        W3_obj = sw.W3(counts=unp.uarray(flat_un_proj, flat_un_proj_unc))
        for i in range(1, 7): # W3s
            expec_val = W3_obj.expec_val(i, *W_E_params[i-1])
            W_E_unc.append(unp.std_devs(expec_val))
        
        W5_obj = sw.W5(counts=unp.uarray(flat_un_proj, flat_un_proj_unc))
        for i in range(1, 10): # W5s
            expec_val = W5_obj.expec_val(i, *W_E_params[i+5]) # offset for W3s
            W_E_unc.append(unp.std_devs(expec_val))

        if do_W7s_W8s:
            W7_obj = sw.W7(counts=unp.uarray(flat_un_proj, flat_un_proj_unc))
            for i in range(1, 109): # W7s
                expec_val = W7_obj.expec_val(i, *W_E_params[i+14]) # offset for W3s and W5s
                W_E_unc.append(unp.std_devs(expec_val))
            
            W8_obj = sw.W8(counts=unp.uarray(flat_un_proj, flat_un_proj_unc))
            for i in range(1, 37): # W8s
                expec_val = W8_obj.expec_val(i, *W_E_params[i+122]) # offset for W3s, W5s, W7s
                W_E_unc.append(unp.std_devs(expec_val))

        ##################
        ## PARSING LISTS
        ##################

        # make nested dictionaries to hold all data
        # NOTE: W5s, W7s, and W8s are all grouped by the measurements they require/exclude
        data = {
            'W3': {},
            'W5': {
                't1': {},
                't2': {},
                't3': {}
            }
        }
        if do_W7s_W8s:
            W7s_W8s_dict = {
                'W7': {
                    'no_XY_YX': {},
                    'no_XY_YZ': {},
                    'no_XY_ZX': {},
                    'no_XZ_ZX': {},
                    'no_XZ_ZY': {},
                    'no_XZ_YX': {},
                    'no_YX_ZY': {},
                    'no_YZ_ZY': {},
                    'no_YZ_ZX': {}
                },
                'W8': {
                    'no_XY': {},
                    'no_YX': {},
                    'no_XZ': {},
                    'no_ZX': {},
                    'no_YZ': {},
                    'no_ZY': {}
                }
            }
            data.update(W7s_W8s_dict)

        # Theoretical data
        data = parse_W_ls(W_T_params, W_T_vals, do_W7s_W8s, data, "T")

        # Adjusted theory
        data = parse_W_ls(W_AT_params, W_AT_vals, do_W7s_W8s, data, "AT")

        # Experimental data
        data = parse_W_ls(W_E_params, W_E_vals, do_W7s_W8s, data, "E", W_E_unc)

        print("\n------\nW3s\n------")
        print("\nAll theoretical W3s:", W_T_vals[:6])
        print("All adjusted theory W3s:", W_AT_vals[:6])
        print("All experimental W3s:", W_E_vals[:6])
        # Print names, values, and params of T, AT, E minima
        print("\nTheoretical W3 min: W3_" + data['W3']['name_T'], data['W3']['min_T'])
        print("Adjusted theory W3 min: W3_" + data['W3']['name_AT'], data['W3']['min_AT'])
        print("Experimental W3 min: W3_" + data['W3']['name_E'], data['W3']['min_E'], "+/-", data['W3']['unc_E'])
        print("\nTheoretical theta:", data['W3']['param_T'])
        print("Adjusted theory theta:", data['W3']['param_AT'])
        print("Experimental theta:", data['W3']['param_E'])

        print("\n------\nW5 t1\n------")
        print("\nAll theoretical W5 t1s:", W_T_vals[6:9])
        print("All adjusted theory W5 t1s:", W_AT_vals[6:9])
        print("All experimental W5 t1s:", W_E_vals[6:9])
        # Print names, values, and params of T, AT, E minima
        print("\nTheoretical W5 triplet 1 min: W5_" + data['W5']['t1']['name_T'], data['W5']['t1']['min_T'])
        print("Adjusted theory W5 triplet 1 min: W5_" + data['W5']['t1']['name_AT'], data['W5']['t1']['min_AT'])
        print("Experimental W5 triplet 1 min: W5_" + data['W5']['t1']['name_E'], data['W5']['t1']['min_E'], "+/-", data['W5']['t1']['unc_E'])
        print("\nTheoretical params:", data['W5']['t1']['params_T'])
        print("Adjusted theory params:", data['W5']['t1']['params_AT'])
        print("Experimental params:", data['W5']['t1']['params_E'])

        print("\n------\nW5 t2\n------")
        print("\nAll theoretical W5 t2s:", W_T_vals[9:12])
        print("All adjusted theory W5 t2s:", W_AT_vals[9:12])
        print("All experimental W5 t2s:", W_E_vals[9:12])
        # Print names, values, and params of T, AT, E minima
        print("\nTheoretical W5 triplet 2 min: W5_" + data['W5']['t2']['name_T'], data['W5']['t2']['min_T'])
        print("Adjusted theory W5 triplet 2 min: W5_" + data['W5']['t2']['name_AT'], data['W5']['t2']['min_AT'])
        print("Experimental W5 triplet 2 min: W5_" + data['W5']['t2']['name_E'], data['W5']['t2']['min_E'], "+/-", data['W5']['t2']['unc_E'])
        print("\nTheoretical params:", data['W5']['t2']['params_T'])
        print("Adjusted theory params:", data['W5']['t2']['params_AT'])
        print("Experimental params:", data['W5']['t2']['params_E'])

        print("\n------\nW5 t3\n------")
        print("\nAll theoretical W5 t3s:", W_T_vals[12:15])
        print("All adjusted theory W5 t3s:", W_AT_vals[12:15])
        print("All experimental W5 t3s:", W_E_vals[12:15])
        # Print names, values, and params of T, AT, E minima
        print("\nTheoretical W5 triplet 3 min: W5_" + data['W5']['t3']['name_T'], data['W5']['t3']['min_T'])
        print("Adjusted theory W5 triplet 3 min: W5_" + data['W5']['t3']['name_AT'], data['W5']['t3']['min_AT'])
        print("Experimental W5 triplet 3 min: W5_" + data['W5']['t3']['name_E'], data['W5']['t3']['min_E'], "+/-", data['W5']['t3']['unc_E'])
        print("\nTheoretical params:", data['W5']['t3']['params_T'])
        print("Adjusted theory params:", data['W5']['t3']['params_AT'])
        print("Experimental params:", data['W5']['t3']['params_E'])

        if do_W7s_W8s:
            # TODO: fix to reflect new groupings
            print("")
            # print("\nTheoretical W7 min:", data['W7']['min_T'])
            # print("Adjusted theory W7 min:", data['W7']['min_AT'])
            # print("Experimental W7 min:", data['W7']['min_E'], "+/-", data['W7']['unc_E'])

            # print("\nTheoretical W8 min:", data['W8']['min_T'])
            # print("Adjusted theory W8 min:", data['W8']['min_AT'])
            # print("Experimental W8 min:", data['W8']['min_E'], "+/-", data['W8']['unc_E'])

        #######################
        ## BUILDING DATAFRAME
        #######################

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
        new_df_row = pd.DataFrame.from_dict([min_data])

        # Insert columns for other important data that isn't the witness minima
        new_df_row.insert(0, 'trial', trial)
        new_df_row.insert(1, 'fidelity', fidelity)
        new_df_row.insert(2, 'purity', purity)
        new_df_row['UV_HWP'] = angles[0]
        new_df_row['QP'] = angles[1]
        new_df_row['B_HWP'] = angles[2]
        
        if chi is not None:
            adj_fidelity = get_fidelity(adjust_rho(rho_actual, purity), rho)
            new_df_row.insert(1, 'chi', chi)
            new_df_row.insert(4, 'AT_fidelity', adj_fidelity)

        # Concatenate new row to the multifile dataframe
        df = pd.concat([df, new_df_row])

    # Save df
    print('saving dataframe...')
    df.to_csv(join(DATA_PATH, f'analysis_{id}.csv'))

def make_plots_E0(dfname, fig_title):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    Parameters:
        dfname: str, name of df to read in
    '''
    print("plotting...")
    # read in df
    df = pd.read_csv(join(DATA_PATH, dfname))
    fig, ax = plt.subplots(figsize = (8, 8))
    chi = df['chi'].to_numpy()

    # TODO: Fix purity and fidelity plotting code commented below
    # if plot_purity:
    #     purity = df['purity'].to_numpy()
    #     ax[1,0].scatter(chi, purity, label='Purity', color='gold')

    # if plot_fidelity:
    #     fidelity = df['fidelity'].to_numpy()
    #     adj_fidelity = df['AT_fidelity'].to_numpy()
    #     ax[1,1].scatter(chi, fidelity, label='Fidelity', color='turquoise')
    #     ax[1,2].plot(chi, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

    # extract witness values
    W3_min_T = df['W3_min_T'].to_numpy()
    W3_min_AT = df['W3_min_AT'].to_numpy()
    W3_min_E = df['W3_min_E'].to_numpy()
    W3_unc = df['W3_unc_E'].to_numpy()

    W5_min_T = df[['W5_t1_min_T', 'W5_t2_min_T', 'W5_t3_min_T']].min(axis=1).to_numpy()
    W5_min_AT = df[['W5_t1_min_AT', 'W5_t2_min_AT', 'W5_t3_min_AT']].min(axis=1).to_numpy()
    W5_min_E = df[['W5_t1_min_E', 'W5_t2_min_E', 'W5_t3_min_E']].min(axis=1).to_numpy()
    W5_name_E = df[['W5_t1_min_E', 'W5_t2_min_E', 'W5_t3_min_E']].idxmin(axis=1)
    # retrieve the correct uncertainty depending on which W5 triplet gave the best minimum
    W5_unc = np.where(W5_name_E == 'W5_t1_min_E', df['W5_t1_unc_E'], np.where(W5_name_E == 'W5_t2_min_E', df['W5_t2_unc_E'], df['W5_t3_unc_E']))

    # plot curves for T and AT
    def sinsq(x, a, b, c, d):
        return a*np.sin(b*np.deg2rad(x) + c)**2 + d

    """
    NOTE: popt is a list of optimal values for chi so that the sum of the
            squared residuals of f(xdata, *popt) - ydata is minimized, while pcov is a
            matrix representing the estimated approximate covariance of popt
    """
    popt_W3_T, pcov_W3_T = curve_fit(sinsq, chi, W3_min_T, maxfev = 10000)
    popt_W3_AT, pcov_W3_AT = curve_fit(sinsq, chi, W3_min_AT, maxfev = 10000)
    #print('popt_W3 are:', popt_W3_AT)
    popt_W5_T, pcov_W5_T = curve_fit(sinsq, chi, W5_min_T, maxfev = 10000)
    popt_W5_AT, pcov_W5_AT = curve_fit(sinsq, chi, W5_min_AT, maxfev = 10000)
    
    chi_ls = np.linspace(min(chi), max(chi), 1000)

    ax.plot(chi_ls, sinsq(chi_ls, *popt_W3_T), label='$W_T^3$', color='navy')
    ax.plot(chi_ls, sinsq(chi_ls, *popt_W3_AT), label='$W_{AT}^3$', linestyle='dashed', color='blue')
    ax.errorbar(chi, W3_min_E, yerr=W3_unc, fmt='o', color='slateblue', markersize=10)

    ax.plot(chi_ls, sinsq(chi_ls, *popt_W5_T), label="$W_T^5$", color='crimson')
    ax.plot(chi_ls, sinsq(chi_ls, *popt_W5_AT), label="$W_{AT}^5$", linestyle='dashed', color='red')
    ax.errorbar(chi, W5_min_E, yerr=W5_unc, fmt='o', color='salmon', markersize=10)

    ax.set_ylabel('Witness value', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(ncol=2, fontsize=20)
    ax.set_xlabel('$\chi$ (deg)', fontsize=20)
    ax.axhline(y=0, color='black')

    # TODO: fix
    #"Entangled State Witnessed by 3rd W5 Triplet (2025)"
    plt.suptitle(fig_title, fontsize=20)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'{STATE_ID}_trial{TRIAL}_2025_analysis.pdf'))
    plt.show()

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho(state, chi):
    '''
    Calculates the density matrix (rho) for a given set of parameters (chi) for Stuart's states
    
    Parameters:
        state (string): The name of the state we are analyzing
        chi (float): The parameter chi
    
    Returns:
        rho (numpy.ndarray): The density matrix
    '''
    # Define kets and bell states in vector form 
    H = ket([1,0])
    V = ket([0,1])
    R = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
    L = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
    D = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
    A = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])
    
    ## The following state(s) are an attempt to find new positive W3 negative W5 states.
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

    if state == 'hr_negpi_6_vl_target':
        phi = np.cos(chi/2) * np.kron(H, R) + np.exp(-1j * np.pi/6) * np.sin(chi/2) * np.kron(V, L)

    if state == 'hr_negpi_6_vl':
        theta = np.arctan(np.sqrt(.225/.275))
        R = np.cos(theta) * H + 1j * np.sin(theta) * V
        L = np.sin(theta) * H - 1j * np.cos(theta) * V
        phi = np.cos(chi/2) * np.kron(H, R) + np.exp(-1j * np.pi/6) * np.sin(chi/2) * np.kron(V, L)

    if state =='cosHA_minusphasesinVD':
        phi = np.cos(chi/2) * np.kron(H, A) + np.exp(-1j * 1.27) * np.sin(chi/2) * np.kron(V,D)

    if state == 'cosHH_minusisinVV':
        phi = np.cos(chi/2) * np.kron(H, H) - 1j * np.sin(chi/2) * np.kron(V, V)

    if state == 'cosHV_minusisinVH':
        phi = np.cos(chi/2) * np.kron(H, V) - 1j * np.sin(chi/2) * np.kron(V, H)
    
    # create rho and return it
    rho = phi @ phi.conj().T
    return rho

def main(filepath):
    try:
        with open(filepath, 'r') as file:
            content = []
            for line in file:
                content.append(line.strip())
            
            data_path, trial, fig_title, state_id = content[:4]
            # for pure states, the txt file won't contain names and probs
            offset = 0
            names = []
            probs = []
            if "mix" in state_id:
                names = content[4].split(" ")
                probs = list(map(float, content[5].split(" ")))
                # add an offset to the line indices
                offset = 2

            chis_range = content[4 + offset]
            if chis_range.lower() == "y":
                chis = np.linspace(0.001, np.pi/2, 6)
            else:
                chis_str = content[5 + offset]
                if chis_str == "":
                    chis = [np.pi/2]
                else:
                    chis = [eval(chis_str)]
            return data_path, trial, fig_title, state_id, names, probs, chis

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")

if __name__ == '__main__':
    # check if a text file was passed as an argument
    if len(sys.argv) > 1:
            file_path = sys.argv[1]
            inputs = main(file_path)
            DATA_PATH, TRIAL, FIG_TITLE, STATE_ID, names, probs, chis = inputs
    else:
        # set path & other user input variables
        current_path = dirname(abspath(__file__))
        DATA_PATH = input("Input the path to the lowest-level directory that your data file is in: ")
        TRIAL = int(input("Trial number: "))
        FIG_TITLE = input("Input the desired title for the generated figure: ")
        STATE_ID = input("State name (must contain 'mix' if analyzing a mixed state): ")

        if "mix" in STATE_ID:
            mixed_states = input("Input the names of the individual states that compose your mixed state, separated by spaces: ")
            names = mixed_states.split(" ")
            probabilities = input("Input the probabilities of each respective state, separated by spaces: ")
            probs = list(map(float, probabilities.split(" ")))

        chis_range = input("Are you analyzing the full range of chi values? [y/n]: ")
        if chis_range.lower() == "y":
            chis = np.linspace(0.001, np.pi/2, 6)
        else:
            chis_str = input("Which chi value do you want to test (must be in radians; e.g. 'np.pi/2')?\nType nothing and hit ENTER to assign a default value of pi/2 radians: ")
            if chis_str == "":
                chis = [np.pi/2]
            else:
                chis = [eval(chis_str)]

    # ignore escape sequences in FIG_TITLE to allow parsing LaTeX
    FIG_TITLE = FIG_TITLE.encode('utf-8').decode('unicode_escape')
    rho_actuals = []
    filenames = []
    rho_actuals = []

    # Obtain the density matrix for each state
    for chi in chis:
        if "mix" in STATE_ID:
            rho_actuals.append(gen_mixed_state(names, probs, chi))
        else:
            rho_actuals.append(get_theo_rho(STATE_ID, chi))
        filenames.append(f"rho_({STATE_ID}-{np.rad2deg(chi)}-{TRIAL}).npy")

    # analyze rho files
    analyze_rhos(filenames, rho_actuals, id=STATE_ID)
    if len(filenames) > 1:
        make_plots_E0(f'analysis_{STATE_ID}.csv', FIG_TITLE)