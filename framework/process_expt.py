"""
Authors: Lev G., Isabel G.
Last updated: 6/16/2025

This file reads and processes experimentally collected density matrices using functionality from
states_and_witnesses.py and operations.py, so make sure to either copy those files to your directory
or update the path variables to import them. This file no longer depends on rho_methods.py or
sample_rho.py.

To run this file, first change the figure title as desired on line 723, then run this file and fill
in the user inputs when prompted in your command line.
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
    Parameters:
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

def adjust_rho(rho, expt_purity):
    ''' Adjusts theo density matrix to account for experimental impurity
        Multiplies unwanted experimental impurities (top right bottom right block) by expt purity
        to account for non-entangled particles in our system '''
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
    Returns: df with:
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
        
        #########################
        ## MINIMIZING WITNESSES
        #########################
        
        # Redirects print statements to an output file.
        # NOTE: MUST REMOVE TO PRINT IN TERMINAL
        import sys
        original_stdout = sys.stdout
        sys.stdout = open('chi0.001_min_output.txt', 'w')

        # calculate W and W' theory
        print("Minimizing witnesses for theoretical data...")
        W_T_params, W_T_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho_actual)
        print("Minimizing witnesses for adjusted theory data...")
        W_AT_params, W_AT_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=adjust_rho(rho_actual, purity))

        # calculate W and W' expt
        flat_un_proj = un_proj.flatten()
        flat_un_proj_unc = un_proj_unc.flatten()
        print("Minimizing witnesses for experimental data...")
        # NOTE: do not put in uncertainties here
        W_E_params, W_E_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho)

        # close output file
        sys.stdout.close()
        sys.stdout = original_stdout

        # check if we calculated W7s and W8s
        do_W7s_W8s = False
        if len(W_E_vals) > 15:
            do_W7s_W8s = True
        
        ##############################
        ## CALCULATING UNCERTAINTIES
        ##############################

        # NOTE: i is indexed from one because it represents a witness superscript
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
        print("\nTheoretical W3 min:", data['W3']['min_T'])
        print("Adjusted theory W3 min:", data['W3']['min_AT'])
        print("Experimental W3 min:", data['W3']['min_E'], "+/-", data['W3']['unc_E'])
        # Initialize W3 objects with the experimental and theoretical rhos
        W3_E_obj = sw.W3(rho=rho)
        W3_T_obj = sw.W3(rho=rho_actual)
        # Save W3's min name and params
        W3_idx_T = data['W3']['name_T']
        W3_param_T = data['W3']['param_T']
        W3_idx_E = data['W3']['name_E']
        W3_param_E = data['W3']['param_E']
        # Print variables and expectation values
        print("\nFor theoretical data, the most minimized W3 was W3_" + W3_idx_T)
        print("It had the following theta:", W3_param_T)
        print("Theoretical W3 min val based on theoretical params:", W3_T_obj.expec_val(int(W3_idx_T), *W3_param_T))
        print("Experimental W3 min val based on theoretical params:", W3_E_obj.expec_val(int(W3_idx_T), *W3_param_T))
        print("\nFor experimental data, the most minimized W3 was W3_" + W3_idx_E)
        print("It had the following theta:", W3_param_E)
        print("Theoretical W3 min val based on experimental params:", W3_T_obj.expec_val(int(W3_idx_E), *W3_param_E))
        print("Experimental W3 min val based on experimental params:", W3_E_obj.expec_val(int(W3_idx_E), *W3_param_E))

        print("\n------\nW5 t1\n------")
        print("\nAll theoretical W5s:", W_T_vals[6:9])
        print("All adjusted theory W5s:", W_AT_vals[6:9])
        print("All experimental W5s:", W_E_vals[6:9])
        print("\nTheoretical W5 triplet 1 min:", data['W5']['t1']['min_T'])
        print("Adjusted theory W5 triplet 1 min:", data['W5']['t1']['min_AT'])
        print("Experimental W5 triplet 1 min:", data['W5']['t1']['min_E'], "+/-", data['W5']['t1']['unc_E'])
        # Initialize W5 objects with the experimental and theoretical rhos
        W5_E_obj = sw.W5(rho=rho)
        W5_T_obj = sw.W5(rho=rho_actual)
        # Save W5 t1's experimental min name and params
        W5t1_idx_T = data['W5']['t1']['name_T']
        W5t1_params_T = data['W5']['t1']['params_T']
        W5t1_idx_E = data['W5']['t1']['name_E']
        W5t1_params_E = data['W5']['t1']['params_E']
        # Print variables and expectation values
        print("\nFor theoretical data, the most minimized W5 t1 was W5_" + W5t1_idx_T)
        print("It had the following params:", W5t1_params_T)
        print("Theoretical W5 t1 min val based on theoretical params:", W5_T_obj.expec_val(int(W5t1_idx_T), *W5t1_params_T))
        print("Experimental W5 t1 min val based on theoretical params:", W5_E_obj.expec_val(int(W5t1_idx_T), *W5t1_params_T))
        print("\nFor experimental data, the most minimized W5 t1 was W5_" + W5t1_idx_E)
        print("It had the following params:", W5t1_params_E)
        print("Theoretical W5 t1 min val based on experimental params:", W5_T_obj.expec_val(int(W5t1_idx_E), *W5t1_params_E))
        print("Experimental W5 t1 min val based on experimental params:", W5_E_obj.expec_val(int(W5t1_idx_E), *W5t1_params_E))

        print("\n------\nW5 t2\n------")
        print("\nAll theoretical W5s:", W_T_vals[9:12])
        print("All adjusted theory W5s:", W_AT_vals[9:12])
        print("All experimental W5s:", W_E_vals[9:12])
        print("\nTheoretical W5 triplet 2 min:", data['W5']['t2']['min_T'])
        print("Adjusted theory W5 triplet 2 min:", data['W5']['t2']['min_AT'])
        print("Experimental W5 triplet 2 min:", data['W5']['t2']['min_E'], "+/-", data['W5']['t2']['unc_E'])
        # Save W5 t2's experimental min name and params
        W5t2_idx_T = data['W5']['t2']['name_T']
        W5t2_params_T = data['W5']['t2']['params_T']
        W5t2_idx_E = data['W5']['t2']['name_E']
        W5t2_params_E = data['W5']['t2']['params_E']
        # Print variables and expectation values
        print("\nFor theoretical data, the most minimized W5 t2 was W5_" + W5t2_idx_T)
        print("It had the following params:", W5t2_params_T)
        print("Theoretical W5 t2 min val based on theoretical params:", W5_T_obj.expec_val(int(W5t2_idx_T), *W5t2_params_T))
        print("Experimental W5 t2 min val based on theoretical params:", W5_E_obj.expec_val(int(W5t2_idx_T), *W5t2_params_T))
        print("\nFor experimental data, the most minimized W5 t2 was W5_" + W5t2_idx_E)
        print("It had the following params:", W5t2_params_E)
        print("Theoretical W5 t2 min val based on experimental params:", W5_T_obj.expec_val(int(W5t2_idx_E), *W5t2_params_E))
        print("Experimental W5 t2 min val based on experimental params:", W5_E_obj.expec_val(int(W5t2_idx_E), *W5t2_params_E))

        print("\n------\nW5 t3\n------")
        print("\nAll theoretical W5s:", W_T_vals[12:15])
        print("All adjusted theory W5s:", W_AT_vals[12:15])
        print("All experimental W5s:", W_E_vals[12:15])
        print("\nTheoretical W5 triplet 3 min:", data['W5']['t3']['min_T'])
        print("Adjusted theory W5 triplet 3 min:", data['W5']['t3']['min_AT'])
        print("Experimental W5 triplet 3 min:", data['W5']['t3']['min_E'], "+/-", data['W5']['t3']['unc_E'])
        # Save W5 t3's experimental min name and params
        W5t3_idx_T = data['W5']['t3']['name_T']
        W5t3_params_T = data['W5']['t3']['params_T']
        W5t3_idx_E = data['W5']['t3']['name_E']
        W5t3_params_E = data['W5']['t3']['params_E']
        # Print variables and expectation values
        print("\nFor theoretical data, the most minimized W5 t3 was W5_" + W5t3_idx_T)
        print("It had the following params:", W5t3_params_T)
        print("Theoretical W5 t3 min val based on theoretical params:", W5_T_obj.expec_val(int(W5t3_idx_T), *W5t3_params_T))
        print("Experimental W5 t3 min val based on theoretical params:", W5_E_obj.expec_val(int(W5t2_idx_T), *W5t3_params_T))
        print("\nFor experimental data, the most minimized W5 t3 was W5_" + W5t3_idx_E)
        print("It had the following params:", W5t3_params_E)
        print("Theoretical W5 t3 min val based on experimental params:", W5_T_obj.expec_val(int(W5t3_idx_E), *W5t3_params_E))
        print("Experimental W5 t3 min val based on experimental params:", W5_E_obj.expec_val(int(W5t2_idx_E), *W5t3_params_E))

        if do_W7s_W8s:
            # TODO: fix to reflect new groupings
            print("")
            # print("\nTheoretical W7 min:", data['W7']['min_T'])
            # print("Adjusted theory W3 min:", data['W7']['min_AT'])
            # print("Experimental W3 min:", data['W7']['min_E'], "+/-", data['W7']['unc_E'])

            # print("\nTheoretical W3 min:", data['W8']['min_T'])
            # print("Adjusted theory W3 min:", data['W8']['min_AT'])
            # print("Experimental W3 min:", data['W8']['min_E'], "+/-", data['W8']['unc_E'])

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
        
        if eta is not None and chi is not None:
            adj_fidelity = get_fidelity(adjust_rho(rho_actual, purity), rho)
            new_df_row.insert(1, 'eta', eta)
            new_df_row.insert(2, 'chi', chi)
            new_df_row.insert(5, 'AT_fidelity', adj_fidelity)

        # Concatenate new row to the multifile dataframe
        df = pd.concat([df, new_df_row])

    # Save df
    print('saving dataframe...')
    df.to_csv(join(DATA_PATH, f'analysis_{id}.csv'))

def make_plots_E0(dfname):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    Parameters:
        dfname: str, name of df to read in
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
        W3_min_T = df_eta['W3_min_T'].to_numpy()
        W3_min_AT = df_eta['W3_min_AT'].to_numpy()
        W3_min_E = df_eta['W3_min_E'].to_numpy()
        W3_unc = df_eta['W3_unc_E'].to_numpy()

        W5_min_T = df_eta[['W5_t1_min_T', 'W5_t2_min_T', 'W5_t3_min_T']].min(axis=1).to_numpy()
        W5_min_AT = df_eta[['W5_t1_min_AT', 'W5_t2_min_AT', 'W5_t3_min_AT']].min(axis=1).to_numpy()
        W5_min_E = df_eta[['W5_t1_min_E', 'W5_t2_min_E', 'W5_t3_min_E']].min(axis=1).to_numpy()
        W5_best_E = df_eta[['W5_t1_min_E', 'W5_t2_min_E', 'W5_t3_min_E']].idxmin(axis=1)
        W5_unc = np.where(W5_best_E == 'W5_t1_min_E', df_eta['W5_t1_unc_E'], np.where(W5_best_E == 'W5_t2_min_E', df_eta['W5_t2_unc_E'], df_eta['W5_t3_unc_E']))

        # plot curves for T and AT
        def sinsq(x, a, b, c, d):
            return a*np.sin(b*np.deg2rad(x) + c)**2 + d

        """
        NOTE: popt is a list of optimal values for eta and chi so that the sum of the
              squared residuals of f(xdata, *popt) - ydata is minimized, while pcov is a
              matrix representing the estimated approximate covariance of popt
        """
        popt_W3_T_eta, pcov_W3_T_eta = curve_fit(sinsq, chi_eta, W3_min_T, maxfev = 10000)
        popt_W3_AT_eta, pcov_W3_AT_eta = curve_fit(sinsq, chi_eta, W3_min_AT, maxfev = 10000)
        #print('popt_W are:', popt_W_AT_eta)
        popt_W5_T_eta, pcov_W5_T_eta = curve_fit(sinsq, chi_eta, W5_min_T, maxfev = 10000)
        popt_W5_AT_eta, pcov_W5_AT_eta = curve_fit(sinsq, chi_eta, W5_min_AT, maxfev = 10000)
        
        chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W3_T_eta), label='$W_T^3$', color='navy')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W3_AT_eta), label='$W_{AT}^3$', linestyle='dashed', color='blue')
        ax.errorbar(chi_eta, W3_min_E, yerr=W3_unc, fmt='o', color='slateblue', markersize=10)

        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W5_T_eta), label="$W_T^5$", color='crimson')
        ax.plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W5_AT_eta), label="$W_{AT}^5$", linestyle='dashed', color='red')
        ax.errorbar(chi_eta, W5_min_E, yerr=W5_unc, fmt='o', color='salmon', markersize=10)
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
            W3_min_T = df_eta['W3_min_T'].to_numpy()
            W3_min_AT = df_eta['W3_min_AT'].to_numpy()
            W3_min_E = df_eta['W3_min_E'].to_numpy()
            W3_unc = df_eta['W3_unc_E'].to_numpy()

            W5_min_T = df_eta[['W5_t1_min_T', 'W5_t2_min_T', 'W5_t3_min_T']].min(axis=1).to_numpy()
            W5_min_AT = df_eta[['W5_t1_min_AT', 'W5_t2_min_AT', 'W5_t3_min_AT']].min(axis=1).to_numpy()
            W5_min_E = df_eta[['W5_t1_min_E', 'W5_t2_min_E', 'W5_t3_min_E']].min(axis=1).to_numpy()
            W5_best_E = df_eta[['W5_t1_min_E', 'W5_t2_min_E', 'W5_t3_min_E']].idxmin(axis=1)
            W5_unc = np.where(W5_best_E == 'W5_t1_min_E', df_eta['W5_t1_unc_E'], np.where(W5_best_E == 'W5_t2_min_E', df_eta['W5_t2_unc_E'], df_eta['W5_t3_unc_E']))

            # plot curves for T and AT
            def sinsq(x, a, b, c, d):
                return a*np.sin(b*np.deg2rad(x) + c)**2 + d
            popt_W3_T_eta, pcov_W3_T_eta = curve_fit(sinsq, chi_eta, W3_min_T)
            popt_W3_AT_eta, pcov_W3_AT_eta = curve_fit(sinsq, chi_eta, W3_min_AT)

            popt_W5_T_eta, pcov_W5_T_eta = curve_fit(sinsq, chi_eta, W5_min_T)
            popt_W5_AT_eta, pcov_W5_AT_eta = curve_fit(sinsq, chi_eta, W5_min_AT)

            chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W3_T_eta), label='$W_T^3$', color='navy')
            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W3_AT_eta), label='$W_{AT}^3$', linestyle='dashed', color='blue')
            ax[i].errorbar(chi_eta, W3_min_E, yerr=W3_unc, fmt='o', color='slateblue')

            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W5_T_eta), label="$W_T^5$", color='crimson')
            ax[i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W5_AT_eta), label="$W_{AT}^5$", linestyle='dashed', color='red')
            ax[i].errorbar(chi_eta, W5_min_E, yerr=W5_unc, fmt='o', color='salmon')

            ax[i].set_title('$\eta = 30\degree$', fontsize=33)
            ax[i].set_ylabel('Witness value', fontsize=31)
            ax[i].tick_params(axis='both', which='major', labelsize=25)
            ax[i].legend(ncol=2, fontsize=25)
            ax[i].set_xlabel('$\chi$', fontsize=31)
            # ax[1,i].set_ylabel('Value', fontsize=31)
            # ax[1,i].legend()

    plt.suptitle("Min. Witness Values for $\cos(\\frac{\chi}{2}) |HD \u27E9 + \sin(\\frac{\chi}{2}) e^{\\frac{-i\pi}{3}} |VA \u27E9$", fontsize=20)
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'{STATE_ID}_trial{TRIAL}.pdf'))
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
    R = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
    L = ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
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

    if state == 'hr_negpi_6_vl':
        phi = np.cos(chi/2) * np.kron(H, R) + np.exp(-1j * np.pi/6) * np.sin(chi/2) * np.kron(V, L)

    if state =='cosHA_minusphasesinVD':
        phi = np.cos(chi/2) * np.kron(H, A) + np.exp(-1j * 1.27) * np.sin(chi/2) * np.kron(V,D)

    if state =='HAVD_mix':
        psi_3 = np.cos(chi/2) * np.kron(H, A) - 1j * np.sin(chi/2) * np.kron(V, D)
        psi_4 = np.cos(chi/2) * np.kron(H, R) - 1j * np.sin(chi/2) * np.kron(V, L)
        phi = 0.65 * np.outer(psi_3, psi_3) + 0.35 * np.outer(psi_4, psi_4)
    
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
        chis_var = input("Would you like to test variations in the minima using the same chi multiple times? [y/n]: ")
        if chis_var.lower() == "y":
            chis = [np.pi/2]*10
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
    if len(filenames) > 1:
        make_plots_E0(f'analysis_{STATE_ID}.csv')