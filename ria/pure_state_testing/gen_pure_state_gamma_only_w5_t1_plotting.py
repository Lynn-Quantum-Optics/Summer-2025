"""
Authors: Ria H.
Last updated: 7/17/2025

This file
"""

print("initializing...")

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
import os

from uncertainties import ufloat
from uncertainties import unumpy as unp

# import Isabel & Brayden's files
import states_and_witnesses as sw
import operations as op


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
    data_dict['W3']['params_' + intype] = W_params_dict[W3_min_name]

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

def print_witness_data(group, W_vals, dict, list_slice, not_measured=""):
    """
    A generic function for printing the minimized witness data.
    Parameters:
        group: the name of the group the witnesses belong to, e.g. "W5 t1"
        W_vals: contains the lists of theoretical witness minima, adjusted theoretical witness
                minima, and experimental witness minima, in that order
        dict: the dictionary or subdictionary containing the minimum data
        list_slice: a slice object containing the range of values in the minimization
                    output lists that pertain to a certain group. e.g. for W5 t1,
                    slice(6, 9) produces [6:9]
        not_measured: if we want to provide additional info about which measurements
                        are excluded by a group of witnesses, this will display
                        that info in the header for the group
    """
    # check if excluded measurements were given
    if not_measured == "":
        header = group
    else:
        header = group + ": " + not_measured

    # get the witness class from the group name
    w_class = group[:2]

    print(f"\n------\n{header}\n------")
    print("\nAll theoretical minima:", W_vals[0][list_slice])

    # print names, values, and params of T, AT, E minima
    print(f"\nTheoretical {group} min: {w_class}_" + dict['name_T'], dict['min_T'])
    print("Theoretical params:", dict['params_T'])


def analyze_rhos(rho_actuals, id, directory):
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
    do_W7s_W8s = False

    for rho_actual in rho_actuals:
            
        #########################
        ## MINIMIZING WITNESSES
        #########################

        # calculate W3 and W5 theory
        # TODO: edit lists of witness classes to calculate W7s and W8s
        print("Minimizing witnesses for theoretical data...")
        W_T_params, W_T_vals = op.minimize_witnesses([sw.W3, sw.W5], rho=rho_actual)
        

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

        # Theoretical data
        data = parse_W_ls(W_T_params, W_T_vals, do_W7s_W8s, data, "T")

        # Put all lists of minimized witness values into one list for more elegant passing
        all_W_vals = [W_T_vals]

        # W3s
        # print_witness_data("W3", all_W_vals, data['W3'], slice(6))
        # W5s
        print_witness_data("W5 t1", all_W_vals, data['W5']['t1'], slice(6, 9))
        # print_witness_data("W5 t2", all_W_vals, data['W5']['t2'], slice(9, 12))
        # print_witness_data("W5 t3", all_W_vals, data['W5']['t3'], slice(12, 15))


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
 
        # Concatenate new row to the multifile dataframe
        df = pd.concat([df, new_df_row])

    # Save df
    print('saving dataframe...')
    df.to_csv(f'{directory}/{id}_analysis.csv')

def make_plots_E0(dfname, fig_title, chi_vals, directory):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    Parameters:
        dfname: str, name of df to read in
    '''
    print("plotting...")
    # read in df
    df = pd.read_csv(f"{directory}/{dfname}")
    fig, ax = plt.subplots(figsize = (8, 8))
    chi = chi_vals

    # extract witness values
    W3_min_T = df['W3_min_T'].to_numpy()

    # W5_min_T = df[['W5_t1_min_T', 'W5_t2_min_T', 'W5_t3_min_T']].min(axis=1).to_numpy()
    W5_min_T1 = df['W5_t1_min_T']

    # plot curves for T and AT
    def sinsq(x, a, b, c, d):
        return a*np.sin(b*np.deg2rad(x) + c)**2 + d

    """
    NOTE: popt is a list of optimal values for chi so that the sum of the
            squared residuals of f(xdata, *popt) - ydata is minimized, while pcov is a
            matrix representing the estimated approximate covariance of popt
    """
    # popt_W3_T, pcov_W3_T = curve_fit(sinsq, chi, W3_min_T, maxfev = 10000)
    # popt_W5_T, pcov_W5_T = curve_fit(sinsq, chi, W5_min_T, maxfev = 10000)
    
    # chi_ls = np.linspace(min(chi), max(chi), 1000)

    ax.plot(chi_vals, W3_min_T, label='$W_T^3$', color='navy')

    ax.plot(chi_vals, W5_min_T1, label="$W_T^5$", color='crimson')

    ax.set_ylabel('Witness value', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.legend(ncol=2, fontsize=20)
    ax.set_xlabel('$\chi$ (deg)', fontsize=20)
    ax.axhline(y=0, color='black')

    fig.suptitle(fig_title, fontsize=20)
    fig.tight_layout()
    fig.savefig(f'{directory}/{fig_title}_analysis_plotted.png')
    # fig.show()

def ket(data):
    return np.array(data, dtype=complex).reshape(-1,1)

def get_theo_rho(state, chi, gamma):
    '''
    Calculates the density matrix (rho) for a given set of parameters (chi) for Stuart's states
    
    Parameters:
        state (string): The name of the state we are analyzing
        chi (float): The parameter chi
        gamma: the phase shift
    
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

    splitState = state.split("_")
    basis1 = splitState[0][0]
    basis2 = splitState[0][1]
    basis1_perp = splitState[1][0]
    basis2_perp = splitState[1][1]
    
    phi = np.cos(chi/2) * np.kron(eval(basis1), eval(basis2)) + np.exp(1j*gamma) * np.sin(chi/2) * np.kron(eval(basis1_perp), eval(basis2_perp))

    # create rho and return it
    rho = phi @ phi.conj().T
    return rho


def oneRun(ID, gamma):
    """ ID: a state ID of the form HA_VD. A string.
        Gamma: a float, of the form np.pi/3
    """
    # set path & other user input variables
    STATE_ID = f"{ID}_{gamma}"

    chis = np.linspace(0.001, np.pi/2, 6)

    rho_actuals = []

    # Obtain the density matrix for each state
    for chi in chis:
        rho_actuals.append(get_theo_rho(ID, chi, gamma))

    # analyze rho files
    analyze_rhos(rho_actuals, id=STATE_ID, directory=ID)
    make_plots_E0(f'{STATE_ID}_analysis.csv', STATE_ID, chis, directory=ID)

if __name__ == '__main__':
    IDs = ["HA_VD", "HD_VA", "HH_VV", "HV_VH", "HR_VL", "HL_VR", "DD_AA", "DA_AD", "DR_AL", "DL_AR", "DH_AV", "DV_AH", "RR_LL", "RL_LR", "RD_LA", "RA_LD", "RH_LV", "RV_LH"]
    for ID in IDs:
        gammas = np.linspace(0.001, 2*np.pi, 12, endpoint=False)

        directory = f'./{ID}'
        if os.path.isdir(directory):
            print('Output folder already exists')
            quit()
        else:
            os.mkdir(directory)

        for gamma in gammas:
            oneRun(ID, gamma)