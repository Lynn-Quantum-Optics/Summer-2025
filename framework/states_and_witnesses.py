import numpy as np
import operations as op

#######################
## QUANTUM STATES
#######################

# Single-qubit States
H = op.ket([1,0])
V = op.ket([0,1])
R = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
L = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
D = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
A = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])

### Jones/Column Vectors & Density Matrices ###
HH = op.ket([1, 0, 0, 0])
HV = op.ket([0, 1, 0, 0])
VH = op.ket([0, 0, 1, 0])
VV = op.ket([0, 0, 0, 1])
HH_RHO = op.get_rho(HH)
HV_RHO = op.get_rho(HV)
VH_RHO = op.get_rho(VH)
VV_RHO = op.get_rho(VV)

### Bell States & Density Matrices ###
PHI_P = (HH + VV)/np.sqrt(2)
PHI_M = (HH - VV)/np.sqrt(2)
PSI_P = (HV + VH)/np.sqrt(2)
PSI_M = (HV - VH)/np.sqrt(2)
PHI_P_RHO = op.get_rho(PHI_P)
PHI_M_RHO = op.get_rho(PHI_M)
PSI_P_RHO = op.get_rho(PSI_P)
PSI_M_RHO = op.get_rho(PSI_M)

### Eritas's States (see spring 2023 writeup) ###
def E0_PSI(eta, chi, rho = False):
    """
    Returns the state of form cos(eta)PSI_P + e^(i*chi)*sin(eta)PSI_M
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = np.cos(eta)*PSI_P + np.exp(chi*1j)*np.sin(eta)*PSI_M

    if rho:
        return op.get_rho(state)
    return state

def E0_PHI(eta, chi, rho = False):
    """
    Returns the state of form cos(eta)PHI_P + e^(i*chi)*sin(eta)PHI_M
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = np.cos(eta)*PHI_P + np.exp(chi*1j)*np.sin(eta)*PHI_M
    if rho:
        return op.get_rho(state)
    return state

def E1(eta, chi, rho = False):
    """
    Returns the state of the form 
    1/sqrt(2) * (cos(eta)*(PSI_P + iPSI_M) + e^(i*chi)*sin(eta)*(PHI_P + iPHI_M))
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = 1/np.sqrt(2) * (np.cos(eta)*(PSI_P + PSI_M*1j) + np.sin(eta)*np.exp(chi*1j)*(PHI_P + PHI_M*1j))
    if rho:
        return op.get_rho(state)
    return state


### Amplitude Damped States ###
def ADS(gamma):
    """
    Returns the amplitude damped state with parameter gamma
    """
    return np.array([[.5, 0, 0, .5*np.sqrt(1-gamma)], 
                     [0, 0, 0, 0], [0, 0, .5*gamma, 0], 
                     [.5*np.sqrt(1-gamma), 0, 0, .5-.5*gamma]])

### Sample state to illustrate power of W5 over W3 ###
def sample_state(phi):
    """
    Returns a state with parameter phi
    """
    ex1 = np.cos(phi)*np.kron(H,D) - np.sin(phi)*np.kron(V,A)
    return op.get_rho(ex1)

##################
## MATRICES
##################

### Pauli Matrices ###
PAULI_X = np.array([[0,1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1,0], [0,-1]])
IDENTITY = np.array([[1,0], [0,1]])

# Get tensor'd Pauli matrices
PAULI = [np.kron(IDENTITY, IDENTITY), np.kron(IDENTITY, PAULI_X), np.kron(IDENTITY, PAULI_Y),
            np.kron(IDENTITY, PAULI_Z), np.kron(PAULI_X, IDENTITY), np.kron(PAULI_X, PAULI_X),
            np.kron(PAULI_X, PAULI_Y), np.kron(PAULI_X, PAULI_Z), np.kron(PAULI_Y, IDENTITY),
            np.kron(PAULI_Y, PAULI_X), np.kron(PAULI_Y, PAULI_Y), np.kron(PAULI_Y, PAULI_Z), 
            np.kron(PAULI_Z, IDENTITY), np.kron(PAULI_Z, PAULI_X), np.kron(PAULI_Z, PAULI_Y), 
            np.kron(PAULI_Z, PAULI_Z)
        ]

### Rotation Matrices ###
#
# Params: 
#  theta - the angle to rotate by
def R_z(theta):
    return np.array([[np.cos(theta/2) - np.sin(theta/2)*1j, 0], 
                    [0, np.cos(theta/2) + np.sin(theta/2)*1j]])

def R_x(theta):
    return np.array([[np.cos(theta/2), np.sin(theta/2)*-1j],
                    [np.sin(theta/2)*-1j, np.cos(theta/2)]])

def R_y(theta):
    return np.array([[np.cos(theta/2), -(np.sin(theta/2))],
                    [np.sin(theta/2), (np.cos(theta/2))]])


##########################################
##        ENTANGLEMENT WITNESSES        ## 
##########################################

## Keep track of count indices
# int to str, i.e. COUNT[0] = 'HH'
COUNTS = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 
          'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 
          'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL']
# str to int, i.e. COUNTS_INDEX['HH'] = 0
COUNTS_INDEX = {count: index for index, count in enumerate(COUNTS)} 

class W3:
    """
    W3 (Riccardi) witnesses. These use local measurements on 3 Pauli bases.

    Attributes: 
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
    stokes            - the Stokes parameters of the density matrix

    NOTE: One of rho or counts must be given
    NOTE: If counts is given, experimental calculations will be used
    NOTE: If rho is given, theoretical calculations will be used
    """
    def __init__(self, rho = None, counts = None):
        self.counts = counts
            
        # counts not given, so we want to use the given theoretical rho
        if counts is None:
            assert rho is not None, "ERROR: counts not given, so rho should be given"
            self.rho = rho
            self.stokes = self.stokes_from_mtx(rho)

        else:
            # counts given, so we will construct an experimental density matrix, so
            # rho should not be given as it's for theoretical states
            assert rho is None, "ERROR: counts was given, so rho should not be given"

            # store individual counts in variables
            self.hh, self.hv = counts[COUNTS_INDEX['HH']], counts[COUNTS_INDEX['HV']]
            self.hd, self.ha = counts[COUNTS_INDEX['HD']], counts[COUNTS_INDEX['HA']]
            self.hr, self.hl = counts[COUNTS_INDEX['HR']], counts[COUNTS_INDEX['HL']]
            self.vh, self.vv = counts[COUNTS_INDEX['VH']], counts[COUNTS_INDEX['VV']]
            self.vd, self.va = counts[COUNTS_INDEX['VD']], counts[COUNTS_INDEX['VA']]
            self.vr, self.vl = counts[COUNTS_INDEX['VR']], counts[COUNTS_INDEX['VL']]
            self.dh, self.dv = counts[COUNTS_INDEX['DH']], counts[COUNTS_INDEX['DV']]
            self.dd, self.da = counts[COUNTS_INDEX['DD']], counts[COUNTS_INDEX['DA']]
            self.dr, self.dl = counts[COUNTS_INDEX['DR']], counts[COUNTS_INDEX['DL']]
            self.ah, self.av = counts[COUNTS_INDEX['AH']], counts[COUNTS_INDEX['AV']]
            self.ad, self.aa = counts[COUNTS_INDEX['AD']], counts[COUNTS_INDEX['AA']]
            self.ar, self.al = counts[COUNTS_INDEX['AR']], counts[COUNTS_INDEX['AL']]
            self.rh, self.rv = counts[COUNTS_INDEX['RH']], counts[COUNTS_INDEX['RV']]
            self.rd, self.ra = counts[COUNTS_INDEX['RD']], counts[COUNTS_INDEX['RA']]
            self.rr, self.rl = counts[COUNTS_INDEX['RR']], counts[COUNTS_INDEX['RL']]
            self.lh, self.lv = counts[COUNTS_INDEX['LH']], counts[COUNTS_INDEX['LV']]
            self.ld, self.la = counts[COUNTS_INDEX['LD']], counts[COUNTS_INDEX['LA']]
            self.lr, self.ll = counts[COUNTS_INDEX['LR']], counts[COUNTS_INDEX['LL']]

            self.stokes = self.stokes_from_counts()

            # calculate experimental density matrix
            # NOTE: doesn't necessarily represent a full state

            # commented out line below because of uncertainty issue when dealing with complex rho
            # TODO: decide if we want to calculate expt_rho at all?
            # self.rho = self.expt_rho()
        
    
    
    def check_all_counts(self, not_zz=False, not_yy=False, not_xx=False, not_xy=False,
                         not_yx=False, not_zy=False, not_yz=False, not_xz=False, not_zx=False):
        """
        By default, check that all counts were provided, this is the same
        as taking a full tomography. 

        NOTE: By setting any of the parameters to true, you can check all counts except the 
              count corresponding to the command. For example, if not_zz = True, then the 
              function will check all counts except zz
        """
        if not not_zz:
            self.check_zz(quiet=True)
        if not not_yy:
            self.check_yy(quiet=True)
        if not not_xx:
            self.check_xx(quiet=True)
        if not not_xy:
            self.check_xy(quiet=True)
        if not not_yx:
            self.check_yx(quiet=True)
        if not not_zy:
            self.check_zy(quiet=True)
        if not not_yz:
            self.check_yz(quiet=True)
        if not not_xz:
            self.check_xz(quiet=True)
        if not not_zx:
            self.check_zx(quiet=True)

        # true if we check all counts
        full_tomo = not (not_zz or not_yy or not_xx or not_xy or not_yx 
                         or not_zy or not_yz or not_xz or not_zx)
        
        if full_tomo:
            print("Full tomography taken!")

    ##################################
    ## WITNESS DEFINITIONS/FUNCTIONS
    ##################################
    def W3_1(self, theta):
        """
        The first W3 witness

        Params:
        theta - the parameter for the rank-1 projector

        NOTE: this returns the operator, not an expectation value
        """
        a = np.cos(theta)
        b = np.sin(theta)

        # For experimental data, ensure we have the necessary counts
        if self.counts is not None:
            W3.check_counts(self)

        phi1 = a*PHI_P + b*PHI_M
        return op.partial_transpose(phi1 * op.adjoint(phi1))
        
    def W3_2(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts is not None:
            W3.check_counts(self)
        
        phi2 = a*PSI_P + b*PSI_M
        return op.partial_transpose(phi2 * op.adjoint(phi2))
        
    def W3_3(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts is not None:
            W3.check_counts(self)
        
        phi3 = a*PHI_P + b*PSI_P
        return op.partial_transpose(phi3 * op.adjoint(phi3))

    def W3_4(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts is not None:
            W3.check_counts(self)

        phi4 = a*PHI_M + b*PSI_M
        return op.partial_transpose(phi4 * op.adjoint(phi4))
        
    def W3_5(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts is not None:
            W3.check_counts(self)

        phi5 = a*PHI_P - 1j*b*PSI_M
        return op.partial_transpose(phi5 * op.adjoint(phi5))
        
    def W3_6(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts is not None:
            W3.check_counts(self)

        phi6 = a*PHI_M - 1j*b*PSI_P
        return op.partial_transpose(phi6 * op.adjoint(phi6))
    
    
    def W_stokes(self, idx, theta):
        """
        Returns the Stokes parameters for one W3

        Parameters:
            idx: the subscript of the W3 witness to get stokes for
            theta: the minimization parameter

        NOTE: indexes start from 1 so that they correspond with
        the witness subscripts
        """
        assert theta is not None, "ERROR: theta not given"
        ws = [self.W3_1, self.W3_2, self.W3_3, self.W3_4, self.W3_5, self.W3_6]

        stokes = self.stokes_from_mtx(ws[idx-1](theta))
        return stokes
    
    def expec_val(self, idx, theta):
        """
        Returns the expectation value of one W3

        Parameters:
            idx: the subscript of the W3 witness to get expectation value for
            theta: the minimization parameter

        NOTE: indexes start from 1 so that they correspond with
        the witness subscripts
        """
        W_stokes = self.W_stokes(idx, theta)
        return 0.25 * np.dot(self.stokes, W_stokes)

    def get_witnesses(self, return_type, theta=None):
        """
        If return_type is 'stokes':
            Returns the Stokes parameters for each of the 6 W3 witnesses
        If return_type is 'operators':
            Returns all 6 W3 witnesses as operators
        If return type is 'vals':
            Returns the expectation values of the W3 witnesses with a given theta

        Params:
        return_type - determines whether to return Stokes parameters or expectation values
                      of the witnesses
        theta (optional) - the theta value to calculate expectation values or stokes params

        NOTE: theta is not given by default, and must be given when returning vals or stokes
        NOTE: even if theta is given, vals are only calculated for return_type vals
        """
        
        if return_type == "operators":
            return [self.W3_1, self.W3_2, self.W3_3, self.W3_4, self.W3_5, self.W3_6]

        elif return_type == "stokes":
            return [self.W_stokes(idx, theta) for idx in range(1, 7)]
        
        elif return_type == "vals":
            values = [self.expec_val(idx, theta) for idx in range(1, 7)]
            return values
        
        else: # invalid return_type
            raise ValueError("Invalid return_type. Must be 'stokes', 'operators', or 'vals'.")

    ##################################
    ## COUNT HANDLING FUNCTIONS
    ##################################
    def expt_rho(self):
        # TODO: remove this function if not in use
        """
        Calculates the experimental density matrix
        
        NOTE: this only represents an actual state if all counts were given
              otherwise, the resulting matrix only contains partial information
              of the state
        """
        
        rho = np.zeros((4, 4), dtype='complex128')
        for i in range(len(PAULI)):
            rho += self.stokes[i] * PAULI[i]
        
        return 0.25 * rho

    def stokes_from_counts(self):
        """
        Calculates Stokes parameters of a density matrix and outputs them as a 16-dim vector
        
        NOTE: The order of this list is the same as S_{i, j} as listed in the 
              1-photon and 2-photon states from Beili Nora Hu paper (page 9)
        """
        assert self.counts is not None, "ERROR: counts not given"

        stokes = [1,                                                                         # 0
            (self.dd - self.da + self.ad - self.aa)/(self.dd + self.da + self.ad + self.aa),
            (self.rr + self.lr - self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.hh - self.hv + self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.dd + self.da - self.ad - self.aa)/(self.dd + self.da + self.ad + self.da),
            (self.dd - self.da - self.ad + self.aa)/(self.dd + self.da + self.ad + self.aa), # 5
            (self.dr - self.dl - self.ar + self.al)/(self.dr + self.dl + self.ar + self.al),
            (self.dh - self.dv - self.ah + self.av)/(self.dh + self.dv + self.ah + self.av),
            (self.rr - self.lr + self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.rd - self.ra - self.ld + self.la)/(self.rd + self.ra + self.ld + self.la),
            (self.rr - self.rl - self.lr + self.ll)/(self.rr + self.rl + self.lr + self.ll), # 10
            (self.rh - self.rv - self.lh + self.lv)/(self.rh + self.rv + self.lh + self.lv),
            (self.hh + self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.hd - self.ha - self.vd + self.va)/(self.hd + self.ha + self.vd + self.va),
            (self.hr - self.hl - self.vr + self.vl)/(self.hr + self.hl + self.vr + self.vl),
            (self.hh - self.hv - self.vh + self.vv)/(self.hh + self.hv + self.vh + self.vv)  # 15
        ]
        
        return stokes
    
    def stokes_from_mtx(self, M):
        """
        Calculates Stokes parameters of a matrix or operator and outputs them as a 16-dim vector

        Params:
        M - the 4x4 complex matrix to find Stokes params for
        """
        S_M = np.empty(16)
        S_M[0] = np.trace(M).real
        for i in range(1, 16):
            S_M[i] = np.trace(PAULI[i] @ M).real
        return S_M

    def check_zz(self, quiet=False):
        """
        Checks the necessary counts to determine if the zz measurement was taken

        Params:
        quiet - if true, don't tell the user if the measurement is given
        """
        assert self.hh != 0, "Missing HH measurement"
        assert self.hv != 0, "Missing HV measurement"
        assert self.vh != 0, "Missing VH measurement"
        assert self.vv != 0, "Missing VV measurement"

        if not quiet:
            print("ZZ measurement was taken!")

    def check_xx(self, quiet=False):
        """Determines if the xx measurement was taken"""
        assert self.dd != 0, "Missing DD measurement"
        assert self.da != 0, "Missing DA measurement"
        assert self.ad != 0, "Missing AD measurement"
        assert self.aa != 0, "Missing AA measurement"

        if not quiet:
            print("XX measurement was taken!")

    def check_yy(self, quiet=False):
        """Determines if the yy measurement was taken"""
        assert self.rr != 0, "Missing RR measurement"
        assert self.rl != 0, "Missing RL measurement"
        assert self.lr != 0, "Missing LR measurement"
        assert self.ll != 0, "Missing LL measurement"

        if not quiet:
            print("YY measurement was taken!")

    def check_xy(self, quiet=False):
        """Determines if the xy measurement was taken"""
        assert self.dr != 0, "Missing DR measurement"
        assert self.dl != 0, "Missing DL measurement"
        assert self.ar != 0, "Missing AR measurement"
        assert self.al != 0, "Missing AL measurement"

        if not quiet:
            print("XY measurement was taken!")

    def check_yx(self, quiet=False):
        """Determines if the yx measurement was taken"""
        assert self.rd != 0, "Missing RD measurement"
        assert self.ra != 0, "Missing RA measurement"
        assert self.ld != 0, "Missing LD measurement"
        assert self.la != 0, "Missing LA measurement"

        if not quiet:
            print("YX measurement was taken!")

    def check_zy(self, quiet=False):
        """Determines if the zy measurement was taken"""
        assert self.hr != 0, "Missing HR measurement"
        assert self.hl != 0, "Missing HL measurement"
        assert self.vr != 0, "Missing VR measurement"
        assert self.vl != 0, "Missing VL measurement"

        if not quiet:
            print("ZY measurement was taken!")

    def check_yz(self, quiet=False):
        """Determines if the yz measurement was taken"""
        assert self.rh != 0, "Missing RH measurement"
        assert self.rv != 0, "Missing RV measurement"
        assert self.lh != 0, "Missing LH measurement"
        assert self.lv != 0, "Missing LV measurement"   

        if not quiet:
            print("YZ measurement was taken!")

    def check_xz(self, quiet=False):
        """Determines if the xz measurement was taken"""
        assert self.dh != 0, "Missing DH measurement"
        assert self.dv != 0, "Missing DV measurement"
        assert self.ah != 0, "Missing AH measurement"
        assert self.av != 0, "Missing AV measurement"

        if not quiet:
            print("XZ measurement was taken!")

    def check_zx(self, quiet=False):
        """Determines if the zx measurement was taken"""
        assert self.hd != 0, "Missing HD measurement"
        assert self.ha != 0, "Missing HA measurement"
        assert self.vd != 0, "Missing VD measurement"
        assert self.va != 0, "Missing VA measurement"

        if not quiet:
            print("ZX measurement was taken!")

    def check_all_counts(self, not_zz=False, not_yy=False, not_xx=False, not_xy=False,
                         not_yx=False, not_zy=False, not_yz=False, not_xz=False, not_zx=False):
        """
        By default, check that all counts were provided, this is the same
        as taking a full tomography. 

        NOTE: By setting any of the parameters to true, you can check all counts except the 
              count corresponding to the command. For example, if not_zz = True, then the 
              function will check all counts except zz
        """
        if not not_zz:
            self.check_zz(quiet=True)
        if not not_yy:
            self.check_yy(quiet=True)
        if not not_xx:
            self.check_xx(quiet=True)
        if not not_xy:
            self.check_xy(quiet=True)
        if not not_yx:
            self.check_yx(quiet=True)
        if not not_zy:
            self.check_zy(quiet=True)
        if not not_yz:
            self.check_yz(quiet=True)
        if not not_xz:
            self.check_xz(quiet=True)
        if not not_zx:
            self.check_zx(quiet=True)

        # true if we check all counts
        full_tomo = not (not_zz or not_yy or not_xx or not_xy or not_yx 
                         or not_zy or not_yz or not_xz or not_zx)
        
        if full_tomo:
            print("Full tomography taken!")

    def check_counts(self):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data
        """
        self.check_zz(quiet=True)
        self.check_xx(quiet=True)
        self.check_yy(quiet=True)

    # TODO: Fix this
    def __str__(self):
        return (
            f'Rho: {self.rho}\n'
            f'Counts: {self.counts}\n'
        )

class W5(W3):
    """
    W5 witnesses, calculated with Paco's rotations (section 3.4.2 of Navarro thesis).
    These use local measurements on 5 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W3, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)

    ## Triplet 1: Rotate about z ##
    def W5_1(self, theta, alpha):
        """
        First W5 witness, rotates particle 1 about the z-axis
        from the W3_1 witness

        Params:
        theta - free parameter used in W3_1
        alpha - rotation angle
        """
        # W3 witness to rotate from
        w1 = self.W3_1(theta)
        
        if self.counts is not None:
            W5.check_counts(self, triplet=1)

        # Create appropriate rotation matrix and apply it to the W3 witness
        rotation = np.kron(R_z(alpha), IDENTITY)
        return op.rotate_m(w1, rotation)
    
    def W5_2(self, theta, alpha):
        w2 = self.W3_2(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), IDENTITY)
        return op.rotate_m(w2, rotation)
    
    def W5_3(self, theta, alpha, beta):
        w3 = self.W3_3(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), R_z(beta))
        return op.rotate_m(w3, rotation)
    

    ## Triplet 2: Rotate about x ##
    def W5_4(self, theta, alpha):
        w3 = self.W3_3(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), IDENTITY)
        return op.rotate_m(w3, rotation)
    
    def W5_5(self, theta, alpha):
        w4 = self.W3_4(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), IDENTITY)
        return op.rotate_m(w4, rotation)
    
    def W5_6(self, theta, alpha, beta):
        w1 = self.W3_1(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), R_x(beta))
        return op.rotate_m(w1, rotation)
    
    
    ## Triplet 3: Rotate about y ##
    def W5_7(self, theta, alpha):
        w5 = self.W3_5(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), IDENTITY)
        return op.rotate_m(w5, rotation)
    
    def W5_8(self, theta, alpha):
        w6 = self.W3_6(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), IDENTITY)
        return op.rotate_m(w6, rotation)

    def W5_9(self, theta, alpha, beta):
        w1 = self.W3_1(theta)

        if self.counts is not None:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), R_y(beta))
        return op.rotate_m(w1, rotation)
    
    def W_stokes(self, idx, *params):
        """
        Returns the Stokes parameters for one W5

        Params:
            idx: the subscript of the W5 witness to get stokes for
            params (theta, alpha, beta): the minimization parameters

        NOTE: indexes start from 1 so that they correspond with
        the witness subscripts
        """

        w5s = [self.W5_1, self.W5_2, self.W5_3, 
                self.W5_4, self.W5_5, self.W5_6,
                self.W5_7, self.W5_8, self.W5_9]
        
        # witnesses 3, 6, and 9 need beta (NOTE: idx is one-indexed)
        num_params = len(params)
        if idx % 3 == 0:
            assert num_params == 3, f"ERROR: {num_params} params given, expected 3"
        else:
            assert num_params == 2, f"ERROR: {num_params} params given, expected 2"
        
        stokes = self.stokes_from_mtx(w5s[idx-1](*params))
        return stokes
    
    def expec_val(self, idx, *params):
        """
        Returns the expectation value of one W5

        Parameters:
        idx: the subscript of the W5 witness to get expectation value for
        params (theta, alpha, beta): the minimization parameters

        NOTE: indexes start from 1 so that they correspond with
        the witness subscripts
        """
        W_stokes = self.W_stokes(idx, *params)
        return 0.25 * np.dot(self.stokes, W_stokes)
    
    def get_witnesses(self, return_type, theta=None, alpha=None, beta=None):
        """
        If return_type is 'stokes':
            Returns the Stokes parameters for each of the 9 W5 witnesses
        If return_type is 'operators':
            Returns all 9 W5 witnesses as operators
        If return type is 'vals':
            Returns the expectation values of the W5 witnesses with given parameters

        NOTE: theta, alpha, beta must be given when returning stokes or vals
        NOTE: Works the same as the get_witnesses function from W3, see docstring from that
              function for more details
        """

        if return_type == "operators":
            return [self.W5_1, self.W5_2, self.W5_3, 
                self.W5_4, self.W5_5, self.W5_6,
                self.W5_7, self.W5_8, self.W5_9]
        
        elif return_type == "stokes":
            return [self.W_stokes(idx, theta, alpha, beta) for idx in range(1, 10)]
        
        elif return_type == "vals":
            return [self.expec_val(idx, theta, alpha, beta) for idx in range(1, 10)]
        
        else: # invalid return_type
            raise ValueError("Invalid return_type. Must be 'stokes', 'operators', or 'vals'.")
    
    def check_counts(self, triplet):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        triplet - which triplet the witness belongs to

        NOTE: The xx, yy, and zz measurement have already been checked
              by a previous call to W3.check_counts, which happens when
              getting the W3 witness to perform the rotation on to get the W5
        """
        if triplet == 1:
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)

        elif triplet == 2:
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)

        elif triplet == 3:
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)

        else:
            assert False, "Invalid triplet specified"
    

class W8(W5):
    """
    W8 witnesses, calculated with Paco's rotations (section 4.1 of Navarro thesis).
    These use local measurements on 8 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W5, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)


    ######################################################
    ## SET 1 (W8_{1-6}): EXCLUDES XY MEASUREMENT
    ######################################################
    
    ## Triplet 1: Rotate particle 1 on y then x ##
    def W8_1(self, theta, alpha, beta):
        """
        First W8 witness

        Params:
        theta             - free parameter used in W3
        alpha             - first rotation angle (for W5)
        beta              - second rotation angle (to go from W5 to W8)
        """
        # W5 witness to rotate from
        w5_7 = self.W5_7(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=1)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_7, rotation)
    
    def W8_2(self, theta, alpha, beta):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=1)
        
        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_8, rotation)
    
    def W8_3(self, theta, alpha, beta, gamma):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=1)
        
        rotation = np.kron(R_x(gamma), IDENTITY)
        return op.rotate_m(w5_9, rotation)
    

    ## Triplet 2: Rotate about x, then rotate particle 2 about y ##
    def W8_4(self, theta, alpha, beta):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_4, rotation)
    
    def W8_5(self, theta, alpha, beta):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_5, rotation)
    
    def W8_6(self, theta, alpha, beta, gamma):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(gamma))
        return op.rotate_m(w5_6, rotation)
    

    #############################################
    ## SET 2: EXCLUDES YX
    #############################################

    ## Triplet 3: Rotate particle 1 on x then y ##
    def W8_7(self, theta, alpha, beta):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=3)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_4, rotation)
    
    def W8_8(self, theta, alpha, beta):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=3)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_5, rotation)
    
    def W8_9(self, theta, alpha, beta, gamma):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=3)

        rotation = np.kron(R_y(gamma), IDENTITY)
        return op.rotate_m(w5_6, rotation)
    

    ## Triplet 4: Rotate about y, then rotate particle 2 about x ##
    def W8_10(self, theta, alpha, beta):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_7, rotation)
    
    def W8_11(self, theta, alpha, beta):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_8, rotation)
    
    def W8_12(self, theta, alpha, beta, gamma):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(gamma))
        return op.rotate_m(w5_9, rotation)
    
    #############################################
    ## SET 3: EXCLUDES XZ
    #############################################

    ## Triplet 5: Rotate particle 1 by z then x ##
    def W8_13(self, theta, alpha, beta):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=5)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_1, rotation)
    
    def W8_14(self, theta, alpha, beta):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=5)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_2, rotation)
    
    def W8_15(self, theta, alpha, beta, gamma):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=5)

        rotation = np.kron(R_x(gamma), IDENTITY)
        return op.rotate_m(w5_3, rotation)
    

    ## Triplet 6: Rotate about x, then rotate particle 2 by z ##
    def W8_16(self, theta, alpha, beta):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_4, rotation)
    
    def W8_17(self, theta, alpha, beta):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_5, rotation)
    
    def W8_18(self, theta, alpha, beta, gamma):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(gamma))
        return op.rotate_m(w5_6, rotation)
    

    #############################################
    ## SET 4: EXCLUDES ZX
    #############################################

    ## Triplet 7: Rotate particle 1 by x then z ##
    def W8_19(self, theta, alpha, beta):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=7)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_4, rotation)
    
    def W8_20(self, theta, alpha, beta):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=7)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_5, rotation)
    
    def W8_21(self, theta, alpha, beta, gamma):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=7)

        rotation = np.kron(R_z(gamma), IDENTITY)
        return op.rotate_m(w5_6, rotation)
    

    ## Triplet 8: Rotate about z, then rotate particle 2 by x ##
    def W8_22(self, theta, alpha, beta):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_1, rotation)
    
    def W8_23(self, theta, alpha, beta):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_2, rotation)
    
    def W8_24(self, theta, alpha, beta, gamma):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(gamma))
        return op.rotate_m(w5_3, rotation)
    
    
    #############################################
    ## SET 5: EXCLUDES YZ
    #############################################

    ## Triplet 9: Rotate particle 1 by z then y ##
    def W8_25(self, theta, alpha, beta):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=9)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_1, rotation)
    
    def W8_26(self, theta, alpha, beta):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=9)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_2, rotation)
    
    def W8_27(self, theta, alpha, beta, gamma):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=9)

        rotation = np.kron(R_y(gamma), IDENTITY)
        return op.rotate_m(w5_3, rotation)
    

    ## Triplet 10: Rotate about y, then rotate particle 2 by z ##
    def W8_28(self, theta, alpha, beta):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_7, rotation)
    
    def W8_29(self, theta, alpha, beta):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_8, rotation)
    
    def W8_30(self, theta, alpha, beta, gamma):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(gamma))
        return op.rotate_m(w5_9, rotation)
    

    #############################################
    ## SET 6: EXCLUDES ZY
    #############################################

    ## Triplet 11: Rotate particle 1 by y then z ##
    def W8_31(self, theta, alpha, beta):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=11)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_7, rotation)
    
    def W8_32(self, theta, alpha, beta):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=11)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_8, rotation)
    
    def W8_33(self, theta, alpha, beta, gamma):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=11)

        rotation = np.kron(R_z(gamma), IDENTITY)
        return op.rotate_m(w5_9, rotation)
    

    ## Triplet 12: Rotate about z, then rotate particle 2 by y ##
    def W8_34(self, theta, alpha, beta):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_1, rotation)
    
    def W8_35(self, theta, alpha, beta):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts is not None:
            self.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_2, rotation)
    
    def W8_36(self, theta, alpha, beta, gamma):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts is not None:
            self.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(gamma))
        return op.rotate_m(w5_3, rotation)
    

    def W_stokes(self, idx, *params):
        """
        Returns the Stokes parameters for one W8

        Params:
            idx: the subscript of the W8 witness to get stokes for
            params (theta, alpha, beta, gamma): the minimization parameters

        NOTE: indexes start from 1 so that they correspond with
        the witness subscripts
        """

        w8s = [self.W8_1, self.W8_2, self.W8_3, self.W8_4, self.W8_5, self.W8_6,
               self.W8_7, self.W8_8, self.W8_9, self.W8_10, self.W8_11, self.W8_12,
               self.W8_13, self.W8_14, self.W8_15, self.W8_16, self.W8_17, self.W8_18,
               self.W8_19, self.W8_20, self.W8_21, self.W8_22, self.W8_23, self.W8_24,
               self.W8_25, self.W8_26, self.W8_27, self.W8_28, self.W8_29, self.W8_30,
               self.W8_31, self.W8_32, self.W8_33, self.W8_34, self.W8_35, self.W8_36]
        
        # every 3rd witness needs gamma (NOTE: idx is one-indexed)
        num_params = len(params)
        if idx % 3 == 0:
            assert num_params == 4, f"ERROR: {num_params} params given, expected 4"
        else:
            assert num_params == 3, f"ERROR: {num_params} params given, expected 3"
        stokes = self.stokes_from_mtx(w8s[idx-1](*params))
        return stokes
    
    def expec_val(self, idx, *params):
        """
        Returns the expectation value of one W8

        Parameters:
            idx: the subscript of the W8 witness to get expectation value for
            params (theta, alpha, beta, gamma): the minimization parameters

        NOTE: indexes start from 1 so that they correspond with
        the witness subscripts
        """
        W_stokes = self.W_stokes(idx, *params)
        return 0.25 * np.dot(self.stokes, W_stokes)

    def get_witnesses(self, return_type, theta=None, alpha=None, beta=None, gamma=None):
        """
        If return_type is 'stokes':
            Returns the Stokes parameters for each of the 36 W8 witnesses
        If return_type is 'operators':
            Returns all 36 W8 witnesses as operators
        If return type is 'vals':
            Returns the expectation values of the W8 witnesses with given parameters

        NOTE: theta, alpha, beta, gamma must be given when returning vals or stokes
        NOTE: Works the same as the get_witnesses function from W3, see docstring from that
              function for more details
        """

        if return_type == "operators":
            return [self.W8_1, self.W8_2, self.W8_3, self.W8_4, self.W8_5, self.W8_6,
               self.W8_7, self.W8_8, self.W8_9, self.W8_10, self.W8_11, self.W8_12,
               self.W8_13, self.W8_14, self.W8_15, self.W8_16, self.W8_17, self.W8_18,
               self.W8_19, self.W8_20, self.W8_21, self.W8_22, self.W8_23, self.W8_24,
               self.W8_25, self.W8_26, self.W8_27, self.W8_28, self.W8_29, self.W8_30,
               self.W8_31, self.W8_32, self.W8_33, self.W8_34, self.W8_35, self.W8_36]
        
        elif return_type == "stokes":
            return [self.W_stokes(idx, theta, alpha, beta, gamma) for idx in range(1, 37)]
        
        elif return_type == "vals":
            return [self.expec_val(idx, theta, alpha, beta, gamma) for idx in range(1, 37)]
        
        else: # invalid return_type
            raise ValueError("Invalid return_type. Must be 'stokes', 'operators', or 'vals'.")
    
    def check_counts(self, triplet):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        triplet - which triplet the witness belongs to
        """
        ## Triplets 1-2 exclude the xy measurement ##
        if triplet == 1 or triplet == 2:
            self.check_all_counts(not_xy=True)


        ## Triplets 3-4 exclude yx ##
        elif triplet == 3 or triplet == 4:
            self.check_all_counts(not_yx=True)
        

        ## Triplets 5-6 exclude xz ##
        elif triplet == 5 or triplet == 6:
            self.check_all_counts(not_xz=True)
        

        ## Triplets 7-8 exclude zx ##
        elif triplet == 7 or triplet == 8:
            self.check_all_counts(not_zx=True)
        

        ## Triplets 9-10 exclude yz ##
        elif triplet == 9 or triplet == 10:
            self.check_all_counts(not_yz=True)
        

        ## Triplets 11-12 exclude zy ##
        elif triplet == 11 or triplet == 12:
            self.check_all_counts(not_zy=True)

        else:
            assert False, "Invalid triplet specified"


if __name__ == '__main__':
    print("States, Matrices, and Witnesses Loaded.")