import numpy as np
import uncertainties as unp
import states_and_witnesses as states
from scipy.optimize import minimize
from inspect import signature

## NOTE: Be sure to import states_and_witnesses before this file
##       when working in a REPL (e.g. ipython) or notebook

#########################################
## State & Density Matrix Operations
#########################################

def ket(state):
    """
    Return the given state (represented as an array) as a ket

    Example: ket([1, 0]) -> [[1]
                             [0]]
    """
    return np.array(state, dtype=complex).reshape(-1,1)

def adjoint(state):
    """
    Returns the adjoint of a state vector
    """
    return np.conjugate(state).T

def get_rho(state):
    """
    Computes the density matrix given a 2-qubit state vector

    Param: state - the 2-qubit state vector
    """
    return state @ adjoint(state)

def partial_transpose(rho, subsys='B'):
    """ 
    Helper function to compute the partial transpose of a density matrix. 
    Useful for the Peres-Horodecki criterion, which states that if the partial transpose 
    of a density matrix has at least one negative eigenvalue, then the state is entangled.
    
    Parameters:
        rho: density matrix
        subsys: which subsystem to compute partial transpose wrt, i.e. 'A' or 'B'
    """
    # decompose rho into blocks
    b1 = rho[:2, :2]
    b2 = rho[:2, 2:]
    b3 = rho[2:, :2]
    b4 = rho[2:, 2:]

    PT = np.array(np.block([[b1.T, b2.T], [b3.T, b4.T]]))

    if subsys=='B':
        return PT
    elif subsys=='A':
        return PT.T

# Rotation operations
def rotate_z(m, theta):
    """Rotate matrix m by theta about the z axis"""
    return states.R_z(theta) @ m @ adjoint(states.R_z(theta))

def rotate_x(m, theta):
    """Rotate matrix m by theta about the x axis"""
    return states.R_x(theta) @ m @ adjoint(states.R_x(theta))

def rotate_y(m, theta):
    """Rotate matrix m by theta about the y axis"""
    return states.R_y(theta) @ m @ adjoint(states.R_y(theta))

def rotate_m(m, n):
    """
    Rotate matrix m with matrix n
    
    Params:
    m - the matrix to be rotated
    n - the matrix that is doing the rotation (i.e. R_z)

    NOTE: m and n must be of the same size
    """
    return n @ m @ adjoint(n)


##################
## MINIMIZATION
##################

# TODO: REVIEW THIS BY LOOKING AT SUMMER 2024 PAPER DRAFT 
#       FIGURES 4,5,6 (SOLID LINES) AND EQUATIONS 3,4,5
def minimize_witnesses(witness_classes, rho=None, counts=None, num_guesses=10):
    """
    Calculates the minimum expectation values for each the witnesses specified
    in a given witness class for a given theoretical density matrix or for given
    experimental data

    Parameters:
        witness_classes: a class or list of classes of witnesses
        rho:             the density matrix
        counts:          experimental data 
        num_guesses:     the number of random initial guesses to use in minimization
        NOTE: allowable witness classes are W3, W5, W7, W8, and NavarroWitness (which is all witnesses)
        NOTE: There are an additional 2 guesses at the bounds on top of the random guesses
        TODO: The W7 witnesses have not been implemented yet

    Returns: (min_thetas, min_vals)
        min_thetas: a list of the thetas corresponding to the minimum expectation values
        min_vals:   a list of the minimum expectation values
        NOTE: These are listed in the order of the witnesses (e.g. W3_1 first and W3_6 last).
              If multiple classes given, they are listed in the order the classes are listed
    """    
    # Lists to keep track of minimum expectation values and their corresponding parameters
    min_params = []
    min_vals = []

    # Get necessary witnesses
    if type(witness_classes) != list:
        num_classes = 1
        all_expec_vals = [[witness_classes(rho=rho, counts=counts).expec_val]]
    else:
        all_expec_vals = []
        num_classes = 0
        for c in witness_classes:
            all_expec_vals.append(c(rho=rho, counts=counts).expec_val)
            num_classes += 1

    def optimize(W_val, witness_idx, params, bounds):
        """
        Generic minimization loop that works for any number of minimization parameters

        Parameters:
            W_val:                a function to get expectation value for one witness of a certain class
            witness_idx:          iterator that keeps track of which witness we are minimizing
            params:               the witness parameters to be minimized (i.e. theta, alpha, beta)
        """
        def print_callback(params):
            """Callback function called after every iteration of scipy minimization.
            Prints the current iterator value, expectation value, and parameters.
            """
            expec_val = loss(params)
            #print(expec_val)
            print_callback.iter += 1

        def loss(params):
            """
            Uses the expectation value of each witness as the loss function for minimization
            """
            loss = W_val(witness_idx, *params)

            # Handle experimental uncertainties
            loss_np = np.empty(16)
            if counts is not None:
                loss_np = unp.nominal_value(loss)
            else:
                loss_np = loss
            return loss_np
        
        # Initialize the iteration number for the callback function
        print_callback.iter = 0
    
        # Scipy minimization using BFGS gradient descent
        min_W = minimize(loss, params, method="L-BFGS-B", bounds=bounds, callback=print_callback)

        # return the best-fit params and the minimized expectation value
        return min_W.x, min_W.fun


    # Minimize each witness
    # TODO: look into MULTITHREADING or multipooling
    for class_idx, W_class in zip([3, 5, 7, 8], all_expec_vals):
        if class_idx == 3:
            num_witnesses = 6
            num_params = 1

        elif class_idx == 5:
            num_witnesses = 9
            num_params = 3

        elif class_idx == 7:
            num_witnesses = 108
            num_params = 4

        elif class_idx == 8:
            num_witnesses = 36
            num_params = 4

        else:
            raise IndexError("ERROR: Witness class not found or out-of-bounds")

        # initialize bounds for the parameters to be minimized
        lower_bound = 0.0
        alpha_bound = lower_bound # initialize these upper bounds to
        beta_bound = lower_bound  #    prevent error in minimization loop
        gamma_bound = lower_bound

        if num_params == 1:
            theta_bound = np.pi
        elif num_params == 2:
            theta_bound = np.pi
            alpha_bound = np.pi 
        elif num_params == 3:
            theta_bound = np.pi/2
            alpha_bound = 2*np.pi 
            beta_bound = 2*np.pi
        else:
            theta_bound = np.pi/2
            alpha_bound = 2*np.pi
            beta_bound = 2*np.pi
            gamma_bound = 2*np.pi

        bounds = [(lower_bound, theta_bound), (lower_bound, alpha_bound), (lower_bound, beta_bound), (lower_bound, gamma_bound)][:num_params]

        # Set the initial "best value" to infinity
        min_val = float("inf")
        
        for witness_idx in range(1, num_witnesses+1): # witnesses are indexed from 1
            #print("\nMinimizing witness W" + str(class_idx) + "_" + str(witness_idx))      
            # Try different random initial guesses and use the best result
            for _ in range(num_guesses):
                theta = np.random.uniform(low=lower_bound, high=theta_bound)
                alpha = np.random.uniform(low=lower_bound, high=alpha_bound)
                beta = np.random.uniform(low=lower_bound, high=beta_bound)
                gamma = np.random.uniform(low=lower_bound, high=gamma_bound)

                # use the right number of parameters
                param_vars = [theta, alpha, beta, gamma][:num_params]
                this_min_params, this_min_val = optimize(W_class, witness_idx, param_vars, bounds)

                if this_min_val < min_val:
                    best_param = this_min_params
                    min_val = this_min_val

            # do a guess at lower_bound
            theta = lower_bound
            alpha = lower_bound
            beta = lower_bound
            gamma = lower_bound
            param_vars = [theta, alpha, beta, gamma][:num_params]
            this_min_params, this_min_val = optimize(W_class, witness_idx, param_vars, bounds)
            if this_min_val < min_val:
                    best_param = this_min_params
                    min_val = this_min_val

            # do a guess at upper_bound
            theta = theta_bound
            alpha = alpha_bound
            beta = beta_bound
            gamma = gamma_bound
            param_vars = [theta, alpha, beta, gamma][:num_params]
            this_min_params, this_min_val = optimize(W_class, witness_idx, param_vars, bounds)
            if this_min_val < min_val:
                    best_param = this_min_params
                    min_val = this_min_val

            min_params.append(best_param)
            min_vals.append(min_val)

    return (min_params, min_vals)


if __name__ == "__main__":
    print("Operations Loaded.")