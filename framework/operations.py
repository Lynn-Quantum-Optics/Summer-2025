import numpy as np
import sympy as sp
import states_and_witnesses as states
import tensorflow as tf
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
    
    Params:
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

#theta1 = sp.symbols('theta1')
#alpha1 = sp.symbols('alpha1')
#beta1 = sp.symbols('beta1')
#gamma1 = sp.symbols('gamma1')

# TODO: REVIEW THIS BY LOOKING AT SUMMER 2024 PAPER DRAFT 
#       FIGURES 4,5,6 (SOLID LINES) AND EQUATIONS 3,4,5
def minimize_witnesses(witness_classes, rho=None, counts=None, num_guesses=10):
    """
    Calculates the minimum expectation values for each the witnesses specified
    in a given witness class for a given theoretical density matrix or for given
    experimental data

    Params:
        witness_classes - a class or list of classes of witnesses
        rho             - the density matrix
        counts          - experimental data 
        num_guesses     - the number of random initial guesses to use in minimization
        NOTE: allowable witness classes are W3, W5, W7, W8, and NavarroWitness (which is all witnesses)
        NOTE: There are an additional 2 guesses at the bounds on top of the random guesses
        TODO: The W7 witnesses have not been implemented yet

    Returns: (min_thetas, min_vals)
        min_thetas - a list of the thetas corresponding to the minimum expectation values
        min_vals   - a list of the minimum expectation values
        NOTE: These are listed in the order of the witnesses (e.g. W3_1 first and W3_6 last).
              If multiple classes given, they are listed in the order the classes are listed
    """    
    # Lists to keep track of minimum expectation values and their corresponding parameters
    min_params = []
    min_vals = []

    # Get necessary witnesses
    if type(witness_classes) != list:
        all_W_stokes = witness_classes(rho=rho, counts=counts).get_witnesses("stokes")
        rho_stokes = witness_classes(rho=rho, counts=counts).stokes
    else:
        all_W_stokes = []
        for c in witness_classes:
            all_W_stokes.append(c(rho=rho, counts=counts).get_witnesses("stokes"))
        rho_stokes = states.W3(rho=rho, counts=counts).stokes

    def svec_to_tf(s):
        """
        Convert a vector of Stokes params to TensorFlow object
        """
        return tf.constant(s, dtype=tf.float64)

    def loss(W_stokes, rho_stokes, params):
        """
        Loss function for minimization: 1/16 * (S_w dot S_rho)

        NOTE: this is the expectation value of W defined by the Stokes params
        """
        W_stokes_tf = svec_to_tf(W_stokes(*params))
        print("W_stokes_tf: ", W_stokes_tf)
        rho_stokes_tf = svec_to_tf(rho_stokes)
        print("rho_stokes_tf: ", rho_stokes_tf)
        loss = 0.0625 * tf.tensordot(W_stokes_tf, rho_stokes_tf, axes=1)
        print("loss: ", loss)
        return loss
    
    # minimize using the Adam optimizer
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.05,
        first_decay_steps=250,
        t_mul=2.0,               # each restart lasts twice as long
        m_mul=1.0,               # learning rate stays the same after each restart
        alpha=0.001              # minimum LR
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    def optimize(W_stokes, rho_stokes, params, threshold=1e-10, max_iters=1000):
        """
        Generic minimization loop that works for any number of minimization parameters

        Parameters:
        W_stokes             - a 16d vector of Stokes parameters representing the witness
                               whose expectation value will be minimized
        rho_stokes           - a 16d vector of Stokes parameters representing the density matrix
                               of the state
        params               - the witness parameters to be minimized (i.e. theta, alpha, beta)
        threshold (optional) - smallest allowed change in the loss function (i.e. expectation value)
        max_iters (optional) - maximum number of iterations allowed in the optimization loop

        NOTE: threshold is 1e-10 by default
        NOTE: max_iters is 1000 by default
        """
        prev_loss = float("inf")

        # Optimization loop, stops when minimized value starts converging or when
        # the maximum iterations 
        for _ in range(max_iters):
            with tf.GradientTape() as tape:
                loss_value = loss(W_stokes, rho_stokes, params)

            loss_real = tf.math.real(loss_value).numpy()
            print("loss_real: ", loss_real)
            print("prev_loss: ", prev_loss)
            print("prev_loss - loss_real = ", prev_loss - loss_real)
        
            # Check if minimized value has converged within the threshold
            if abs(prev_loss - loss_real) < threshold:
                break
            prev_loss = loss_real

            # TODO: add fudging to get out of local minima
            grads = tape.gradient(loss_value, params)
            for g, p in zip(grads, params):
                if g is not None: # Check if gradient exists (avoid NoneType errors)
                    optimizer.apply_gradients([(g, p)])
                    p.assign(tf.clip_by_value(p, 0.0, np.pi))  # enforce bounds

        return [p.numpy() for p in param_vars], loss_real


    # Minimize each witness
    # TODO: look into MULTITHREADING or multipooling
    for W_stokes in all_W_stokes:
        # determine number of parameters to be minimized
        num_params = len(signature(W_stokes).parameters)
        
        # initialize bounds for the parameters to be minimized
        lower_bound = tf.constant(0.0, dtype=tf.float64)
        alpha_bound = lower_bound # initialize these upper bounds to
        beta_bound = lower_bound  #    prevent error in minimization loop
        gamma_bound = lower_bound

        if num_params == 1:
            theta_bound = tf.constant(np.pi, dtype=tf.float64)
        elif num_params == 2:
            theta_bound = tf.constant(np.pi, dtype=tf.float64)
            alpha_bound = tf.constant(np.pi, dtype=tf.float64) 
        elif num_params == 3:
            theta_bound = tf.constant(np.pi/2, dtype=tf.float64)
            alpha_bound = tf.constant(2*np.pi, dtype=tf.float64) 
            beta_bound = tf.constant(2*np.pi, dtype=tf.float64)
        else:
            theta_bound = tf.constant(np.pi/2, dtype=tf.float64)
            alpha_bound = tf.constant(2*np.pi, dtype=tf.float64) 
            beta_bound = tf.constant(2*np.pi, dtype=tf.float64)
            gamma_bound = tf.constant(2*np.pi, dtype=tf.float64)
        
        
        # Try different random initial guesses and use the best result
        min_val = float("inf")
        for _ in range(num_guesses):
            theta = tf.Variable(tf.random.uniform(shape=[], minval=lower_bound, 
                                                maxval=theta_bound, dtype=tf.float64))
            alpha = tf.Variable(tf.random.uniform(shape=[], minval=lower_bound, 
                                                maxval=alpha_bound, dtype=tf.float64))
            beta = tf.Variable(tf.random.uniform(shape=[], minval=lower_bound, 
                                                maxval=beta_bound, dtype=tf.float64))
            gamma = tf.Variable(tf.random.uniform(shape=[], minval=lower_bound, 
                                                maxval=gamma_bound, dtype=tf.float64))

            # use the right number of parameters
            param_vars = [theta, alpha, beta, gamma][:num_params]
            this_min_params, this_min_val = optimize(W_stokes, rho_stokes, param_vars)

            if this_min_val < min_val:
                best_param = this_min_params
                min_val = this_min_val

        # do a guess at lower_bound
        theta = tf.Variable(lower_bound, dtype=tf.float64)
        alpha = tf.Variable(lower_bound, dtype=tf.float64)
        beta = tf.Variable(lower_bound, dtype=tf.float64)
        gamma = tf.Variable(lower_bound, dtype=tf.float64)
        param_vars = [theta, alpha, beta, gamma][:num_params]
        this_min_params, this_min_val = optimize(W_stokes, rho_stokes, param_vars)
        if this_min_val < min_val:
                best_param = this_min_params
                min_val = this_min_val

        # do a guess at upper_bound
        theta = tf.Variable(theta_bound, dtype=tf.float64)
        alpha = tf.Variable(alpha_bound, dtype=tf.float64)
        beta = tf.Variable(beta_bound, dtype=tf.float64)
        gamma = tf.Variable(gamma_bound, dtype=tf.float64)
        param_vars = [theta, alpha, beta, gamma][:num_params]
        this_min_params, this_min_val = optimize(W_stokes, rho_stokes, param_vars)
        if this_min_val < min_val:
                best_param = this_min_params
                min_val = this_min_val

        min_params.append(best_param)
        min_vals.append(min_val)


    return (min_params, min_vals)


if __name__ == "__main__":
    print("Operations Loaded.")