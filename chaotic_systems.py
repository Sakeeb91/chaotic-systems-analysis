import numpy as np
from scipy.integrate import solve_ivp

def rossler_system(t, state, a=0.2, b=0.2, c=5.7):
    """
    The Rössler system of ODEs.
    
    Parameters:
        t (float): Time (not used, but required by the ODE solver)
        state (array): Current state [x, y, z]
        a, b, c (float): System parameters
        
    Returns:
        list: Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = -y - z
    dy_dt = x + a * y
    dz_dt = b + z * (x - c)
    return [dx_dt, dy_dt, dz_dt]

def simulate_system(system_func, initial_state, t_span, dt, params=None, discard_transient=0):
    """
    Simulate a dynamical system.
    
    Parameters:
        system_func (callable): The system of ODEs
        initial_state (list): Initial conditions [x0, y0, z0, ...]
        t_span (list): Time span [t_start, t_end]
        dt (float): Time step for output
        params (dict): Parameters for the system function
        discard_transient (float): Time to discard as transient
        
    Returns:
        dict: Simulation results with time and state variables
    """
    params = params or {}
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    solution = solve_ivp(
        lambda t, y: system_func(t, y, **params),
        t_span,
        initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Convert to more convenient format
    results = {
        'time': solution.t,
        'states': solution.y.T  # Each row is a state vector at time t
    }
    
    # Discard transients if specified
    if discard_transient > 0:
        mask = results['time'] >= discard_transient
        results['time'] = results['time'][mask]
        results['states'] = results['states'][mask]
        
    return results 