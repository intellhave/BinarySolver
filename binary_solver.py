
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm


def BinarySolver(func, x0, rho, maxIter):
    """
    Use exact penalty method to solve optimization problem with binary constraints
        min_x func(x)
        s.t. x \in {-1,1}

    Inputs:
        - func: Function that needs to be minimized
        - x0:   Initialization
    Refereces: https://www.facebook.com/dangkhoasdc

    """
    
    n = len(x0)  
    #xt, vt: Values of x and v at the previous iteration, which are used to update x and v at the current iteration, respectively
    xt = x0
    vt = np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this


    def fx(x): # Fix v, solve for x
        return func(x) + rho*(np.dot(x,vt))

    def fv(x): # Fix x, solve for v
        return np.dot(xt, x)

    # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1

    xBounds = [[-1,1] for i in range(n)]
  


    # Ball-constraint ||v||^2 <= n
    vConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([n - norm(x)**2]),
            'jac': lambda x: np.array(-2*x)
           })

    # Now, let the iterations begin
    converged = False
    iter = 0
    while iter < maxIter and not converged:
        # Fix v, minimize x
        x_res = minimize(fx, xt, bounds = xBounds)
        x = x_res.x

        # Fix x, update v
        v_res = minimize(fv, vt, constraints = vConstraints)
        v = v_res.x

        # Check for convergence
        if norm(x-xt) < 0.00000000001:
            converged = True
            print('--------Converged---------')

        xt = x
        vt = v
        rho = rho*1.05


    return xt