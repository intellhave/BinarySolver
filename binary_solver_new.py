import pdb
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm


def BinarySolver(func, x0, rho, maxIter, sigma=100):

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
    h0 = x0.copy()
    xt = np.sign(x0)
    vt = np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this
    print("Initial cost: %f norm x = %f" %(func(xt), norm(xt)))

    def fx(x): # Fix v, solve for x
        return (func(x) + rho*(n-np.dot(x,vt))**2) + sigma*np.sum(np.power(x-h0,2))

    def fv(x): # Fix x, solve for v
        return (n-np.dot(xt, x))**2 
    
    # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1
    #xBounds = [[-1,1] for i in range(n)]

    xConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([1 - x[i]**2]),
            'jac': lambda x: np.array(-2*x[i])
           } for i in range(n))

        # Ball-constraint ||v||^2 <= n
    vConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([1 - x[i]**2]),
            'jac': lambda x: np.array(-2*x[i])
           } for i in range(n))
    
    # vConstraints_2 = ({'type':'ineq',
    #          'fun': lambda x: np.array([n - norm(x)**2]),
    #          'jac': lambda x: np.array(-2*x)
    #         })

    

    # Now, let the iterations begin
    converged = False
    iter = 0
    while iter < maxIter and not converged:
        # Fix v, minimize x
        print('----Update x steps')               
        x_res = minimize(fx, xt, constraints=xConstraints, method='COBYLA')
        x = x_res.x
        
        # Fix x, update v
        print('----Update v steps')
        v_res = minimize(fv, vt, constraints = vConstraints, method='COBYLA')
        v = v_res.x
        
        # Check for convergence
        if iter > 4 and ((norm(v - vt) < 1e-6 and abs(func(x) - func(xt) < 1e-6)) or (n-np.dot(xt, vt))**2<1.5):
            converged = True
            print('--------Using LINF  - Converged---------')            
            return vt

        print("Iter: %d , cost: %.3f, rho = %.3f constraints: %f" %(iter, func(xt), rho, (n-np.dot(xt, vt))**2))
        #print (xt)
        rho = rho*1.1
        xt = x
        vt = v
        iter = iter + 1

    return vt
#



