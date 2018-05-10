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
    xt = np.sign(x0) #np.zeros(x0.shape)  #np.sign(x0)
    vt = xt #np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this
    print("Initial cost: %f norm x = %f" %(func(xt), norm(xt)))
    
    # Lagrangian duals
    y1 = np.zeros(x0.shape)
    y2 = np.zeros(x0.shape)

    def fx(x): # Fix v, solve for x
        return func(x) #+ sigma*np.sum(np.power(x-h0,2)) 
        + np.dot(y1, vt - np.ones(xt.shape) + x) + 0.5*rho*(np.sum(np.power(vt + x - np.ones(xt.shape), 2))) 
        + np.dot (y2, np.multiply(vt, x+np.ones(x0.shape))) + 0.5*rho*np.sum(np.power(np.multiply(vt, x+np.ones(x0.shape)),2))

    def fv(x): # Fix x, solve for v
        return np.dot(y1, x - np.ones(x0.shape) + xt) + 0.5*rho*np.sum(np.power(x - np.ones(x0.shape), 2))
        + np.dot(y2, np.multiply(x, np.ones(x0.shape)+xt)) + 0.5*rho*np.sum(np.power(np.multiply(x, np.ones(x0.shape)+xt),2))

    
    # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1    
    xConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([1 - x[i]**2]),
            'jac': lambda x: np.array(-2*x[i])
           } for i in range(n))

    vConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([1 - x[i]**2]),
            'jac': lambda x: np.array(-2*x[i])
           } for i in range(n))
    xbounds= [(-1,1) for i in range(n)]
    vbounds = [(0, 2) for i in range(n)]
    # Now, let the iterations begin
    converged = False
    iter = 0
    while iter < maxIter and not converged:
        # Fix v, minimize x
        print('----Updating x ')               
        x_res = minimize(fx, xt, bounds = xbounds)
        x = x_res.x
        print min(x), max(x)

        # Fix x, update v
        print('----Updating v')
        v_res = minimize(fv, vt, bounds = vbounds)
        v = v_res.x

        print min(v), max(v)
        y1 = y1 - rho*(v + x - np.ones(x0.shape))
        y2 = y2 - rho*(np.multiply(v, np.ones(x0.shape) + x))

        print("Iter: %d , fx = %.3f v diff: %.3f, rho = %.3f constraints: %f" %(iter, func(x)+sigma*np.sum(np.power(x-h0,2)), norm(v - vt), rho, (n-np.dot(xt, vt))**2))
        # Check for convergence
        # if iter > 4 and ((norm(v - vt) < 1e-6 and abs(func(x) - func(xt) < 1e-6)) or (n-np.dot(xt, vt))**2<1.5):
        if iter > 4 and ((norm(v - vt) < 1e-6  or (n-np.dot(xt, vt))**2<1e-6 or abs(fx(x)-fx(xt)<1e-6)  ) ):
            converged = True
            print('--------Using LINF  - Converged---------')            
            return xt
        
        #print (xt)
        rho = rho*1.1
        xt = x
        vt = v

        
        iter = iter + 1

    return xt
#



