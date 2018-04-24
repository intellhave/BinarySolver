import pdb
import numpy as np
#import autograd.numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
########################################################################
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
    vt = x0 #np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this
    print("Initial cost: %f norm x = %f" %(func(xt), norm(xt)))
    
    def fx(x): # Fix v, solve for x
        return (func(x) + rho*(np.dot(x,vt)-n)**2) #+ 0.1*np.sum(np.power(x-x0,2))

    def fv(x): # Fix x, solve for v
        return (n-np.dot(xt, x))**2

    # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1
    #xBounds = [[-1,1] for i in range(n)]
    
    xConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([1 - x[i]**2])            
           } for i in range(n))
    
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
        print('----Update x steps')        
        #x_res = minimize(fx, xt, bounds = xBounds, method='SLSQP',jac = gradx)
        option = {'maxiter': 1000, 'disp':False}
        x_res = minimize(fx, xt, constraints = xConstraints, method ='COBYLA', options=option)
        x = x_res.x
        #print(x_res.success)
        # Fix x, update v
        print('----Update v steps')
        v_res = minimize(fv, vt, constraints = vConstraints , method='COBYLA', options = option)
        v = v_res.x      
        #print(v_res.success)
        # pdb.set_trace()        
        # Check for convergence
        if iter > 10 and ((norm(x - xt) < 1e-6 and abs(func(x) - func(xt) < 1e-6)) or (n-np.dot(xt, vt))**2<0.01):
            converged = True
            print('--------Converged---------')
            #x[x<0.5] = -1
            return x

        print("Iter: %d , cost: %.3f, rho = %.3f constraints: %f" %(iter, func(xt), rho, (n-np.dot(xt, vt))**2))
        #print (xt)
        rho = rho*1.01
        xt = x
        vt = v
        iter = iter + 1

    return xt
    
# 



def NextPermute(x):
    """
        Return the next permutation from the current x in {-1,1}^L
        If x is last element, return an array of all zeros
    """
    n = len(x) 
    i = n - 1
    while x[i] < 0 and i>=0:
        i = i - 1
    
    if i >= 0:
        x[i]= -1
        for j in range(i+1,n):
            x[j] = 1
        return  x
    else:
        return np.zeros(n) 
    
    

def BruteForceBinarySolver(func, x0):
    """
        Solve min_x func(x) s.t. x \in {-1,1} by exhaustive search

    """
    n = len(x0)
    xt = np.ones(n, dtype='int')

    minCost = np.inf
    sol = xt

    while not (xt[0] == 0):
        cost = func(xt)
        #print(xt)
        #print(cost)
        if (cost  < minCost):
            sol = np.copy(xt)
            minCost = cost 
            #print("MinCost  = %f" % minCost)
            #print (sol)
        xt = NextPermute(xt)

    return sol.T





###################################################################################
# def BinarySolver_v1(func, x0, rho, maxIter):
#     """
#     Use exact penalty method to solve optimization problem with binary constraints
#         min_x func(x)
#         s.t. x \in {-1,1}

#     Inputs:
#         - func: Function that needs to be minimized
#         - x0:   Initialization
#     Refereces: https://www.facebook.com/dangkhoasdc

#     """
    
#     n = len(x0)  
#     #xt, vt: Values of x and v at the previous iteration, which are used to update x and v at the current iteration, respectively
#     xt = x0
#     vt = np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this


#     def fx(x): # Fix v, solve for x
#         return func(x) - rho*(np.dot(x,vt))

#     def fv(x): # Fix x, solve for v
#         return -np.dot(xt, x)

#     # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1
#     xBounds = [[-1,1] for i in range(n)]
    
#     # Ball-constraint ||v||^2 <= n
#     vConstraints = ({'type':'ineq',
#             'fun': lambda x: np.array([n - norm(x)**2]),
#             'jac': lambda x: np.array(-2*x)
#            })

#     # Now, let the iterations begin
#     converged = False
#     iter = 0
#     while iter < maxIter and not converged:               
#         # Fix v, minimize x
#         #print('----Update x steps')
#         x_res = minimize(fx, xt, bounds = xBounds, tol=1e-3)
#         x = x_res.x
#         print(x_res)
#         # Fix x, update v
#         #print('----Update v steps')
#         v_res = minimize(fv, vt, constraints = vConstraints, method = 'COBYLA')
#         v = v_res.x
#         print(v_res)
#         # Check for convergence
#         if iter > 3 and (norm(x - xt) < 1e-6 or (func(x) - func(xt) < 1e-6)):
#             converged = True
#             print('--------Converged---------')            
#             return x

#         print("Iter: %d , cost: %f" %(iter, func(xt)))
#         #print (xt)
#         rho = rho*1.1
#         xt = x
#         vt = v
#         iter = iter + 1
    
#     return xt
