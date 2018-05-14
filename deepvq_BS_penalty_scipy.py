import pdb
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

def BinarySolver(decoder_params, x_batch, batch_size,latent_dim,currentH,currentB,rho, maxIter, sigma=100):

    """
    Solve optimization problem with binary constraints
        min_x func(x)
        s.t. x \in {-1,1}

    Inputs:
        - func: Function that needs to be minimized
        - x0:   Initialization
    
    This implementation uses a novel method invented by HKT Team 
    Copyright 2018 HKT

    Refereces: https://www.facebook.com/dangkhoasdc - Khoa will be also the presenter for our oral presentation
    If Khoa can not be reached, please contact Tuan Hoang:
    https://www.facebook.com/hoang.anhxtuan

    """

    x0 = np.sign(currentB) # Input current B is supposed to be binary. Sign here is just  to assure.
    n = len(x0)
    
    #xt, vt: Values of x and v at the previous iteration, which are used to update x and v at the current iteration, respectively
    
    xt = x0 #np.zeros(x0.shape)  #np.sign(x0)
    vt = xt #np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this
    
    # Define the funtion to compute output of decoder based on decoder parameters
    def func(x):
        current_x = np.reshape(x, (batch_size, latent_dim))
        for layer_param in decoder_params:            
            current_x = np.tanh(np.dot(current_x, layer_param[0]) + layer_param[1])
        return np.sum((current_x - x_batch)**2) + sigma*np.sum( (x-currentH)**2 ) 
    
    def fcost(x): 
        return func(x) + sigma*np.sum((x-currentH)**2) 
   

    print("Initial cost with sign: %f without sign = %f" %(fcost(xt), fcost(currentH)))
    print("reconstruction: %f" %(func(x0)))
    print("Encoder error: %f" %(sigma*np.sum((x0-currentH)**2)))
    # pdb.set_trace()

    def fx(x): # Fix v, solve for x                
        return func(x) + rho*(n-np.dot(vt,x))**1       

    def fv(x): # Fix x, solve for v
        return (n - np.dot(xt,x))**1
    
    # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1        
    xbounds= [(-1,1) for i in range(n)]
    vbounds = [(-1, 1) for i in range(n)]
    vConstraints = ({'type':'ineq',
            'fun': lambda x: np.array([n - norm(x)**2]),
            'jac': lambda x: np.array(-2*x)
           })
    # Now, let the iterations begin
    converged = False
    iter = 0
    
    while iter < maxIter and not converged:
        # Fix v, minimize x        
        print('----Updating x ')                       
        x_res = minimize(fx, xt, bounds = xbounds)
        x = x_res.x

        # Fix x, update v
        print('----Updating v')
        v_res = minimize(fv, vt,  constraints= vConstraints, bounds=vbounds)
        v = v_res.x

        print("Iter: %d , fx = %.3f, prev_fx = %.3f, x diff: %.3f, rho = %.3f reconstruction: %f constraints = %.3f" 
              %(iter, fx(x), fx(xt), norm( np.multiply(v, 1+x) ), rho, func(x), n - np.dot(x,v)))        
        print("Total Cost: %f" %(fcost(x)))
        print("Reconsutrction: %f, Encoder error: %f" %( fcost(x)-sigma*np.sum((x-currentH)**2), sigma*np.sum((x-currentH)**2)))
        
        # Check for convergence        
        if iter >=3 and ( ( n - np.dot(x,v) < 1e-2  or                           
                          abs(fx(x) - fx(xt)) < 1e-6 ) ):
            converged = True
            print('--------Using LINF  - Converged---------')  
            if (fcost(xt)<fcost(x0)):          
                return xt         
            else:
                return x0

        
        rho = rho*1.2
        xt = x
        vt = v
        iter = iter + 1

    if (fcost(xt)<fcost(x0)):          
        return xt         
    else:
        return x0




