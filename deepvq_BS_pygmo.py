import pdb
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
from autograd import grad
import pygmo as pg

class x_problem:
    def __init__(self, 
    x_batch, 
    batch_size, 
    latent_dim,
    decoder_params,
    currentH,
    vt,
    y1, 
    y2,
    sigma, 
    rho):
        self.x_batch = x_batch
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.decoder_params = decoder_params
        self.currentH = currentH
        self.vt = vt
        self.y1 = y1
        self.y2 = y2
        self.sigma = sigma
        self.rho = rho        
        self.dimension = vt.shape[0]
    
    def fitness(self, x):
        current_x = np.reshape(x, (self.batch_size, self.latent_dim))
        for layer_param in self.decoder_params:            
            current_x = np.tanh(np.dot(current_x, layer_param[0]) + layer_param[1])
        
        total = np.sum((current_x - self.x_batch)**2) + self.sigma*np.sum(np.power(x-self.currentH,2)) 
        + np.dot(self.y1, self.vt - np.ones(self.vt.shape) + x) + 0.5*self.rho*(np.sum(np.power(self.vt + x - np.ones(self.vt.shape), 2))) 
        + np.dot (self.y2, np.multiply(self.vt, x+np.ones(self.vt.shape))) + 0.5*self.rho*np.sum(np.power(np.multiply(self.vt, x+np.ones(self.vt.shape)),2))
        return [total]

    def get_bounds(self):
        return ([-1 for i in range(self.dimension)], [1 for i in range(self.dimension)])
    



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

    Refereces: https://www.facebook.com/dangkhoasdc

    """
    x0 = currentB
    n = len(x0)
    #xt, vt: Values of x and v at the previous iteration, which are used to update x and v at the current iteration, respectively
    #h0 = x0.copy()    
    xt = x0 #np.zeros(x0.shape)  #np.sign(x0)
    vt = np.ones(xt.shape) - xt #np.zeros(xt.shape)  # Initialize v to zeros!!!!!!! Note on this
    
    # Define the funtion to compute output of decoder based on decoder parameters
    def func(x):
        current_x = np.reshape(x, (batch_size, latent_dim))
        for layer_param in decoder_params:            
            current_x = np.tanh(np.dot(current_x, layer_param[0]) + layer_param[1])
        return np.sum((current_x - x_batch)**2)
    
   
    # Lagrangian duals
    y1 = np.zeros(x0.shape)
    y2 = np.zeros(x0.shape)

    def fcost(x): 
        return func(x) + sigma*np.sum((x-currentH)**2) 
   
    print("Initial cost with sign: %f without sign = %f" %(fcost(xt), fcost(currentH)))
    print("reconstruction: %f" %(func(x0)))
    print("Encoder error: %f" %(sigma*np.sum((x0-currentH)**2)))
    # pdb.set_trace()

    def fx(x): # Fix v, solve for x
        current_x = np.reshape(x, (batch_size, latent_dim))
        for layer_param in decoder_params:            
            current_x = np.tanh(np.dot(current_x, layer_param[0]) + layer_param[1])
        
        return np.sum((current_x - x_batch)**2) + sigma*np.sum(np.power(x-currentH,2)) 
        + np.dot(y1, vt - np.ones(xt.shape) + x) + 0.5*rho*(np.sum(np.power(vt + x - np.ones(xt.shape), 2))) 
        + np.dot (y2, np.multiply(vt, x+np.ones(x0.shape))) + 0.5*rho*np.sum(np.power(np.multiply(vt, x+np.ones(x0.shape)),2))

    def fv(x): # Fix x, solve for v
        return np.dot(y1, x - np.ones(x0.shape) + xt) + 0.5*rho*np.sum(np.power(x - np.ones(x0.shape), 2))
        + np.dot(y2, np.multiply(x, np.ones(x0.shape)+xt)) + 0.5*rho*np.sum(np.power(np.multiply(x, np.ones(x0.shape)+xt),2))

    
    # Define the lower and upper bounds for fx, i.e., -1 <= x <= 1        
    xbounds= [(-1,1) for i in range(n)]
    vbounds = [(0, 2) for i in range(n)]
    # Now, let the iterations begin
    converged = False
    iter = 0
    
    while iter < maxIter and not converged:
        # Fix v, minimize x        
        print('----Updating x ')                       
        algo = pg.algorithm(pg.bee_colony(20,20))
        # uda = pg.nlopt("slsqp")
        # algo = pg.algorithm(uda)
        prob = x_problem(x_batch, batch_size, latent_dim, decoder_params, currentH, vt, y1, y2, sigma, rho)
        pop = pg.population(prob, 50)
        # pop=pg.population(prob, 20, 20 )
        # pop = algo.evolve(pop)
        x = pop.champion_x        
        # x_res = minimize(fx, xt, bounds = xbounds)
        # x = x_res.x

        # print min(x), max(x)
        # Fix x, update v
        print('----Updating v')
        v_res = minimize(fv, vt, bounds = vbounds)
        v = v_res.x

        # print min(v), max(v)
        y1 = y1 + rho*(v + x - np.ones(x0.shape))
        y2 = y2 + rho*(np.multiply(v, np.ones(x0.shape) + x))

        print("Iter: %d , fx = %.3f, prev_fx = %.3f, x diff: %.3f, rho = %.3f reconstruction: %f" 
              %(iter, fcost(x), fcost(xt), norm(x - xt), rho, func(x)))
        print("reconstruction: %f" %(func(x)))
        print("Encoder error: %f" %(sigma*np.sum((x-currentH)**2)))
        # Check for convergence
        # if iter > 4 and ((norm(v - vt) < 1e-6 and abs(func(x) - func(xt) < 1e-6)) or (n-np.dot(xt, vt))**2<1.5):
        if iter > 1 and ( (norm(x - xt) < 1e-6  or                           
                          fcost(x) > fcost(xt)  ) ):
            converged = True
            print('--------Using LINF  - Converged---------')            
            return xt #np.ones(x0.shape) - vt
        
        #print (xt)
        rho = rho*1.1
        xt = x
        vt = v

        
        iter = iter + 1

    return  xt #np.ones(x0.shape) - vt
#



