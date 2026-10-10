import numpy as np 

def GBM_paths(S0, r, q, sigma, T, N, M, seed = None):
    
    """
    This function permits to generate Geometric Brownian Motion for pricing 
    """

    if seed != None:
        np.random.seed(seed)
    
    dt = T/M
    mu = (r - q - 0.5 * sigma**2)*dt
    diffusion = sigma * np.sqrt(dt) * np.random.standard_normal((M, N))

    log_paths = np.vstack([np.zeros(N), np.cumsum(mu + diffusion, axis=0)])
    return S0 * np.exp(log_paths)