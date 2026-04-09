import numpy as np
from typing import Callable

class MCMC:
    def __init__(self, nwalkers: int, nsteps: int, likelihood_func: Callable, args: tuple, step_sizes: np.ndarray | None = None):
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.likelihood_func = lambda x: likelihood_func(x, *args)
        self.step_sizes = step_sizes

    def _step(self, position, likelihood):
        proposed_positions = np.random.normal(loc = position, scale = self.step_sizes, size = (self.nwalkes, position.size)) # Draw Gaussian samples of the given step sizes around the current position for every walker
        
        proposed_likelihoods = np.apply_along_axis(self.likelihood_func, 1, proposed_positions) # Has shape (nwalkers,)
        accept_mask = (proposed_positions > likelihood) # likelihood MAXIMIZES!
        inverted_accept_mask = np.invert(accept_mask)
        
        # For those less likely points, still accept them with a random chance
        accept_mask[inverted_accept_mask] = (np.random.uniform(0, 1, size = (inverted_accept_mask.sum(),)) <= (proposed_likelihoods[inverted_accept_mask] / likelihood[inverted_accept_mask]))

        position[accept_mask] = proposed_positions[accept_mask]
        likelihood[accept_mask] = proposed_likelihoods[accept_mask]
        
        return position, likelihood
    
    
