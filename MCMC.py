import numpy as np
from typing import Callable
from tqdm import tqdm

class MCMC:
    def __init__(self, nwalkers: int, likelihood_func: Callable,
                 args: tuple = (), step_sizes: np.ndarray | None = None):
        self.nwalkers = nwalkers
        self.likelihood_func = lambda x: likelihood_func(x, *args)
        self.step_sizes = step_sizes
        self.accepted_steps = 0
    
    def _calculate_likelihood(self, pos: np.ndarray):
        # For every walker
        return np.apply_along_axis(self.likelihood_func, 0, pos)

    def _step(self):

        # Draw Gaussian samples of the given step sizes around the current position for every walker
        proposed_positions = np.random.normal(loc = self.position, scale = self.step_sizes, size = (self.position.shape[0], self.nwalkers)) 
        proposed_likelihoods = self._calculate_likelihood(proposed_positions) # Has shape (nwalkers,)
        
        accept_mask = (proposed_likelihoods < self.likelihood) # we want a minimum -log(L)
        inverted_accept_mask = np.invert(accept_mask)
    
        # For those less likely points, still accept them with a random chance
        # we have the negative log-likelihood, so we need to take the ratio of the exponents and add minus signs
        accept_mask[inverted_accept_mask] = (np.random.uniform(0, 1, size = (inverted_accept_mask.sum(),)) \
                                             <= np.exp((-proposed_likelihoods[inverted_accept_mask] + self.likelihood[inverted_accept_mask])))
         
        self.position[:, accept_mask] = proposed_positions[:, accept_mask]
        self.likelihood[accept_mask] = proposed_likelihoods[accept_mask]
        self.accepted_steps += accept_mask.sum()
        

    def run_mcmc(self, initial_position: np.ndarray, nsteps: int, verbose: bool = True):
        self.nsteps = nsteps

        if self.step_sizes is None:
            self.step_sizes = np.full(initial_position.shape[0], 0.05)[:, np.newaxis]

        self.position = initial_position
        self.likelihood = self._calculate_likelihood(self.position)

        self.chains = np.empty(shape = (*self.position.shape, self.nsteps))
        self.likelihood_chain = np.empty(shape = (self.nwalkers, self.nsteps))

        steps_taken = 0
        if verbose:
            with tqdm(total = self.nsteps) as pbar:
                while steps_taken < self.nsteps:
                    self.chains[..., steps_taken] = self.position
                    self.likelihood_chain[..., steps_taken] = self.likelihood

                    self._step()
                    steps_taken += 1
                    pbar.update(1)

        else:
            while steps_taken < self.nsteps:
                self.chains[..., steps_taken] = self.position
                self.likelihood_chain[:, steps_taken] = self.likelihood

                self._step()
                steps_taken += 1


    def get_likelihoods(self):
        return self.likelihood_chain
    
    def get_chain(self):
        return self.chains
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #2D rosenbrock

    def rosenbrock_2D(p):
        x,y = p
        return (1-x)**2 + 100*(y -x)**2

    nwalkers, steps = 3, 1000
    rosenbrock_init = np.random.normal(loc = 1, scale = 1, size = (2, nwalkers))

    mcmc = MCMC(nwalkers = nwalkers, likelihood_func = rosenbrock_2D)
    mcmc.run_mcmc(rosenbrock_init, nsteps=steps)

    samples = mcmc.get_chain()
    likelihoods = mcmc.get_likelihoods()

    fig, axes = plt.subplots(nrows = 3)
    for i, ax in enumerate(axes[:-1]):
        chain = samples[:,i,:]
        ax.plot(chain.T)
        ax.set_xlabel('Steps', fontsize = 15)
        ax.set_ylabel('Parameter value', fontsize = 15)
    axes[-1].plot(likelihoods.T)
    plt.tight_layout()
    plt.show()

    best_arg = np.unravel_index(np.argmin(likelihoods), likelihoods.shape)
    best_params = np.array([samples[i, *best_arg] for i in range(2)])
    print(best_params)

    #ND Rosenbrock

    def rosenbrock_ND(p):
        val = 0
        for i,pi in enumerate(p[:-1]):
            val += 100* (p[i+1] - pi**2)**2 + (1- pi)**2
        
        return val

    nwalkers, steps = 15, 1000
    nparams = 6
    rosenbrock_init = np.random.normal(loc = 1, scale = 1, size = (nparams, nwalkers))

    mcmc = MCMC(nwalkers = nwalkers, likelihood_func = rosenbrock_ND)
    mcmc.run_mcmc(rosenbrock_init, nsteps = steps)

    samples = mcmc.get_chain()
    likelihoods = mcmc.get_likelihoods()

    fig, axes = plt.subplots(nrows = nparams + 1, figsize = (8, 16))
    for i, ax in enumerate(axes[:-1]):
        chain = samples[:,i,:]
        ax.plot(chain.T)
        ax.set_xlabel('Steps', fontsize = 15)
        ax.set_ylabel('Parameter value', fontsize = 15)
    axes[-1].plot(likelihoods.T)
    plt.tight_layout()
    plt.show()

    best_arg = np.unravel_index(np.argmin(likelihoods), likelihoods.shape)
    best_params = np.array([samples[i, *best_arg] for i in range(nparams)])
    print(best_params)

