import numpy as np
from typing import Callable

GLOBAL_PRIOR_RANGE = [[1., 1500.], [1., 1500.], [0., 0.5]]

def perturb_around_likelihood(L: float, params: np.array, log_likelihood_func: Callable[[np.array], float], single_gauss: bool = False) -> dict:
    """ Given a maximum from MCMC, perturb around this found maximum in small steps to estimate the error and refine the estimate.

    Args:
        L (float): Likelihood value at the best found parameter set
        params (np.array): Best found parameter set
        log_likelihood_func (function, optional): Log likelihood of the model. Defaults to mod_gaussian_log_likelihood.
        single_gauss (bool, optional): Controls whether we "turn off" the lambda parameter, i.e. fix it to 0.

    Returns:
        dict: dictionary of parameter values and error estimates
    """
    
    #TODO: fix for joint model, there will be more parameters and this is hardcoded to 3 parameters now

    param_steps = [1., 1., 0.1]  #NOTE: subject to change

    L_thresh = 1.01 * L #1%
    if not single_gauss:
        param_dict = {'sigma_1':params[0], 'sigma_2':params[1], 'lambda': params[2], 'errors':[[] for i in range(len(params))]}
        for i in range(len(params)):
            param_up, params = _perturb_param(params, i, param_steps[i], L, L_thresh, dir = 1, log_likelihood_func = log_likelihood_func)
            param_down, params = _perturb_param(params, i, param_steps[i], L, L_thresh, dir = -1, log_likelihood_func = log_likelihood_func)

            param_dict['errors'][i] = [np.abs(params[i] - param_down), np.abs(params[i] - param_up)] # errors are the parameter differences

        return param_dict

    else:
        # If we do not use lambda and sigma2, only perturb for sigma1
        param_dict = {'sigma_1':params[0], 'sigma_2':params[1], 'lambda': 0., 'errors':[[] for i in range(len(params))]}

        param_up, params = _perturb_param(params, 0, param_steps[0], L, L_thresh, dir = 1, log_likelihood_func = log_likelihood_func)
        param_down, params = _perturb_param(params, 0, param_steps[0], L, L_thresh, dir = -1, log_likelihood_func = log_likelihood_func)

        param_dict['errors'][0] = [np.abs(params[0] - param_down), np.abs(params[0] - param_up)] # errors are the parameter differences
        param_dict['errors'][1] = [np.abs(params[1] - GLOBAL_PRIOR_RANGE[1][0]), np.abs(params[1] - GLOBAL_PRIOR_RANGE[1][1])] # sigma_2 over entire prior range
        param_dict['errors'][-1] = [0.,0.] # fix lambda at 0 with no error
        
        return param_dict

def _perturb_param(params: np.array, param_idx: int, step: float, L: float, L_thresh:float, dir: int, log_likelihood_func: Callable[[np.array], float]):
    """ Perturb parameter value, either upwards or downwards, and return parameter value when likelihood threshold is reached.

    Args:
        params (list or array): list or array containing the parameter values in order [sigma1, sigma2, lambda]
        param_idx (int): index pointing to the parameter to perturb in the params list
        step (float): Stepsize for parameter perturbation
        L (float): Likehood at the best point
        L_thresh (float): Threshold likelihood to reach.
        dir (int, optional): Whether to perturb the parameter upwards or downwards; 1 for upwards, -1 for downwards. Defaults to 1.
        log_likelihood_func (function, optional): Log likelihood of the model. Defaults to mod_gaussian_log_likelihood.

    Returns:
        float: parameter value at the likelihood threshold
    """
    L_current = L
    L_best = L
    perturbed_params = np.copy(params)

    prior_ranges = GLOBAL_PRIOR_RANGE.copy()

    if None in GLOBAL_PRIOR_RANGE[0]: # sigma_1
        prior_ranges[0][0] = params[1] 
    
    if None in GLOBAL_PRIOR_RANGE[1]: #sigma_2
        prior_ranges[1][1] = params[0]

    while L_current >= L_thresh:
        # Save previous value
        previous_param_value = perturbed_params[param_idx]
        L_previous = L_current
        # Step parameter
        perturbed_params[param_idx] += dir * step
        L_current = log_likelihood_func(perturbed_params)

        if not np.isfinite(L_current):
            # Assume the previous step was the best we could get within prior range
            # so return the relevant prior bound as the value
            prior_idx = 0 if dir == -1 else 1
            return prior_ranges[param_idx][prior_idx], params

        # It might be that we step further into a maximum, in that case update the likelihood and parameter values accordingly
        if L_current > L_best:
            L_best = L_current
            L_thresh = 1.01 * L_best # so that we know to perturb around this new maximum
            params = np.copy(perturbed_params)
        

            ##BELOW: when this works it is really nice, but it might take VERY long
            # newstep = step
            # perturbed_params[param_idx] = previous_param_value # reset
            # # take succesively smaller steps until L_current is finite again
            # while not np.isfinite(L_current):
            #     newstep *= 0.5
            #     perturbed_params[param_idx] += dir * newstep
            #     L_current = log_likelihood_func(perturbed_params)
            
            # step = newstep # in case we need to continue, 
            # don't be guaranteed to make the same mistake of overstepping to inf

    # thresh is not precisely halfway between the points, 
    # this is more accurate but still assumes linearity
    y = L_current - L_thresh
    x = L_thresh - L_previous
    a = (y + x) / (perturbed_params[param_idx] - previous_param_value)
    b = L_thresh - x - a * previous_param_value
    param_bound = (L_thresh - b) / a

    return param_bound, params
