import numpy as np
from scipy.special import gammainc
from scipy.optimize import minimize, brentq
from scipy.differentiate import derivative
from typing import Callable
import os

GLOBAL_PRIOR_RANGE = [[1., 1500.], [1., 1500.], [0., 0.5]]
# GLOBAL_PRIOR_RANGE = [[1., 1500.], [1., 1500.], [0., 0.5]] #STANDARD


## FUNCTIONS FOR ONEHALO LIKELIHOODS

def mod_gaussian(v, sigma1, sigma2, lambda_): #for regular (i.e. untransformed velocity) input
    # normalization done to break degeneracy between lambda_ and sigma_i as much as possible

    norm = 1 / (((1- lambda_) * sigma1 + lambda_ * sigma2)* np.sqrt(2 * np.pi))
    vsq = -1 * np.square(v) * 0.5
    return norm * ((1-lambda_) * np.exp(vsq / sigma1**2) + lambda_ * np.exp(vsq / sigma2**2))

def mod_gaussian_loglambda(v, sigma1, sigma2, lambda_): #for lambda in log scale
    lambda_10 = 10**lambda_
    norm = 1 / (((1- lambda_10) * sigma1 + lambda_10 * sigma2)* np.sqrt(2 * np.pi))
    vsq = -1 * np.square(v) * 0.5
    return norm * ((1-lambda_10) * np.exp(vsq / sigma1**2) + lambda_10 * np.exp(vsq / sigma2**2))

def mod_gaussian_integral(sigma1,sigma2,lambda_,x_i,x_f):
    integral, _ = Romberg(x_i, x_f, lambda x: mod_gaussian(x,sigma1,sigma2,lambda_)) 
    return integral

def mod_gaussian_integral_loglambda(sigma1,sigma2,lambda_,x_i,x_f):
    integral, _ = Romberg(x_i, x_f, lambda x: mod_gaussian_loglambda(x,sigma1,sigma2,lambda_)) 
    return integral

def log_prior(theta):
    sigma_1, sigma_2, lambda_ = theta
    
    if None in GLOBAL_PRIOR_RANGE[0]: # sigma_1 must be larger than sigma_2
        sigma_1_prior = sigma_2 < sigma_1 <= GLOBAL_PRIOR_RANGE[0][1]
    else:
        sigma_1_prior = GLOBAL_PRIOR_RANGE[0][0] <= sigma_1 <= GLOBAL_PRIOR_RANGE[0][1]
    
    if None in GLOBAL_PRIOR_RANGE[1]: #sigma_2 must be smaller than sigma_1
        sigma_2_prior = GLOBAL_PRIOR_RANGE[1][0] <= sigma_2 < sigma_1
    else:
        sigma_2_prior = GLOBAL_PRIOR_RANGE[1][0] <= sigma_2 <= GLOBAL_PRIOR_RANGE[1][1]
    
    lambda_prior = GLOBAL_PRIOR_RANGE[2][0] <= lambda_ <= GLOBAL_PRIOR_RANGE[2][1]

    if sigma_1_prior and sigma_2_prior and lambda_prior:
        return 0.0
    
    return -np.inf

def log_prior_loglambda(theta):
    sigma_1, sigma_2, lambda_ = theta
    if sigma_2 < sigma_1 <= 1500. and 1. <= sigma_2 < sigma_1 and -np.inf < lambda_ <= 0.:
        return 0.0
    return -np.inf

def mod_gaussian_log_likelihood_binned(params, bin_edges, bin_heights): #full with prior
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    sigma1, sigma2, lambda_ = params
    hist_area = np.sum(bin_heights) 
    fit_integral = 1.
    A = hist_area / fit_integral
    
    log_L = 0
    for i in range(1,len(bin_edges)):
        f_b = A * mod_gaussian_integral(sigma1,sigma2,lambda_,bin_edges[i-1],bin_edges[i])
        
        #penalize negative values and zero
        if f_b <= 0:
            return 10**11
        
        n_b = bin_heights[i-1] #* bin_width
        log_L += (n_b * np.log(f_b)) - f_b  # herein lies the only difference with neg_log_likelihood
    
    return log_L

def mod_gaussian_log_likelihood(params, data, loglambda, single_gauss): #full with prior
    prior = log_prior_loglambda if loglambda else log_prior
    lp = prior(params)
    if not np.isfinite(lp):
        return -np.inf

    if single_gauss: params[-1] = 0

    mu_func = mod_gaussian_loglambda if loglambda else mod_gaussian
    mu_i = mu_func(data, *params)
    mu_i[mu_i < 0] = 0. # cast negative values to 0, raises errors otherwise
    return np.sum(np.log(mu_i)) + 1. #+1. for integral, chose log10 to stay consistent

def mod_gaussian_neg_log_likelihood_binned(params, bin_edges, bin_heights):
    sigma1, sigma2, lambda_ = params
    hist_area = np.sum(bin_heights) 
    # I do not know where this comes from, integral of mod_gaussian dv from -inf to inf is 1.  
    # fit_integral = (sigma1 * np.sqrt(2 * np.pi)) *(3*sigma2 + 1 + 105*lambda_) 
    fit_integral = 1.
    A = hist_area / fit_integral
    
    neg_log_L = 0
    for i in range(1,len(bin_edges)):
        f_b = A * mod_gaussian_integral(sigma1,sigma2,lambda_,bin_edges[i-1],bin_edges[i])
        
        #penalize negative values and zero
        if f_b <= 0:
            return 10**11
        
        n_b = bin_heights[i-1] #* bin_width
        neg_log_L += f_b - (n_b * np.log(f_b))
    
    return neg_log_L

def mod_gaussian_neg_log_likelihood(params, data):
    # "binning" such that each bin contains either 1 or 0 points. Poisson likelihood reduces to this.
    #Data must be x_i values
    sigma1, sigma2, lambda_ = params
    fit_integral = 1. #verified

    return -1 * np.sum(np.log(mod_gaussian(data, sigma1, sigma2, lambda_))) + fit_integral
    
# GOODNESS OF FIT STATISTICS (OUTDATED)
def _Gstat(O,E):
    """G-test statistic.

    Args:
        O (array): Observed counts. MUST be integers.
        E (array): Expected counts under the null hypothesis e.g. the model predictions.

    Returns:
        G (float): The G-statistic value
    """
    G = 0
    for i, Oi in enumerate(O):
        if Oi == 0:
            continue
        G += Oi * np.log(Oi/E[i])
    return 2*G

def _Qval(x,k):
    """Find the Q-value aka p-value from a given chi^2 distributed statistic.

    Args:
        x (float): Statistic value
        k (int): The degrees of freedom of the problem.

    Returns:
        Q (float): The Q-value of the statistic given the degrees of freedom.
    """

    #gammainc in scipy is the *regularized* incomplete lower gamma function so /gamma(k/2) is implicit
    return 1 - gammainc(k/2,x/2)

def get_Qvalue(param_dict, bin_heights, bin_edges, integral_func = mod_gaussian_integral, sanitycheck = False):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_area=np.sum(bin_heights)

    model = hist_area * np.array([integral_func(param_dict['sigma_1'], param_dict['sigma_2'], param_dict['lambda'],
                                                               bin_edges[i], bin_edges[i+1]) for i in range(len(bin_centers))])
    
    Gval = _Gstat(bin_heights, model) 
    if sanitycheck:
        print(f"{sum(model) = }, {sum(bin_heights) = }")
        print(f'{Gval = }')
    #dof = 4 since three parameters and 1 extra less since if N -1 bins are filled we know exactly the Nth bin count
    return _Qval(Gval, len(bin_heights) - 4)

def get_Gstat(param_dict, bin_heights, bin_edges, integral_func = mod_gaussian_integral):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_area=np.sum(bin_heights)

    model = hist_area * np.array([integral_func(param_dict['sigma_1'], param_dict['sigma_2'], param_dict['lambda'],
                                                               bin_edges[i], bin_edges[i+1]) for i in range(len(bin_centers))])
    
    return _Gstat(bin_heights, model) 


# TODO: i think delete
def ddsigma1_likelihood(v, sigma_1, sigma_2, lambda_):
    numerator = (
        np.exp(-(v**2 * (sigma_1**2 + sigma_2**2)) / (2 * sigma_1**2 * sigma_2**2))
        * (-1 + lambda_)
        * (
            np.exp(v**2 / (2 * sigma_1**2)) * sigma_1**3 * lambda_
            + np.exp(v**2 / (2 * sigma_2**2))
            * (
                -sigma_1**3 * (-1 + lambda_)
                + v**2 * (sigma_1 * (-1 + lambda_) - sigma_2 * lam)
            )
        )
    )

    denominator = (
        np.sqrt(2 * np.pi)
        * sigma_1**3
        * (-sigma_1 * (-1 + lambda_) + sigma_2 * lambda_) ** 2
    )

    return numerator / denominator

def ddsigma2_likelihood(v, sigma_1, sigma_2, lambda_):
    numerator = (
        lambda_
        * (
            np.exp(-v**2 / (2 * sigma_1**2)) * (-1 + lambda_)
            + (
                np.exp(-v**2 / (2 * sigma_2**2))
                * (-sigma_2**3 * lambda_ + v**2 * (sigma_1 - sigma_1 * lambda_ + sigma_2 * lambda_))
            )
            / sigma_2**3
        )
    )

    denominator = (
        np.sqrt(2 * np.pi)
        * (sigma_1 - sigma_1 * lambda_ + sigma_2 * lambda_) ** 2
    )

    return numerator / denominator

def ddlambda_likelihood(v, sigma_1, sigma_2, lambda_):
    numerator = (
        np.exp(-(v**2 * (sigma_1**2 + sigma_2**2)) / (2 * sigma_1**2 * sigma_2**2))
        * (
            np.exp(v**2 / (2 * sigma_1**2)) * sigma_1
            - np.exp(v**2 / (2 * sigma_2**2)) * sigma_2
        )
    )

    denominator = (
        np.sqrt(2 * np.pi)
        * (-sigma_1 * (-1 + lambda_) + sigma_2 * lambda_) ** 2
    )

    return numerator / denominator




    # Then estimating the error bounds
    # L_ratio_func = lambda x: (log_likelihood_func(x) / L_best) - 1.01
    
    ### OLD and inefficient

    # while L_current >= L_thresh:
    #     # Save previous value
    #     previous_param_value = perturbed_params[param_idx]
    #     L_previous = L_current
    #     # Step parameter
    #     perturbed_params[param_idx] += dir * step
    #     L_current = log_likelihood_func(perturbed_params)

    #     if not np.isfinite(L_current):
    #         # Assume the previous step was the best we could get within prior range
    #         # so return the relevant prior bound as the value
    #         prior_idx = 0 if dir == -1 else 1
    #         return prior_ranges[param_idx][prior_idx], params

    #     # It might be that we step further into a maximum, in that case update the likelihood and parameter values accordingly
    #     if L_current > L_best:
    #         L_best = L_current
    #         L_thresh = 1.01 * L_best # so that we know to perturb around this new maximum
    #         params = np.copy(perturbed_params)
        

    # thresh is not precisely halfway between the points, 
    # # this is more accurate but still assumes linearity
    # y = L_current - L_thresh
    # x = L_thresh - L_previous
    # a = (y + x) / (perturbed_params[param_idx] - previous_param_value)
    # b = L_thresh - x - a * previous_param_value
    # param_bound = (L_thresh - b) / a


### FOR PERTURBING AROUND THE OPTIMUM FROM MCMC
def sample_for_brent(params, found_likelihood, log_likelihood_func, single_gauss = False):
    if single_gauss:
        step_sizes = [100.]
    else:
        step_sizes = [100., 100., 0.1] # high valued steps
    Lthresh = 1.01 * found_likelihood

    bounds = [[] for _ in range(len(step_sizes))]

    for param_idx,step in enumerate(step_sizes):
        param_insert = params.copy()
        current_param_value = params[param_idx]
        # perturbation_of_parameter = np.arange(params[param_idx] - 100*step, params[param_idx] + 100*step, step = step).reshape(200,1)
        # step up
        prior_flag = False
        Lcurrent = found_likelihood

        while Lcurrent < Lthresh:
            current_param_value += step
            if current_param_value >= GLOBAL_PRIOR_RANGE[param_idx][1]:
                # current_param_value -= step # step back to previous
                prior_flag = True
                break

            np.put(param_insert, param_idx, current_param_value)
            Lcurrent = log_likelihood_func(param_insert)

        param_up = current_param_value if not prior_flag else GLOBAL_PRIOR_RANGE[param_idx][1]

        # step down, so reset some stuff
        prior_flag = False
        np.put(param_insert, param_idx, params[param_idx])
        current_param_value = params[param_idx]
        Lcurrent = found_likelihood
        while Lcurrent < Lthresh:
            current_param_value -= step
            if current_param_value <= GLOBAL_PRIOR_RANGE[param_idx][0]:
                # current_param_value -= step # step back to previous
                prior_flag = True
                break

            np.put(param_insert, param_idx, current_param_value)
            Lcurrent = log_likelihood_func(param_insert)

        param_down = current_param_value if not prior_flag else GLOBAL_PRIOR_RANGE[param_idx][0]

        bounds[param_idx] = [param_down, param_up]
    
    return bounds


def _perturb_params(params: np.array, log_likelihood_func: Callable[[np.array], float], single_gauss = False):
    """ Perturb parameter value, either upwards or downwards, and return parameter value when likelihood threshold is reached.

    Args:
        params (list or array): list or array containing the parameter values in order [sigma1, sigma2, lambda]
        param_idx (int): index pointing to the parameter to perturb in the params list
        step (float): Stepsize for parameter perturbation
        L (float): Likehood at the best point
        L_thresh (float): Threshold likelihood to reach.
        dir (int, optional): Whether to perturb the parameter upwards or downwards; 1 for upwards, -1 for downwards. Defaults to 1.
        log_likelihood_func (function, optional): Log likelihood of the model. Defaults to mod_gaussian_log_likelihood. NOTE that this is an L-maximizing function since it is negative numbers we want to be as near to 0 as possible!

    Returns:
        float: parameter value at the likelihood threshold #TODO update
    """
    prior_ranges = GLOBAL_PRIOR_RANGE.copy()

    if None in GLOBAL_PRIOR_RANGE[0]: # sigma_1
        prior_ranges[0][0] = params[1] 
    
    if None in GLOBAL_PRIOR_RANGE[1]: #sigma_2
        prior_ranges[1][1] = params[0]

    # we need the negative of the function we put in so that we can minimize
    ll_func_for_minimize = lambda x: -log_likelihood_func(x)
    opt_params = minimize(ll_func_for_minimize, x0 = params, bounds = prior_ranges).x
    L_best = log_likelihood_func(opt_params)

    # function which perturb only 1 parameter at a time, need to do it like this because the argument position changes
    perturb_sigma1 = lambda x: (log_likelihood_func([x, opt_params[1], opt_params[2]]) - (1.01 * L_best))
    perturb_sigma2 = lambda x: (log_likelihood_func([opt_params[0], x, opt_params[2]]) - (1.01 * L_best))
    perturb_lambda = lambda x: (log_likelihood_func([opt_params[0], opt_params[1], x]) - (1.01 * L_best))
    perturb_funcs = [perturb_sigma1, perturb_sigma2, perturb_lambda]

    # get the bounds for the root finder by taking large samples. note that this is actually the bottleneck of the code
    brent_bounds = sample_for_brent(opt_params, -1*L_best, ll_func_for_minimize, single_gauss = single_gauss)

    param_bounds = [[] for _ in range(len(params))]

    # NOTE runtime error means it did not converge, value error means that the signage of the function 
    # didnt change in the interval so there was no root in it    
    if not single_gauss:
        for param_idx in range(len(params)):
            try:
                lower_bound = brentq(perturb_funcs[param_idx], brent_bounds[param_idx][0], opt_params[param_idx])
            except (RuntimeError, ValueError):
                lower_bound = GLOBAL_PRIOR_RANGE[param_idx][0]
            try:
                upper_bound = brentq(perturb_funcs[param_idx], opt_params[param_idx], brent_bounds[param_idx][1])
            except (RuntimeError, ValueError):
                upper_bound = GLOBAL_PRIOR_RANGE[param_idx][1]
            
            param_bounds[param_idx] = [lower_bound, upper_bound]

    #TODO: single_gauss will not work for joint models, do we want it to?
    else:
        # sigma_1 needs this root-finding optimization
        try:
            lower_bound = brentq(perturb_funcs[0], brent_bounds[0][0], opt_params[0])
        except (RuntimeError, ValueError):
            lower_bound = GLOBAL_PRIOR_RANGE[0][0]
        try:
            upper_bound = brentq(perturb_funcs[0], opt_params[0], brent_bounds[0][1])
        except (RuntimeError, ValueError):
            upper_bound = GLOBAL_PRIOR_RANGE[0][1]
        
        param_bounds[0] = [lower_bound, upper_bound]
        
        # the other two parameters do not need it and just get the prior range as errors
        for param_idx in range(1,len(params)):
            param_bounds[param_idx] = GLOBAL_PRIOR_RANGE[param_idx]

    return param_bounds, opt_params

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
    # ^that is gonna suck so fucking much

    bounds, best_params = _perturb_params(params, log_likelihood_func, single_gauss = single_gauss)
    param_dict = {'sigma_1':best_params[0], 'sigma_2':best_params[1], 'lambda': best_params[2], 'errors':[[] for i in range(len(best_params))]}

    for i in range(len(best_params)):
        param_down, param_up = bounds[i]
        param_dict['errors'][i] = [np.abs(best_params[i] - param_down), np.abs(best_params[i] - param_up)] # errors are the parameter differences

    return param_dict


# GENERAL FUNCTIONS AND OWN NUMERICAL METHODS USED THROUGHOUT

def Romberg(a,b,func,m=6):
    """Romberg's algorithm for integration using Richardson extrapolation.

    Args:
        a (float): Lower bound of the integral (incl).
        b (float): upper bound of the integral (incl).
        func (method): function to integrate. Must have only one callable variable which is the one we integrate over.
        m (int, optional): Order of extrapolation. Defaults to 6.

    Returns:
        tuple of floats: (estimate of the integral, estimated error on integral by absolute difference between current and previous best guess)
    """

    h = (b-a) 
    r = np.zeros(m+1) #this array will hold our estimates
    r[0] = 0.5 * h * (func(a) + func(b)) #first guess
    Np = 1
    for i in range(1,m+1): #loop over rows
        h *= 0.5
        x = a + h #point to evaluate the function at, a little ahead of the previous
        for _ in range(Np):
            r[i] += func(x)
            x += 2*h
    
        r[i] = 0.5 * r[i-1] + h*r[i]

        Np *= 2

    Np4 = 1
    for i in range(m+1): #loop over columns
        Np4 *= 4
        for j in range(0,m - i): #loop over relevant rows
            r[j] = (Np4 * r[j+1] - r[j]) / (Np4 - 1) #update rule for Richardson extrapolation. Np4 = 2^(2j) = 4^j


    return r[0],np.abs(r[0] - r[1])

def modified_logspace(start, stop, num, base = 10):
    stop = np.log10(stop)
    if start == 0:
        # res = np.zeros(num)
        # start = np.log10(0.1)
        # y = np.power(base, np.linspace(start, stop, num - 1))
        # res[1:] = y
        res = np.logspace(0, np.log10(2.5 + 1), 20) - 1
    else:
        start = np.log10(start)
        res = np.power(base, np.linspace(start, stop, num))

    return res

def BIC(L, n, k = 3):
    #L is assumed already as a logarithm and as a MAXIMUM
    return (k * np.log(n)) - 2 * L

def mkdir_if_non_existent(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def flatten(xss):
    # Flatten a python list using list comprehension
    return [x for xs in xss for x in xs]

def rice_bins(N):
    return 2 * int(np.cbrt(N))

