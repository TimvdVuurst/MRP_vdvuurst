import numpy as np
from scipy.special import gammainc
from scipy.optimize import minimize, brentq
from scipy.differentiate import derivative
from typing import Callable
import os

# sigma_1, sigma_2, lambda
GLOBAL_PRIOR_RANGE = [[5., 1500.], [5., 1500.], [0., 0.5]]

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
    return np.sum(np.log10(mu_i)) + 1. #+1. for integral, chose log10 to stay consistent

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

#### VECTORIZED LOG(L) functions
def log_prior_vec(theta):
    # VECTORIZED this works well and is quick
    sigma_1, sigma_2, lambda_ = theta
    
    if None in GLOBAL_PRIOR_RANGE[0]: # sigma_1 must be larger than sigma_2
        sigma_1_prior = sigma_2 < sigma_1 <= GLOBAL_PRIOR_RANGE[0][1]
    else:
        sigma_1_prior = (GLOBAL_PRIOR_RANGE[0][0] <= sigma_1) * (sigma_1 <= GLOBAL_PRIOR_RANGE[0][1])
    
    if None in GLOBAL_PRIOR_RANGE[1]: #sigma_2 must be smaller than sigma_1
        sigma_2_prior = GLOBAL_PRIOR_RANGE[1][0] <= sigma_2 < sigma_1
    else:
        sigma_2_prior = (GLOBAL_PRIOR_RANGE[1][0] <= sigma_2) * (sigma_2 <= GLOBAL_PRIOR_RANGE[1][1])
    
    lambda_prior = (GLOBAL_PRIOR_RANGE[2][0] <= lambda_) * (lambda_ <= GLOBAL_PRIOR_RANGE[2][1])

    cond = np.logical_and(np.logical_and(sigma_1_prior, sigma_2_prior),lambda_prior)
    prior = np.where(cond, 0,-np.inf)

    # so that it can also be used unvectorized if we want
    if theta.size == 3:
        return np.float32(prior)
    return prior

def mod_gaussian_vec(min_half_v_sq, sigma1_sq, sigma2_sq, lambda_, one_min_lambda): 
    #for SQUARED velocity input!!, vectorized over large arrays of sigma1, sigma2 and lambda with operations pre-calced
    # normalization done to break degeneracy between lambda_ and sigma_i as much as possible
    p = 0
    for i in range(3): # loop over all vx, then vy then vz. We do it like this to reconcile the shape difference (Ngal = shape(Vi))
        p += np.sum(np.log((one_min_lambda * np.exp(min_half_v_sq[:,i] / sigma1_sq) + #TODO: pre-calculate inverses to reduce operations further 
                             lambda_ * np.exp(min_half_v_sq[:,i] / sigma2_sq)))) # take the log so we can sum
    return p 

def mod_gaussian_vec_for_plot(min_half_v_sq, sigma1_sq_inv, sigma2_sq_inv, lambda_, one_min_lambda, norm): 
    #for SQUARED velocity input!!, vectorized over large arrays of sigma1, sigma2 and lambda with operations pre-calced
    # normalization done to break degeneracy between lambda_ and sigma_i as much as possible
    p = np.zeros_like(min_half_v_sq)
    for i in range(3): # loop over all vx, then vy then vz. We do it like this to reconcile the shape difference (Ngal = shape(Vi))
        p[:,i] += norm * (one_min_lambda * np.exp(min_half_v_sq[:,i] * sigma1_sq_inv) +\
                             lambda_ * np.exp(min_half_v_sq[:,i] * sigma2_sq_inv)) 
    return p.flatten()

def log_mod_gaussian_vec(min_half_v_sq, sigma1_sq_inv, sigma2_sq_inv, lambda_, one_min_lambda):
    #for SQUARED velocity input!!, vectorized over large arrays of sigma1, sigma2 and lambda with operations pre-calced
    # normalization done to break degeneracy between lambda_ and sigma_i as much as possible
    p = 0
    min_half_v_sq_sigma_1 = min_half_v_sq * sigma1_sq_inv[:, np.newaxis]
    min_half_v_sq_sigma_2 = min_half_v_sq * sigma2_sq_inv[:, np.newaxis]
    # Broadcast to match the shapes
    p = np.sum(min_half_v_sq + np.log(one_min_lambda[:, np.newaxis] + lambda_[:, np.newaxis] * np.exp(min_half_v_sq_sigma_2 - min_half_v_sq_sigma_1)))
    return p

def mod_gaussian_log_likelihood_vec(params, data, single_gauss = False): #full with prior
    lp = log_prior_vec(params) # vectorized prior
    prior_mask = np.isfinite(lp)

    # if single_gauss: params[-1] = 0 #TODO: fix? vectorize somehow? easy enough to do with slicing like 2::3

    #TODO: globalize for reduced operations? 
    prior_data = data[prior_mask]

    # TODO: these can be taken outside of this function as well probably when implementing in MCMC
    # unnecessary to calculate these every time in the loop below (i.e. every time mod_gaussian_vec is called) since the parameter values stay the same
    sigma1, sigma2, lambda_ = params[:, prior_mask]
    norm = 1 / (((1- lambda_) * sigma1 + lambda_ * sigma2)* np.sqrt(2 * np.pi)) # independent of v, so move here
    sigma1_sq_inv = 1 / (2 * sigma1**2)
    sigma2_sq_inv = 1 / (2 * sigma2**2)
    one_min_lambda = 1 - lambda_

    mu = log_mod_gaussian_vec(prior_data, sigma1_sq_inv, sigma2_sq_inv, lambda_, one_min_lambda)
    L = mu + np.sum(np.log(norm))
    
    return L



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


### FOR PERTURBING AROUND THE OPTIMUM FROM MCMC
def sample_for_brent(params, found_likelihood, log_likelihood_func, single_gauss = False, threshold = 1.01):
    if single_gauss:
        step_sizes = [100.]
    else:
        step_sizes = [100., 100., 0.1] # high valued steps relative to global prior
    Lthresh = threshold * found_likelihood # desired likelihood decrease, 1% by default

    bounds = [[] for _ in range(len(step_sizes))]

    for param_idx,step in enumerate(step_sizes):
        param_insert = params.copy()
        current_param_value = params[param_idx]

        # step in the positive direction
        prior_flag = False
        Lcurrent = found_likelihood

        while Lcurrent < Lthresh:
            current_param_value += step
            if current_param_value >= GLOBAL_PRIOR_RANGE[param_idx][1]:
                prior_flag = True
                break

            #inplace operation on param_insert to change the relevant parameter
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
                prior_flag = True
                break

            #inplace operation on param_insert to change the relevant parameter
            np.put(param_insert, param_idx, current_param_value)
            Lcurrent = log_likelihood_func(param_insert)

        param_down = current_param_value if not prior_flag else GLOBAL_PRIOR_RANGE[param_idx][0]

        bounds[param_idx] = [param_down, param_up]
    
    return bounds

def _perturb_params(params: np.array, log_likelihood_func: Callable[[np.array], float], single_gauss: bool = False, threshold: float = 1.01):
    """ Perturb parameter value, either upwards or downwards, and return parameter value when likelihood threshold is reached.

    Args:
        params (list or array): list or array containing the parameter values *in order* [sigma1, sigma2, lambda]
        L_thresh (float): Threshold likelihood to reach.
        log_likelihood_func (function, optional): Log likelihood of the model. Defaults to mod_gaussian_log_likelihood. NOTE that this is an L-maximizing function since it is negative numbers we want to be as near to 0 as possible!
        single_gaussian (bool, optional): Controls whether we are fitting a single gaussian model (True) or a Double Gaussian (False). Defaults to False.
        threshold (float, optional): How much we want to be above the best found likelihood (relative). Defaults to 1.01 (1%). 
    
    Returns:
        param_bounds (list): nested list of 2-lists holding found parameter bounds at which the likelihood meets the set threshold (lower, upper)
        opt_params (np.ndarray): array holding results of scipy.optimize.minimize to find the local optimum near the input values. 
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
    brent_bounds = sample_for_brent(opt_params, -1*L_best, ll_func_for_minimize, single_gauss = single_gauss, threshold=threshold)

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

def perturb_around_likelihood(params: np.array, log_likelihood_func: Callable[[np.array], float], single_gauss: bool = False, threshold: float = 1.01) -> dict:
    """ Given a maximum from MCMC, perturb around this found maximum in small steps to estimate the error and refine the estimate.
        Uses the _perturb_params() functionality as a workhorse, this is moreso a wrapper around that function to format it to a dictionary

    Args:
        params (np.array): Best found parameter set
        log_likelihood_func (function, optional): Log likelihood of the model. Defaults to mod_gaussian_log_likelihood.
        single_gauss (bool, optional): Controls whether we "turn off" the lambda parameter, i.e. fix it to 0.
        threshold (float, optional): How much we want to be above the best found likelihood (relative). Defaults to 1.01 (1%). 

    Returns:
        param_dict: dictionary of parameter values and error estimates
    """
    
    #TODO: fix for joint model, there will be more parameters and this is hardcoded to 3 parameters now
    # ^that is gonna suck so fucking much

    bounds, best_params = _perturb_params(params, log_likelihood_func, single_gauss = single_gauss, threshold = threshold)
    param_dict = {'sigma_1':best_params[0], 'sigma_2':best_params[1], 'lambda': best_params[2], 'errors':[[] for _ in range(len(best_params))]}

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
    #n : no data points
    #k: no parameters
    return (k * np.log(n)) - 2 * L

def mkdir_if_non_existent(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def flatten(xss):
    # Flatten a python list using list comprehension
    return [x for xs in xss for x in xs]

def rice_bins(N):
    return 2 * int(np.cbrt(N))

