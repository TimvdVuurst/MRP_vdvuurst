import numpy as np
from scipy.special import gammainc
from typing import Callable

GLOBAL_PRIOR_RANGE = [[1., 1500.], [1., 1500.], [0., 0.5]]


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
    res = np.zeros(num)
    stop = np.log10(stop)
    if start == 0:
        start = np.log10(0.1)
        y = np.power(base, np.linspace(start, stop, num - 1))
        res[1:] = y
    else:
        start = np.log10(start)
        res = np.power(base, np.linspace(start, stop, num))

    return res

## GENERAL FUNCTIONS FOR ONEHALO

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

def mod_gaussian_log_likelihood(params, data, loglambda, fix_lambda): #full with prior
    prior = log_prior_loglambda if loglambda else log_prior
    lp = prior(params)
    if not np.isfinite(lp):
        return -np.inf

    if fix_lambda: params[-1] = 0

    mu_func = mod_gaussian_loglambda if loglambda else mod_gaussian
    mu_i = mu_func(data, *params)
    mu_i[mu_i < 0] = 0. # cast negative values to 0, raises errors otherwise
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

def rice_bins(N):
    return 2 * int(np.cbrt(N))


def perturb_around_likelihood(L: float, params: np.array, log_likelihood_func: Callable[[np.array], float], loglambda: bool = False, fix_lambda = False) -> dict:
    """_summary_

    Args:
        L (float): Likelihood value at the best found parameter set
        params (np.array): Best found parameter set
        log_likelihood_func (function, optional): Log likelihood of the model. Defaults to mod_gaussian_log_likelihood.

    Returns:
        dict: dictionary of parameter values and error estimates
    """
    
    #TODO: fix for joint model, there will be more parameters and this is hardcoded to 3 parameters now
    # param_names = ['sigma_1', 'sigma_2', 'lambda']
    if not loglambda:
        param_steps = [1., 1., 0.1]  #NOTE: subject to change
    else:
        param_steps = [1., 1., 1.]

    L_thresh = 1.01 * L #1%
    if not fix_lambda:
        param_dict = {'sigma_1':params[0], 'sigma_2':params[1], 'lambda': params[2], 'errors':[[] for i in range(len(params))]}
        for i in range(len(params)):
            param_up = _perturb_param(params, i, param_steps[i], L, L_thresh, dir = 1, log_likelihood_func = log_likelihood_func)
            param_down = _perturb_param(params, i, param_steps[i], L, L_thresh, dir = -1, log_likelihood_func = log_likelihood_func)

            param_dict['errors'][i] = [np.abs(params[i] - param_down), np.abs(params[i] - param_up)] # errors are the parameter differences

        return param_dict

    else:
        param_dict = {'sigma_1':params[0], 'sigma_2':params[1], 'lambda': 0., 'errors':[[] for i in range(len(params))]}

        param_up = _perturb_param(params, 0, param_steps[0], L, L_thresh, dir = 1, log_likelihood_func = log_likelihood_func)
        param_down = _perturb_param(params, 0, param_steps[0], L, L_thresh, dir = -1, log_likelihood_func = log_likelihood_func)

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

    prior_ranges = GLOBAL_PRIOR_RANGE.copy() #TODO?: not suited for loglambda now

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

        # # It might be that we step further into a minimum, in that case update the likelihood and parameter values accordingly
        # if L_current < L_best:
        #     L_best = L_current
        #     params = np.copy(perturbed_params)
        
        if not np.isfinite(L_current):
            # Assume the previous step was the best we could get within prior range
            # so return the relevant prior bound as the value
            prior_idx = 0 if dir == -1 else 1
            return prior_ranges[param_idx][prior_idx]

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
    param_val = (L_thresh - b) / a

    # print(f'{y = }, {x = }, {a = }, {b =}, {param_val = }\n')

    return param_val




