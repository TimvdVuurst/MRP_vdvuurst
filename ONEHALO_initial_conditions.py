import numpy as np
from scipy.optimize import minimize
from ONEHALO import ONEHALO_joint_fitter
from multiprocessing import Pool
from functions import mkdir_if_non_existent
import os
from tqdm import tqdm

joint_fitter = ONEHALO_joint_fitter() # initialize fitter class - this loads in the relevant data and defines functions that need that data to work 

# dictionary that, given a function in r, knows what to set the parameters to for a decent starting position
r_function_dict = {'poly_4':[0,0,0,0,150], 'poly_3':[0,0,0,150], 
                    'parabola':[0,0,50], 'linear':[0,50], 'exponential':[0,0,0.1], 'inverse':[0,0.4]}
# Amount of parameters each function type takes
m_func_n_params_dict = {'linear': 2, 'parabola': 3, 'exponential': 3}

DATA_SAVE_PATH = '/disks/cosmodm/vdvuurst/data/onehalo_joint_initial_conditions'

def _init_conditions(combi_names):
    initial_conditions = []
    for rfunc, mfuncs in combi_names:
        for rparam_value, mfunc in zip(r_function_dict[rfunc], mfuncs):
            if rparam_value == 0:
                mfunc_values = [0 for _ in range(m_func_n_params_dict[mfunc])]
            else: 
                mfunc_values = [0 for _ in range(m_func_n_params_dict[mfunc] - 1)]
                mfunc_values.append(rparam_value) # last value (constant term) is equal with all other parameters set to 0

            initial_conditions.extend(mfunc_values)
    
    return initial_conditions

def _find_starting_point(function_combi, combi_names):
    # Get very simple initial conditions (mostly zeroes) and necessary infotmation from function combination
    initial_conditions = _init_conditions(combi_names)    
    n_params_r, n_params_m, ntot = joint_fitter.param_info(function_combi)
    
    # Call from the joint_fitter object since it has data loaded in which is deeded to get the DG param values
    # NOTE: not a max number of steps given since the scipy source works with a while structure, but it runs quick enough (1hr)
    res = minimize(joint_fitter.get_joint_likelihood, initial_conditions, 
                   args = (n_params_m, n_params_r, function_combi),
                   method = 'BFGS') 
    
    return initial_conditions, res.x

def create_and_store_initial_conditions(inpt):
    function_combi, combi_names, combi_number = inpt # unpack input

    simple_initials, found_starting_point = _find_starting_point(function_combi, combi_names)

    scale_difference = np.abs(simple_initials - found_starting_point)

    # From the scale difference in the parameters, set the steps MCMC will take for this function combination
    min_step = 1e-4
    sigmas_MCMC = scale_difference * 0.1 # TODO: set a minimum? e.g. 1e-3 or 1e-4 note, to set a minimum use np.maximum ;)
    sigmas_MCMC[sigmas_MCMC == 0] = min_step 

    filename = os.path.join(DATA_SAVE_PATH, f'function_combi_{combi_number}')

    np.save(filename, np.vstack((found_starting_point, sigmas_MCMC))) # binary format built-in to numpy, quite efficient


if __name__ == '__main__':
    # Reduce terminal output
    from warnings import filterwarnings
    filterwarnings('ignore', category = RuntimeWarning)

    mkdir_if_non_existent(DATA_SAVE_PATH)

    from functional_forms import all_combis, all_names, combi_numbers

    iterable_input = list(zip(all_combis, all_names, combi_numbers))

    NPROCS = 20 

    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(create_and_store_initial_conditions, iterable_input):
            pbar.update()



    