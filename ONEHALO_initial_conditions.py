import numpy as np
from scipy.optimize import minimize
from ONEHALO import ONEHALO_MADD_fitter, param_info
from multiprocessing import Pool
from functions import mkdir_if_non_existent, flatten, GLOBAL_PRIOR_RANGE
import os
from tqdm import tqdm

MADD_fitter = ONEHALO_MADD_fitter()

# dictionary that, given a function in r, knows what to set the parameters to for a decent starting position
r_function_dict = {'poly_4':[0,0,0,0,500], 'poly_3':[0,0,0,500], 
                    'parabola':[0,0,50], 'linear':[0, 50], 'exponential':[0,0,0.25], 'inverse':[0,0.25]} 

# Amount of parameters each function type takes
m_func_n_params_dict = {'linear': 2, 'parabola': 3, 'exponential': 3}

# Safe because this file is never imported
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-M', '--method', type = str, default = 'Nelder-Mead', help = 'Routine passed to scipy.minimize. Defaults to Nelder-Mead.')
parser.add_argument('-SB', '--set_bounds', type = int, default = 1, help = 'Set bounds on the minimization (lambda subparameters >0, sigma_1 1000 km/s and sigma_2 < sigma_1 clamped). Defaults to 0 (False).')
parser.add_argument('-NC', '--ncores', type = int, default = 30, help = 'Number of cores to run on in parallel. Defaults to 30.')
parser.add_argument('-F', '--full_set', type = int, default = 1, help = 'Whether to use full set of function combinations or subsampled. Defaults to 0 (False), i.e. subsampled.')
parser.add_argument('-RR', '--rerun', type = int, default = 0, help = 'Whether to rerun all wanted combinations or skip those that already exist. Defaults to 0 (False), i.e. those that already exist will not be overwritten.')
parser.add_argument('-D', '--delete_existing', type = int, default = 0, help = 'Whether to delete all initial conditions that have been ran for the specified method. Defaults to 0 (False).')
parser.add_argument('-FF', '--functional_form', type = str, default = '', help = 'Additional information to separate results. Defaults to empty string.')
args = parser.parse_args()
set_bounds = bool(args.set_bounds)

if args.functional_form != '':
    functional_form = args.functional_form if args.functional_form.startswith('_') else '_' + args.functional_form
else:
    functional_form = args.functional_form

DATA_SAVE_PATH = f'/disks/cosmodm/vdvuurst/data/onehalo_MADD_initial_conditions/{args.method}{functional_form}'
mkdir_if_non_existent(DATA_SAVE_PATH)

def _init_conditions(combi_names: list):
    """ Generate simplistic initial conditions based on values specified in the variables r_function_dict. 

    Args:
        combi_names (list): Array-like of structured function names of desired funtction combination. Should be as generated in the functional_forms.py script and of only 1 function combination.

    Returns:
        list: simple initial conditions for the given function combination.
    """
    initial_conditions = []
    for rfunc, mfuncs in combi_names:
        for rparam_value, mfunc in zip(r_function_dict[rfunc], mfuncs):
            if rparam_value == 0:
                mfunc_values = [0 for _ in range(m_func_n_params_dict[mfunc])]
            else: 
                mfunc_values = [0 for _ in range(m_func_n_params_dict[mfunc] - 1)]
                mfunc_values.append(rparam_value) 

            initial_conditions.extend(mfunc_values)
    
    return initial_conditions

def _find_starting_point(function_combi: list, combi_names: list, method: str = args.method, set_bounds: bool = set_bounds, constrain_sigmas: bool = False):
    """ Starting from a simplified starting point as calculated by _init_conditions(), perform a minimization routine
        via scipy.optimize.minimize (with a given method, see scipy documentation for details) to find a good initial condition
        for the MCMC routine. If desired, lambda subparameters can be set to >0 and the sigma_1 and sigma_2 paramters capped at 1000 
        (this stabilizes the process). 

    Args:
        function_combi (list): list of function instances to be passed to the likelihood function. Should be as generated in functional_forms.py.
        combi_names (list):  Array-like of structured function names of desired funtction combination. Should be as generated in the functional_forms.py script and of only 1 function combination.
        method (string, optional): Method name to be passed to scipy.optimize.minimize. Defaults to Nelder-Mead and is controlled via the command-line.
        set_bounds (bool, optional): Whether to set lambda subparameters >0 and sigma_1 and sigma_2 upper bound to 1000. Defaults to False and is controlled via the command-line.

    Returns:
        np.ndarray: The simplified initial conditions from the _init_conditions method.
        np.ndarray: The result from the minimize routine
    """

    initial_conditions = _init_conditions(combi_names)
    if method == 'simple':
        return np.array(initial_conditions), np.array(initial_conditions)

    n_params_r, n_params_m, ntot = param_info(function_combi)
    
    if set_bounds: 
        bnds_sigmas = [(None, None) for _ in range(ntot - sum(n_params_m[-n_params_r[-1]:]))]
        bnds_lam = flatten([[(0,None) for _ in range(n)] for n in n_params_m[-n_params_r[-1]:]])
        bnds = tuple(bnds_sigmas + bnds_lam)
         
        res = minimize(MADD_fitter.get_MADD_likelihood, initial_conditions, 
                    args = (n_params_m, n_params_r, function_combi, 1000., constrain_sigmas),
                    method = method, bounds = bnds) 
    
    else:      
        res = minimize(MADD_fitter.get_MADD_likelihood, initial_conditions, 
                    args = (n_params_m, n_params_r, function_combi),
                    method = method) 
        
    return np.array(initial_conditions), res.x

def create_and_store_initial_conditions(inpt: tuple):
    """ Functionality to be iterated over by multiprocess. Using the _find_starting_point() method, can for a given function combination find a good starting point
        and estimate the MCMC stepsize as 10% of the change in each parameter.

    Args:
        inpt (tuple): Tuple of: function_combi (list), combi_names (list) and combi_number (int)

    Returns:
        None. Results are saved to DATA_SAVE_PATH as a unique .npy file based on the passed combination number.
    """
    function_combi, combi_names, combi_number = inpt

    simple_initials, found_starting_point = _find_starting_point(function_combi, combi_names, constrain_sigmas = False)

    scale_difference = np.abs(simple_initials - found_starting_point)

    min_step = 1e-2
    sigmas_MCMC = scale_difference * 0.1
    sigmas_MCMC[sigmas_MCMC == 0] = min_step 

    filename = os.path.join(DATA_SAVE_PATH, f'function_combi_{combi_number}')

    np.save(filename, np.vstack((found_starting_point, sigmas_MCMC)))

def create_and_store_initial_conditions_constrain_sigmas(inpt: tuple):
    """ 
    The same as above but calls _find_starting_point with constrain_sigmas set to True.
    """
    function_combi, combi_names, combi_number = inpt

    simple_initials, found_starting_point = _find_starting_point(function_combi, combi_names, constrain_sigmas = True)

    scale_difference = np.abs(simple_initials - found_starting_point)

    min_step = 1e-2
    sigmas_MCMC = scale_difference * 0.1
    sigmas_MCMC[sigmas_MCMC == 0] = min_step 

    filename = os.path.join(DATA_SAVE_PATH, f'function_combi_{combi_number}')

    np.save(filename, np.vstack((found_starting_point, sigmas_MCMC)))


if __name__ == '__main__':
    # Reduce terminal output
    from warnings import filterwarnings
    filterwarnings('ignore', category = RuntimeWarning)

    mkdir_if_non_existent(DATA_SAVE_PATH)

    from functional_forms import linear_func, parabola_func, poly_3_func, poly_4_func, exponential_func, inverse_func, get_function_combinations

    if args.functional_form != '':
        name_to_func_dict = {'linear':linear_func, 'parabola':parabola_func, 'poly2':parabola_func, 'poly3':poly_3_func,
                            'poly4':poly_4_func, 'inverse':inverse_func, 'inv':inverse_func, 'exp':exponential_func, 'exponential':exponential_func}
        
        param_to_change, func_form = args.functional_form.split('_')

        match param_to_change:
            case 'sigma1':
                all_combis, all_names, combi_numbers, combi_subsample, combi_subsample_names, combi_subsamples_numbers = get_function_combinations(sigma_1_r_funcs=[name_to_func_dict[func_form]])
            case 'sigma2':
                all_combis, all_names, combi_numbers, combi_subsample, combi_subsample_names, combi_subsamples_numbers = get_function_combinations(sigma_2_r_funcs=[name_to_func_dict[func_form]])
            case 'lambda':
                all_combis, all_names, combi_numbers, combi_subsample, combi_subsample_names, combi_subsamples_numbers = get_function_combinations(lambda_r_funcs=[name_to_func_dict[func_form]])
    else:
        # Base functional forms
        all_combis, all_names, combi_numbers, combi_subsample, combi_subsample_names, combi_subsamples_numbers = get_function_combinations()


    ## To delete the old combinations 
    if bool(args.delete_existing):
        fp = f'/disks/cosmodm/vdvuurst/data/onehalo_MADD_initial_conditions/{args.method}{functional_form}'
        print('Deleting existing initial conditions...')
        for i in tqdm(os.listdir(fp)):
            filename = os.path.join(fp, i)
            if os.path.isfile(filename): 
                os.remove(filename)

    ## Running specified initial condition method, only for the desired combinations

    use_full_set = bool(args.full_set)

    if use_full_set:
        if bool(args.rerun):
            iterable_input = list(zip(all_combis, all_names, combi_numbers))
            tot = len(all_combis)
            skip_rerun = False
        else:
            already_existing_numbers = [int(f.split('_')[-1].strip('.npy')) for f in os.listdir(DATA_SAVE_PATH)]
            rerun_mask = np.invert(np.isin(combi_numbers, already_existing_numbers))
            iterable_input = list(zip(all_combis[rerun_mask], all_names[rerun_mask], combi_numbers[rerun_mask]))
            tot = rerun_mask.sum()
            if tot == 0:
                print('No initial conditions to find...')
                skip_rerun = True
            else:
                skip_rerun = False
    else:
        if bool(args.rerun):
            iterable_input = list(zip(combi_subsample, combi_subsample_names, combi_subsamples_numbers))
            tot = len(combi_subsample)
            skip_rerun = False
        else:
            already_existing_numbers = [int(f.split('_')[-1].strip('.npy')) for f in os.listdir(DATA_SAVE_PATH) if '.npy' in f]
            rerun_mask = np.invert(np.isin(combi_subsamples_numbers, already_existing_numbers))
            iterable_input = list(zip(combi_subsample[rerun_mask], combi_subsample_names[rerun_mask], combi_subsamples_numbers[rerun_mask]))
            tot = rerun_mask.sum()
            if tot == 0:
                print('No initial conditions to find...')
                skip_rerun = True
            else:
                skip_rerun = False

    # if not skip_rerun:
    #     NPROCS = args.ncores
    #     with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
    #         for _ in p.imap_unordered(create_and_store_initial_conditions, iterable_input):
    #             pbar.update()

    ## Verifying that the initial conditions are good
    # The above initial conditions were found by not flipping the sigma values and enforcing sigma2 < sigma1
    # This works for ~2/3 function combinations. For the rest, we do not enforce this but instead only clamp sigma1 and sigma2 to 1000
    # and flip the sigmas if need be. These have to be reran.

    from ONEHALO import ONEHALO_MADD_fitter, param_info
    MADD_test = ONEHALO_MADD_fitter()

    def get_init_DG_from_combi_nr(combi_nr, init_path):
        func_combi = all_combis[combi_nr - 1] if use_full_set else combi_subsample[np.isin(combi_subsamples_numbers, combi_nr)]
        n_params_r, n_params_m, ntot = param_info(func_combi)

        init_guess, _ = np.load(f'{init_path}/function_combi_{combi_nr}.npy')

        split_params = MADD_test.split_parameters(init_guess, n_params_m)
        DG = MADD_test.get_double_gauss_parameters(split_params, func_combi, n_params_r)
        return DG

    def output_init_conditions(method = args.method):
        init_path = f'/disks/cosmodm/vdvuurst/data/onehalo_MADD_initial_conditions/{method}{functional_form}'
        bad_cnrs = []
        iterable = enumerate(tqdm(combi_numbers)) if use_full_set else enumerate(tqdm(combi_subsamples_numbers))
        for i, cnr in iterable:
            already_added = False

            sigma_1_init, sigma_2_init, lambda_init = get_init_DG_from_combi_nr(cnr, init_path = init_path)
        
            if np.all(sigma_1_init == GLOBAL_PRIOR_RANGE[0][1]) or np.all(sigma_1_init == GLOBAL_PRIOR_RANGE[0][0]):
                if cnr not in bad_cnrs:
                    bad_cnrs.append(cnr)
                already_added = True
            
            elif np.all(sigma_2_init == GLOBAL_PRIOR_RANGE[1][1]) or np.all(sigma_2_init == GLOBAL_PRIOR_RANGE[1][0]):
                if not already_added and cnr not in bad_cnrs:
                    bad_cnrs.append(cnr)
                already_added =  True
            
            elif np.all(lambda_init == GLOBAL_PRIOR_RANGE[2][1]) or np.all(lambda_init == GLOBAL_PRIOR_RANGE[2][0]):
                if not already_added and cnr not in bad_cnrs:
                    bad_cnrs.append(cnr)
            

        return np.array(bad_cnrs)

    def verify_initials():
        print('Verifying initial conditions...')
        bad_cnrs = output_init_conditions(args.method).astype(int)
        cnrs_to_check = combi_numbers if use_full_set else combi_subsamples_numbers

        print(f'We find {bad_cnrs.shape[0]} bad combinations and {cnrs_to_check.size - bad_cnrs.shape[0]} good ones. ', end ='')

        return bad_cnrs

    bad_cnrs = verify_initials()
    if bad_cnrs.size != 0:

        if not skip_rerun:
            print('Rerunning bad ones with constraining sigmas...')
            if use_full_set:
                bad_cnrs_mask = np.isin(combi_numbers, bad_cnrs)
                iterable_input = list(zip(all_combis[bad_cnrs_mask], all_names[bad_cnrs_mask], combi_numbers[bad_cnrs_mask]))

            else:
                bad_cnrs_mask = np.isin(combi_subsamples_numbers, bad_cnrs)
                iterable_input = list(zip(combi_subsample[bad_cnrs_mask], combi_subsample_names[bad_cnrs_mask], combi_subsamples_numbers[bad_cnrs_mask]))

            # NPROCS = np.min((NPROCS, bad_cnrs.shape[0]))
            # with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
            #     for _ in p.imap_unordered(create_and_store_initial_conditions_constrain_sigmas, iterable_input):
            #         pbar.update()

            bad_cnrs = verify_initials()
            print('Leaving it at that.')
        
        print('\nSaving indices of bad combinations...')
        savepath = f'/disks/cosmodm/vdvuurst/logs/bad_initial_conditions{'_' if use_full_set else '_subsampled_'}{functional_form}_combi_cnrs'

        np.save(savepath, bad_cnrs)

        ## Checking existence of trends in the bad combinations
        
        # cnrs_to_check = combi_numbers if use_full_set else combi_subsamples_numbers
        # bad_mask = np.isin(cnrs_to_check, bad_cnrs)
        # bad_names = combi_subsample_names[bad_mask] if not use_full_set else all_names[bad_mask]

        # bad_sigma_2_r_funcs = [bn[1][0] for bn in bad_names]
        # bad_sigma_1_m1, bad_sigma_1_m2, bad_sigma_1_m3, bad_sigma_1_m4 = np.array([bn[0][1] for bn in bad_names]).T
        # bad_sigma_2_m1, bad_sigma_2_m2, bad_sigma_2_m3 = np.array([bn[1][1] for bn in bad_names]).T
        # bad_lambda_m1, bad_lambda_m2 = np.transpose([bn[2][1] for bn in bad_names])

        # def bad_counts(input, param):
        #     unique, counts = np.unique(input, return_counts = True)
        #     print(f'{param}: funcs are {unique} with respective bad counts {counts}')

        # bad_counts(bad_sigma_2_r_funcs, 'sigma_2_r')
        # print()
        # bad_counts(bad_sigma_1_m1, 'sigma_1_m1')
        # bad_counts(bad_sigma_1_m2, 'sigma_1_m2')
        # bad_counts(bad_sigma_1_m3, 'sigma_1_m3')
        # bad_counts(bad_sigma_1_m4, 'sigma_1_m4')
        # print()
        # bad_counts(bad_sigma_2_m1, 'sigma_2_m1')
        # bad_counts(bad_sigma_2_m2, 'sigma_2_m2')
        # bad_counts(bad_sigma_2_m3, 'sigma_2_m3')
        # print()
        # bad_counts(bad_lambda_m1, 'lambda_m1')
        # bad_counts(bad_lambda_m2, 'lambda_m2')