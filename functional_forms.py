import numpy as np
from itertools import product
from inspect import signature

def constant_func(x,a):
    return np.full_like(x, a)

def linear_func(x, m, c):
    return m * x + c

def parabola_func(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_func(x, A, B, C):
    return A * np.exp(-B * x) + C

def exponential_squared_func(x, A, B, C):
    return A * np.exp(-B * x**2) + C

def power_law_func(x, p, n, q):
    return p * np.power(x, n) + q

def power_linear_func(x, p, n, q, b):
    return p * np.power(x, n) + q * x + b

def inverse_func(x, a, b):
    return a/x + b

def poly_3_func(x, a, b, c, d):
    return a * np.power(x,3) + b*np.square(x) + c*x + d

def poly_4_func(x, a, b, c, d, e):
    return a*np.power(x,4) + b*np.power(x, 3) + c* np.square(x) + d*x + e

# STANDARD FUNCTIONS OF R1h!
lambda_r_funcs = [inverse_func]
sigma_1_r_funcs = [poly_3_func]
sigma_2_r_funcs = [linear_func]

m_funcs = [linear_func, parabola_func, exponential_func]

no_params = lambda f: len(signature(f).parameters) - 1
get_func_name = lambda f: f.__name__.split('_func')[0]

def flatten(xss):
    return [x for xs in xss for x in xs]

def create_function_combinations(funclist, m_funcs = m_funcs): 
    func_combis = [list(product([rfunc], list(product(m_funcs, repeat = no_params(funclist[0]))))) for rfunc in funclist]
    return flatten(func_combis)

def create_function_combination_namelist(funclist, m_func_names):
    name_combis = [list(product([get_func_name(rfunc)], list(product(m_func_names, repeat = no_params(rfunc))))) for rfunc in funclist]
    return flatten(name_combis)

def get_function_combinations(sigma_1_r_funcs = sigma_1_r_funcs, sigma_2_r_funcs = sigma_2_r_funcs, 
                              lambda_r_funcs = lambda_r_funcs, m_funcs = m_funcs, create_subsample = True):
    
    m_func_names = [get_func_name(m) for m in m_funcs]
    
    lambda_funcs = create_function_combinations(lambda_r_funcs)
    sigma_1_funcs = create_function_combinations(sigma_1_r_funcs)
    sigma_2_funcs = create_function_combinations(sigma_2_r_funcs)

    lambda_funcs_names = create_function_combination_namelist(lambda_r_funcs, m_func_names)
    sigma_1_funcs_names = create_function_combination_namelist(sigma_1_r_funcs, m_func_names)
    sigma_2_funcs_names = create_function_combination_namelist(sigma_2_r_funcs, m_func_names)

    all_combis = np.array(list(product(sigma_1_funcs, sigma_2_funcs, lambda_funcs)), dtype = object)
    all_names = np.array(list(product(sigma_1_funcs_names, sigma_2_funcs_names, lambda_funcs_names)), dtype = object)
    combi_numbers = np.arange(len(all_names)) + 1

    if create_subsample:
        np.random.seed(42) # For consistency
        combi_subsample_idx = np.random.choice(len(all_combis),size = len(all_combis)//5, replace = False) #20% subsample  
        all_combis = np.array(all_combis, dtype=object)
        combi_subsample = all_combis[combi_subsample_idx]
        combi_subsample_names = np.array(all_names, dtype = object)[combi_subsample_idx]
        combi_subsamples_numbers = combi_numbers[combi_subsample_idx]

        return all_combis, all_names, combi_numbers, combi_subsample, combi_subsample_names, combi_subsamples_numbers

    return all_combis, all_names, combi_numbers

all_combis, all_names, combi_numbers, combi_subsample, combi_subsample_names, combi_subsamples_numbers = get_function_combinations()


subsample_num_to_idx = dict(zip(combi_subsamples_numbers, np.arange(combi_subsamples_numbers.size)))

if __name__ == '__main__':
    num_to_func_dict = dict(zip([int(c) for c in combi_numbers],
                                    [[list(l) for l in fnames] for fnames in all_names]))

    from json import dump

    with open(f'/disks/cosmodm/vdvuurst/data/func_nums_to_names.json', 'w') as f:
        dump(num_to_func_dict, f, indent = 2)

    print(f'There are {combi_subsample.shape[0]} subsampled function combinations')
    print(f'There are {len(all_combis)} function combinations')
    
    from ONEHALO import param_info
    no_params_all = [param_info(c)[2] for c in all_combis]
    print(f'At least {np.min(no_params_all)} params and at most {np.max(no_params_all)}')
