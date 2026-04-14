import numpy as np
from itertools import product, combinations_with_replacement
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

lambda_r_funcs = [exponential_func,inverse_func]
sigma_1_r_funcs = [poly_3_func, poly_4_func]
sigma_2_r_funcs = [linear_func, parabola_func]

m_funcs = [linear_func, parabola_func, exponential_func]


no_params = lambda f: len(signature(f).parameters) - 1
get_func_name = lambda f: f.__name__.split('_func')[0]

m_func_names = [get_func_name(m) for m in m_funcs]
# lambda_r_func_names = [get_func_name(r) for r in lambda_r_funcs]
# sigma_1_func_names = [get_func_name(r) for r in sigma_1_r_funcs]
# sigma_2_func_names = [get_func_name(r) for r in sigma_2_r_funcs]

def flatten(xss):
    return [x for xs in xss for x in xs]

def create_function_combinations(funclist): 
    func_combis = [list(product([rfunc], list(combinations_with_replacement(m_funcs, no_params(rfunc))))) for rfunc in funclist]
    return flatten(func_combis)

def create_function_combination_namelist(funclist):
    name_combis = [list(product([get_func_name(rfunc)], list(combinations_with_replacement(m_func_names, no_params(rfunc))))) for rfunc in funclist]
    return flatten(name_combis)

lambda_funcs = create_function_combinations(lambda_r_funcs)
sigma_1_funcs = create_function_combinations(sigma_1_r_funcs)
sigma_2_funcs = create_function_combinations(sigma_2_r_funcs)

lambda_funcs_names = create_function_combination_namelist(lambda_r_funcs)
sigma_1_funcs_names = create_function_combination_namelist(sigma_1_r_funcs)
sigma_2_funcs_names = create_function_combination_namelist(sigma_2_r_funcs)

all_combis = np.array(list(product(sigma_1_funcs, sigma_2_funcs, lambda_funcs)), dtype = object)
all_names = np.array(list(product(sigma_1_funcs_names, sigma_2_funcs_names, lambda_funcs_names)), dtype = object)
combi_numbers = np.arange(len(all_names)) + 1

# Subsampling
np.random.seed(42)
combi_subsample_idx = np.random.choice(len(all_combis),size = len(all_combis)//100) #1% subsample 
all_combis = np.array(all_combis, dtype=object)
combi_subsample = all_combis[combi_subsample_idx]
combi_subsample_names = np.array(all_names, dtype = object)[combi_subsample_idx]
combi_subsamples_numbers = combi_numbers[combi_subsample_idx]

# print(f'There are {combi_subsample.shape[0]} function combinations')
# print(f'There are {len(all_combis)} function combinations')

