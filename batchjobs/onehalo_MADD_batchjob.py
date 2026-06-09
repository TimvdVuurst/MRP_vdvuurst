import sys
if '/disks/cosmodm/vdvuurst' not in sys.path:
    sys.path.append('/disks/cosmodm/vdvuurst')

import numpy as np
from multiprocessing import Pool
from ONEHALO import ONEHALO_MADD_fitter, param_info
from functional_forms import *
from functions import mkdir_if_non_existent
from tqdm.auto import tqdm
from argparse import ArgumentParser
import os
from glob import glob
from onehalo_plotter import format_plot

format_plot()

from warnings import filterwarnings
# this might be a liiiitle dangerous but cleans up the output by a lot. that's because we often see invalid values in log10, but that's ok
filterwarnings('ignore',category = RuntimeWarning)


parser = ArgumentParser()
parser.add_argument('-SD', '--subsampled_data', type = int, default = 1, help = 'Whether to use subsampled data or not. Defaults to 1 (True).')
parser.add_argument('-SF', '--subsampled_funcs', type = int, default = 1, help = 'Whether to use subsampled functions or not. Defaults to 1 (True).')
parser.add_argument('-NC', '--ncores', type = int, default = 20, help = 'Whether to use subsampled data or not. Defaults to 1 (True).')
parser.add_argument('-V', '--verbose', type = int, default = 0, help = 'Whether to print diagnostics and timings. Defaults to 0 (False).')
parser.add_argument('-P', '--plot', type = int, default = 0, help = 'Whether to plot stuff. Defaults to 0 (False).')
parser.add_argument('-RM', '--remove_previous', type = int, default = 0, help = 'Whether to remove all previous run data. Defaults to 0 (False).')
parser.add_argument('-IM', '--init_method', type = str, default = 'Nelder-Mead', help = 'Which initial conditions to use. Defaults to Nelder-Mead.')
parser.add_argument('-FF', '--functional_form', type = str, default = '', help = 'Additional information to separate results. Defaults to empty string.')
parser.add_argument('-NS', '--num_steps', type = int, default = 500, help = 'How many steps to take in the MCMC process. Defaults to 1000.')
parser.add_argument('-SB', '--skip_bad', type = int, default = 1, help = 'Whether to skip the fitting of function combination that were found to have bad initial conditions. Defaults to 1 (True).')
parser.add_argument('-O', '--overwrite', type = int, default = 0, help = 'Whether to overwrite existing fits. Defaults to 0 (False).')

args = parser.parse_args()

# Open log file and clear it before run
with open('./logs/MADD_log.txt', 'r+') as f:
    f.seek(0)
    f.truncate()

# pre-setting some controlling variables
use_subsampled_data = bool(args.subsampled_data)
use_subsampled_funcs = bool(args.subsampled_funcs)
verbose = bool(args.verbose)
skip_bad_combis = bool(args.skip_bad)

if args.functional_form != '':
    functional_form = args.functional_form if args.functional_form.startswith('_') else '_' + args.functional_form
else:
    functional_form = args.functional_form
fp = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/MADD{'_subsample' if use_subsampled_data else ''}/{args.init_method}{functional_form}/'
mkdir_if_non_existent(fp)

function_combis = combi_subsample if use_subsampled_funcs else all_combis
function_names = combi_subsample_names if use_subsampled_funcs else all_names
function_numbers = combi_subsamples_numbers if use_subsampled_funcs else combi_numbers

if skip_bad_combis: 
    if functional_form != '':
        badpath = f'./logs/bad_initial_conditions{'_' if not use_subsampled_funcs else '_subsampled_'}{functional_form}_combi_cnrs.npy'
    else:
        badpath = f'./logs/bad_initial_conditions{'_' if not use_subsampled_funcs else '_subsampled_'}{functional_form}combi_cnrs.npy'

    bad_cnrs = np.load(badpath)
    cnrs_to_check = combi_numbers if not use_subsampled_funcs else combi_subsamples_numbers
    good_mask = np.invert(np.isin(cnrs_to_check, bad_cnrs))

    function_combis = function_combis[good_mask]
    function_names = function_names[good_mask]
    function_numbers = function_numbers[good_mask]


DATAPATH = '/disks/cosmodm/vdvuurst/data'
subsample_str = '_subsampled' if use_subsampled_data else ''
datapath = os.path.join(DATAPATH, f'Onehalo_M_12-15.5{subsample_str}.hdf5')

# loads in subsampled data by default
MADD_fitter = ONEHALO_MADD_fitter(PATH = datapath) 

def _create_iterable_input():
    # For all combis get metainfo, need this to set the number of walkers
    n_params_r, n_params_m, ntot = list(zip(*[param_info(function_combi) for function_combi in function_combis]))

    nsteps = [args.num_steps for _ in range(len(ntot))] # fill with nsteps for every combination
    nwalkers = [int(np.ceil(ntot_i * 2.5)) for ntot_i in ntot] #factor as many walkers as we have parameters

    return list(zip(function_combis, function_names, function_numbers, nwalkers, nsteps, n_params_r, n_params_m, ntot)) #TODO: edit filepath arg for when non-subsampled


def run_experiment(inpt):
    MADD_fitter.fit_function_combi_to_data(*inpt, verbose = verbose, 
                                            plot = bool(args.plot), filepath = fp,
                                            init_condition_method = args.init_method,
                                            functional_form=functional_form,
                                            overwrite = bool(args.overwrite))

if __name__ == '__main__':
    NPROCS = args.ncores
    iterable_input = _create_iterable_input()

    if bool(args.remove_previous):
        old_files = glob(f'{fp}*.json')
        print('Removing old fits...')
        for f in tqdm(old_files):
            os.remove(f)

    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(run_experiment, iterable_input):
            pbar.update()






