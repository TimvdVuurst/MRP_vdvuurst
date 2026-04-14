import sys
if '/disks/cosmodm/vdvuurst' not in sys.path:
    sys.path.append('/disks/cosmodm/vdvuurst')

import numpy as np
from multiprocessing import Pool
from ONEHALO import ONEHALO_joint_fitter, param_info
from functional_forms import *
# from functions import * 
from tqdm import tqdm
from argparse import ArgumentParser
import os

from warnings import filterwarnings
# this might be a liiiitle dangerous but cleans up the output by a lot. that's because we often see invalid values in log10, but that's ok
filterwarnings('ignore',category = RuntimeWarning)


parser = ArgumentParser()
parser.add_argument('-S', '--subsampled', type = int, default = 1, help = 'Whether to use subsampled data or not. Defautls to 1 (True)')
parser.add_argument('-NC', '--ncores', type = int, default = 20, help = 'Whether to use subsampled data or not. Defautls to 1 (True)')
parser.add_argument('-V', '--verbose', type = int, default = 0, help = 'Whether to print diagnostics and timings. 1 for True, 0 for False.')
args = parser.parse_args()

use_subsampled = bool(args.subsampled)
verbose = bool(args.verbose)

function_combis = combi_subsample if use_subsampled else all_combis
function_names = combi_subsample_names if use_subsampled else all_names
function_numbers = combi_subsamples_numbers if use_subsampled else combi_numbers

joint_fitter = ONEHALO_joint_fitter() # loads in subsampled data by default TODO: make 

def _create_iterable_input():
    # For all combis get metainfo, need this to set the number of walkers
    n_params_r, n_params_m, ntot = list(zip(*[param_info(function_combi) for function_combi in function_combis]))

    nsteps = [1000 for _ in range(len(ntot))] # fill with 1000 for every combination
    nwalkers = [int(np.ceil(ntot_i * 1.5)) for ntot_i in ntot]

    return list(zip(function_combis, function_names, function_numbers, nwalkers, nsteps, n_params_r, n_params_m, ntot)) #TODO: edit filepath arg for when non-subsampled


def run_experiment(inpt):
    joint_fitter.fit_function_combi_to_data(*inpt, verbose = verbose, plot = False)

if __name__ == '__main__':
    NPROCS = args.ncores
    iterable_input = _create_iterable_input()

    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(run_experiment, iterable_input):
            pbar.update()






