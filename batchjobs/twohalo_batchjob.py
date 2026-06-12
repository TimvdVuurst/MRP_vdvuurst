import sys
if '/disks/cosmodm/vdvuurst' not in sys.path:
    sys.path.append('/disks/cosmodm/vdvuurst')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from TWOHALO import TWOHALO_fitter
from tqdm import tqdm
from argparse import ArgumentParser
from functions import *
from multiprocessing import Pool

from warnings import filterwarnings
# this might be a liiiitle dangerous but cleans up the output by a lot. that's because we often see invalid values in log10 or smth, but that's ok
filterwarnings('ignore',category = RuntimeWarning)

def format_plot():
    # Define some properties for the figures so that they look good
    SMALL_SIZE = 10 * 2 
    MEDIUM_SIZE = 12 * 2
    BIGGER_SIZE = 14 * 2

    plt.rc('axes', titlesize=BIGGER_SIZE)                     # fontsize of the axes title\n",
    plt.rc('axes', labelsize=BIGGER_SIZE)                    # fontsize of the x and y labels\n",
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
    plt.rc('legend', fontsize=MEDIUM_SIZE)                    # legend fontsize\n",
    mpl.rcParams['axes.titlesize'] = BIGGER_SIZE
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'STIXgeneral'

    mpl.rcParams['figure.dpi'] = 300

    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True

    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['ytick.minor.size'] = 4

    mpl.rcParams['xtick.major.width'] = 1.25
    mpl.rcParams['ytick.major.width'] = 1.25
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.minor.width'] = 1

format_plot()

parser = ArgumentParser()
parser.add_argument('-F', '--function', type = str, default='skew-t', help = 'Distribution to fit. Either skew-normal or skew-t. Defaults to skew-t.')
parser.add_argument('-P', '--plot_only', type = int, default= 0, help = 'Whether to fit and plot (0) or plot only from existing results (1). Defaults to 0.')
args = parser.parse_args()

function_str = args.function.lower()
if function_str == 'skew-t':
    func = skew_t_pdf
    ll_func = skew_t_log_likelihood
elif function_str == 'skew-norm' or function_str == 'skew-normal':
    func = skewnorm_func
    ll_func = skew_gaussian_log_likelihood

#TODO: be able to change which mass bin you're pointing to
# and maybe make a bunch so that you can parallelize over mass bins instead of radial bins
twohalo_fitter = TWOHALO_fitter()

bindices = np.arange(2, 19)

if not bool(args.plot_only):
    nsteps = [500 if bidx < 7 else 200 for bidx in bindices]

    iterable_input = [(bidx, ll_func, func, 20, nstep) for (bidx, nstep) in zip(bindices, nsteps)]

    def _run_experiment(inpt):
        twohalo_fitter.run_two_halo_emcee(*inpt)

    NPROCS = len(bindices)
    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(_run_experiment, iterable_input):
            pbar.update()

else:
    base_path = f'/disks/cosmodm/vdvuurst/data/TwoHalo_param_fits/{function_str[0].upper()}{function_str[1:]}/M1_13.0-13.5_M2_13.5-14.0-rbin'

    iterable_input = [(base_path+f'{bidx}.json', bidx) for bidx in bindices]

    def _run_experiment(inpt):
        twohalo_fitter.plot_dist_from_result(*inpt)

    NPROCS = len(bindices)
    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(_run_experiment, iterable_input):
            pbar.update()

