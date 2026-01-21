import os
import numpy as np
from ONEHALO import ONEHALO, ONEHALO_fitter
from onehalo_plotter import format_plot
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functions import modified_logspace, rice_bins

import warnings

# def fxn():
#     # warnings.warn("RuntimeWarning: invalid value encountered in scalar subtract lnpdiff = f + nlp - state.log_prob[j]", RuntimeWarning)
#     warnings.warn(RuntimeWarning)

# with warnings.catch_warnings(action="ignore"):
#     fxn()

#TODO: this might be a liiiitle dangerous but cleans up the output by a lot. find out if we can ignore the warning specified in fxn() above specifically
warnings.filterwarnings('ignore',category = RuntimeWarning)

SOAP_PATH_DEFAULT = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5"

parser = argparse.ArgumentParser()
parser.add_argument('-M1','--lower_mass', type = np.float32, default = 2, help = 'Lower bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 2.')
parser.add_argument('-M2','--upper_mass', type = np.float32, default = 5.5, help = 'Upper bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 5.5.') #always EXCLUSIVE upper bound
parser.add_argument('-r1','--lower_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!')
parser.add_argument('-r2','--upper_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!') 
parser.add_argument('-S', '--step', type = np.float32, default = 0.5, help = 'Size of bins in dex. Defaults to 0.5.')
parser.add_argument('-O', '--overwrite', type = int, default = 1, help = 'If a catalogue already exist, control whether to overwrite it. 1 for True, 0 for False.')
# parser.add_argument('-B', '--bins', type = int, default = 500, help = 'Number of velocity bins')
parser.add_argument('-M', '--method', type = str, default = 'emcee', help = 'Fitting procedure. Choose either emcee, minimize or both.')
parser.add_argument('-V', '--verbose', type = int, default = 1, help = 'Whether to print diagnostics and timings. 1 for True, 0 for False.')
parser.add_argument('-NW', '--num_walkers', type = int, default = 10, help = 'Number of MCMC walkers passed to emcee.')
parser.add_argument('-NS', '--num_steps', type = int, default = 1500, help = 'Number of walker steps passed to emcee.')
parser.add_argument('-MP', '--multiprocess', type = int, default = 1, help = '1 for multiprocessing; uses 1 core per mass bin, 0 for sequential. Default is 1.')
parser.add_argument('-LL', '--loglambda', type = int, default = 0, help = 'Have the lambda parameter scale logarithmically instead of linearly. Will alter filename structure. Default is 0.')
parser.add_argument('-FL', '--fix_lambda', type = int, default = 0, help = 'Have the lambda be fixed to a value of 0 to emulate a single Gaussian. Default is 0.')
parser.add_argument('-C', '--catalogued', type = int, default = 1, help = 'Whether radial bins are precalculated (and catalogued). 1 for True. Default is 1.')

args = parser.parse_args()
mass_range = np.arange(args.lower_mass, args.upper_mass + args.step, args.step).astype(np.float32)
mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])

BASEPATH = '/disks/cosmodm/vdvuurst'
data_dir = os.path.join(BASEPATH,f'data/OneHalo_{args.step}dex')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# Set some args to bools
overwrite = bool(args.overwrite)
multiprocess = bool(args.multiprocess) and len(mass_bins) > 1 #if we have only 1 mass bin we do not need to go through the hassle of multiprocessing
verbose = bool(args.verbose) and not multiprocess # set verbose to false during multiprocess
loglambda = bool(args.loglambda)
catalogued = bool(args.catalogued)
fix_lambda = bool(args.fix_lambda)

def create_kwargs(**kwargs):
    return kwargs

if args.lower_radius and args.upper_radius:
    lr = float(args.lower_radius)
    ur = float(args.upper_radius)
    create_range = False
else:
    lr = 0.
    ur = 5.
    create_range = True

#NOTE: hardcoded r_start, r_stop and r_steps with this, might make modifiable later
default_kwargs = create_kwargs(r_start= lr, r_stop = ur, r_steps = 18, bins= rice_bins, bounds = [(50, 1000), (50, 1000), (0, 1)], plot= True,
                            nwalkers = args.num_walkers, nsteps = args.num_steps, non_bin_threshold = -1,
                            distname = 'Modified Gaussian', verbose = verbose, save_params = True, overwrite = overwrite,
                            return_values = False, loglambda = loglambda, fix_lambda = fix_lambda)


def _create_iterable_input(**kwargs):
    fitters = []
    if create_range:
        r_range = modified_logspace(kwargs['r_start'], kwargs['r_stop'], kwargs['r_steps']) 
    else:
        r_range(kwargs['r_start'], kwargs['r_stop'])
    
    kwargs['rbins'] = np.array([[r_range[i],r_range[i+1]] for i in range(len(r_range)-1)])

    for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
        filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
        filepath =  os.path.join(data_dir,filename)

        fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, joint = False, loglambda = kwargs['loglambda'])

        fitters.append(fitter)

    return [[f, m, kwargs] for f,m in zip(fitters,reversed(mass_bins))]

def _run_experiment_radial_bins(inpt):
    fitter, mass_bin, kwargs = inpt
    mass_path = os.path.join(data_dir, f'M_1{mass_bin[0]}-1{mass_bin[1]}')
    fitter.fit_to_radial_bins(catalogued = catalogued, datapath = mass_path, **kwargs)
    tqdm.write(f'MASS BIN 1{mass_bin[0]}-1{mass_bin[1]} dex COMPLETED')


#This makes plots that are made in the processes pretty by changing global mpl settings
format_plot()

if multiprocess:
    # all kwargs given here explicitly will be passed on to ONEHALO_fitter.fit_to_radial_bins() 
    iterable_input = _create_iterable_input(method = args.method.lower(), mass_bins = mass_bins, **default_kwargs)

    NPROCS = len(mass_bins) # 1 per mass bin, so 7 for default run
    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(_run_experiment_radial_bins, iterable_input):
            pbar.update()

else:
    match args.method.lower():
        case 'emcee':
            for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
                if verbose: print(f'WORKING ON MASS BIN M_1{mass_bin[0]}-1{mass_bin[1]}')
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, joint = False, loglambda = default_kwargs['loglambda'])
                                        # initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize/{filehead}.json', joint = False)
                
                if create_range:
                    r_range = modified_logspace(default_kwargs['r_start'], default_kwargs['r_stop'], default_kwargs['r_steps']) 
                else:
                    r_range = np.array([default_kwargs['r_start'], default_kwargs['r_stop']])
                default_kwargs['rbins'] = np.array([[r_range[i],r_range[i+1]] for i in range(len(r_range)-1)])
                mass_path = os.path.join(data_dir, f'M_1{mass_bin[0]}-1{mass_bin[1]}')

                fitter.fit_to_radial_bins(method='emcee',datapath = mass_path, **default_kwargs)
                print()


    #TODO: stuff with kwargs will break, but below will prob never be used anyway
        case 'minimize':
            for mass_bin in tqdm(reversed(mass_bins)): # High mass bins first, since these have the least entries
                if verbose: print()
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                #TODO create similar overwrite structure

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, joint = False)
                
                fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True,
                                                    plot = True, bins = args.bins, overwrite = overwrite)

        case 'both':
            for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                #TODO create similar overwrite structure

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, joint = False)

                fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True, plot = True, bins = args.bins, overwrite = overwrite)
                print(f'MINIMIZE FINISHED ON {filename}')
                res,err = fitter.fit_to_radial_bins(method='emcee', verbose = verbose, save_params = True, plot = True,
                                                    overwrite = overwrite, nwalkers = args.num_walkers, nsteps = args.num_steps)
                print()


                



