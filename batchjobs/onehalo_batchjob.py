from time import time

now = time()

import sys
if '/disks/cosmodm/vdvuurst' not in sys.path:
    sys.path.append('/disks/cosmodm/vdvuurst')


import os
import numpy as np
from ONEHALO import ONEHALO_fitter
from onehalo_plotter import format_plot
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool
from functions import modified_logspace, rice_bins, mkdir_if_non_existent, extract_mass_and_rad_from_filename
from itertools import product

from warnings import filterwarnings

# this might be a liiiitle dangerous but cleans up the output by a lot. that's because we often see invalid values in log10, but that's ok
filterwarnings('ignore',category = RuntimeWarning)

SOAP_PATH_DEFAULT = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5"

parser = ArgumentParser()
# Mass parameters
parser.add_argument('-M1','--lower_mass', type = np.float32, default = 2, help = 'Lower bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 2.')
parser.add_argument('-M2','--upper_mass', type = np.float32, default = 5.5, help = 'Upper bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 5.5.') 
parser.add_argument('-S', '--step', type = np.float32, default = 0.5, help = 'Size of mass bins in dex. Defaults to 0.5.')

#Radial parameteres
parser.add_argument('-r1','--lower_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!')
parser.add_argument('-r2','--upper_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!') 
parser.add_argument('-RB','--r_bins', type = int, default = 20, help = 'Numer of radial bins to use. Defaults to 20.') 
parser.add_argument('-RU', '--rad_unit', type = str, default = 'Rvir', choices = ['rvir', 'mpc', 'Mpc', 'Rvir'], help = 'Unit to use for radial bins. Either Mpc or Rvir (i.e. R200m), not case sensitive in the first letter.')

#Meta settings
parser.add_argument('-O', '--overwrite', type = int, default = 1, help = 'If a catalogue already exist, control whether to overwrite it. 1 for True, 0 for False.')
parser.add_argument('-M', '--method', type = str, default = 'emcee', help = 'Fitting procedure. Choose either emcee, minimize or both.')
parser.add_argument('-V', '--verbose', type = int, default = 1, help = 'Whether to print diagnostics and timings. 1 for True, 0 for False.')
parser.add_argument('-NW', '--num_walkers', type = int, default = 20, help = 'Number of MCMC walkers passed to emcee.')
parser.add_argument('-NS', '--num_steps', type = int, default = 500, help = 'Number of walker steps passed to emcee.')

#Efficiency controllers
parser.add_argument('-C', '--catalogued', type = int, default = 1, help = 'Whether radial bins are precalculated (and catalogued). 1 for True. Default is 1.')
parser.add_argument('-MP', '--multiprocess', type = int, default = 32, help = 'Number of cores to use for multiprocessing, 1 means sequential. Default is 32.')

# Variants
parser.add_argument('-LL', '--loglambda', type = int, default = 0, help = 'Have the lambda parameter scale logarithmically instead of linearly. Will alter filename structure. Default is 0.')
parser.add_argument('-SG', '--single_gauss', type = int, default = 0, help = 'Have the lambda be fixed to a value of 0 to emulate a single Gaussian. Default is 0.')
parser.add_argument('-FS','--flip_sigmas', type = int, default = 0, help = 'If the sigma_1 parameter is smaller than the sigma_2 parameter after fitting, flip them and cast lambda to 1 - lambda. Default is 0.')
parser.add_argument('-T', '--timeit', type = int, default = 0, help = 'Controls whether or not to time the process per bin and write to txt file. Default is 0.')


args = parser.parse_args()

r_unit = args.rad_unit
r_unit = r_unit[0].upper() + r_unit[1:].lower() #typeset for consistency

mass_range = np.arange(args.lower_mass, args.upper_mass + args.step, args.step).astype(np.float32)
mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])

BASEPATH = '/disks/cosmodm/vdvuurst'
data_dir = os.path.join(BASEPATH,f'data/OneHalo_{args.step}dex')
mkdir_if_non_existent(data_dir)

# Set some args to bools
overwrite = bool(args.overwrite)
multiprocess = bool(args.multiprocess > 1)
NPROCS = args.multiprocess # and this one for clarity
verbose = bool(args.verbose) and not multiprocess # set verbose to false during multiprocess
loglambda = bool(args.loglambda)
catalogued = bool(args.catalogued)
single_gauss = bool(args.single_gauss)
flip_sigmas = bool(args.flip_sigmas)
timeit = bool(args.timeit)

def create_kwargs(**kwargs):
    return kwargs

if args.lower_radius and args.upper_radius:
    multiprocess = False
    verbose = True
    lr = float(args.lower_radius)
    ur = float(args.upper_radius)
    create_range = False
else:
    lr = 0.
    if r_unit == 'Mpc':
        ur = 5.
    elif r_unit == 'Rvir':
        ur = 2.5 
    else:
        raise ValueError('This should not have happened, how did you get another radial unit?')
    create_range = True

# hardcoded in here, these are the only bins we want to NOT rerun if lambda > 0.5

flip_bins = ['M_12.0-12.5-r_0.00-0.07','M_12.5-13.0-r_0.00-0.07', 'M_13.0-13.5-r_0.00-0.07', 'M_12.0-12.5-r_0.07-0.14']
# create kwarg dictionary from default values and terminal input
default_kwargs = create_kwargs(r_start= lr, r_stop = ur, r_steps = args.r_bins, r_unit = r_unit, bins= rice_bins, plot= True,
                            nwalkers = args.num_walkers, nsteps = args.num_steps, non_bin_threshold = -1,
                            distname = 'Modified Gaussian', verbose = verbose, save_params = True, overwrite = overwrite,
                            return_values = False, loglambda = loglambda, single_gauss = single_gauss, flip_sigmas = flip_sigmas,
                            flip_bins = flip_bins, timeit = timeit, is_rerun = False, skip_rerun = False)


def _create_iterable_input(**kwargs):
    """ 
        Given a set of kwargs, create iterable input that imap_unordered can handle.
    Returns:
        list: list of lists, each containing a ONEHALO_fitter object, a massbin, optionally a radial bin and a set of kwargs.
    """

    if create_range:
        r_range = modified_logspace(kwargs['r_start'], kwargs['r_stop'], kwargs['r_steps']) 
    else:
        r_range = (kwargs['r_start'], kwargs['r_stop'])
    
    # set up the radial bins in the right format
    kwargs['rbins'] = np.array([[r_range[i],r_range[i+1]] for i in range(len(r_range)-1)])

    # we need a fitter object for every mass bin
    fitters = []
    for i,mass_bin in enumerate(kwargs['mass_bins']):
        filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
        filepath =  os.path.join(data_dir, filename)

        #create fitter objects to use. Only need it to load in data if we use 
        # non-catalogued data (i.e. we need to calculate the radial bin masks within the function)
        fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, loglambda = kwargs['loglambda'], load = not catalogued, enforce_sigma_2_smaller = False)

        fitters.append(fitter)

    if catalogued:
        fitters_and_mass_bins = list(zip(fitters, mass_bins))
        return [[f, m, r, kwargs] for (f,m), r in product(fitters_and_mass_bins, kwargs['rbins'])]

    return [[f, m, kwargs] for (f,m), r in zip(fitters, mass_bins)]

def _run_experiment_radial_bins(inpt):
    if catalogued:
        fitter, mass_bin, rbin, kwargs = inpt
        mass_path = os.path.join(data_dir, f'M_1{mass_bin[0]}-1{mass_bin[1]}')
        fitter.fit_to_catalogued_bin(rbin = rbin, datapath = mass_path, **kwargs)

    else:
        fitter, mass_bin, kwargs = inpt
        mass_path = os.path.join(data_dir, f'M_1{mass_bin[0]}-1{mass_bin[1]}')
        fitter.fit_to_radial_bins(catalogued = catalogued, datapath = mass_path, **kwargs)
        tqdm.write(f'MASS BIN 1{mass_bin[0]}-1{mass_bin[1]} dex COMPLETED')


#This makes plots that are made in the processes pretty by changing global mpl settings
format_plot()

if multiprocess:
    # all kwargs given here explicitly will be passed on to ONEHALO_fitter.fit_to_radial_bins() 
    iterable_input = _create_iterable_input(method = args.method.lower(), mass_bins = mass_bins, **default_kwargs)

    if not catalogued and multiprocess:
        NPROCS = len(mass_bins) # 1 per mass bin, so 7 for default run

    #logfile to keep track of bad fits in the process
    #create it and/or empty it
    with open('./logs/bad_fits_onehalo.txt', 'w') as f:
        f.close()

    overhead = time() - now

    print(f'OVERHEAD WAS {overhead:.3f} seconds.')

    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(_run_experiment_radial_bins, iterable_input):
            pbar.update()
    
    # Now repeat the process for the bad fits without enforcing sigma_1 > sigma_2 explicitly
    with open('./logs/bad_fits_onehalo.txt', 'r') as f:
        bad_bins = f.readlines()
    
    num_bad_bins = len(bad_bins)
    print(f'Found {num_bad_bins} bad bins{', refitting...' if num_bad_bins != 0 else '.'}')
    if num_bad_bins != 0:
        mass_bins = [[*extract_mass_and_rad_from_filename(bb)[:2]] for bb in bad_bins]
        rad_bins = [[*extract_mass_and_rad_from_filename(bb)[2:]] for bb in bad_bins]
  
        fitters = []
        for i,mass_bin in enumerate(mass_bins):
            filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
            filepath =  os.path.join(data_dir, filename)

            #create fitter objects to use. Only need it to load in data if we use 
            # non-catalogued data (i.e. we need to calculate the radial bin masks within the function)
            fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, loglambda = default_kwargs['loglambda'], load = not catalogued, enforce_sigma_2_smaller = True) #this last argument is the crucial difference here
            fitters.append(fitter)

        fitters_and_mass_bins = list(zip(fitters, mass_bins))
        default_kwargs['is_rerun'] = True
        iterable_input = [[f, m, r, default_kwargs] for (f,m), r in zip(fitters_and_mass_bins, rad_bins)]
        if NPROCS > num_bad_bins: NPROCS = num_bad_bins

        with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
            for _ in p.imap_unordered(_run_experiment_radial_bins, iterable_input):
                pbar.update()

else:
    match args.method.lower():
        case 'emcee':
            for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
                if verbose: print(f'WORKING ON MASS BIN M_1{mass_bin[0]}-1{mass_bin[1]}')
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir, filename)

                file_exists = os.path.isfile(filename)

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None, loglambda = default_kwargs['loglambda'], enforce_sigma_2_smaller = True)
                                        # initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize/{filehead}.json')
                
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

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None)
                
                fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True,
                                                    plot = True, bins = args.bins, overwrite = overwrite)

        case 'both':
            for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                #TODO create similar overwrite structure

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = None)

                fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True, plot = True, bins = args.bins, overwrite = overwrite)
                print(f'MINIMIZE FINISHED ON {filename}')
                res,err = fitter.fit_to_radial_bins(method='emcee', verbose = verbose, save_params = True, plot = True,
                                                    overwrite = overwrite, nwalkers = args.num_walkers, nsteps = args.num_steps)
                print()


                



