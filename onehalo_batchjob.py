import os
import numpy as np
from ONEHALO import ONEHALO, ONEHALO_fitter
from plotting import format_plot
import argparse
from tqdm import tqdm
from multiprocessing import Pool

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
parser.add_argument('-S', '--step', type = np.float32, default = 0.5, help = 'Size of bins in dex. Defaults to 0.5.')
parser.add_argument('-P', '--path_to_soap', type = str, default = SOAP_PATH_DEFAULT, help = 'Path specifying the SOAP-HBT data to be used. Should point to SOAP hdf5 file. Defaults to L1000N1800 @ z= 0.')
parser.add_argument('-O', '--overwrite', type = int, default = 1, help = 'If a catalogue already exist, control whether to overwrite it. 1 for True, 0 for False.')
parser.add_argument('-B', '--bins', type = int, default = 100, help = 'Number of velocity bins')
parser.add_argument('-M', '--method', type = str, default = 'emcee', help = 'Fitting procedure. Choose either emcee, minimize or both.')
parser.add_argument('-V', '--verbose', type = int, default = 1, help = 'Whether to print diagnostics and timings. 1 for True, 0 for False.')
parser.add_argument('-NW', '--num_walkers', type = int, default = 10, help = 'Number of MCMC walkers passed to emcee.')
parser.add_argument('-NS', '--num_steps', type = int, default = 2000, help = 'Number of walker steps passed to emcee.')
parser.add_argument('-MP', '--multiprocess', type = int, default = 1, help = '1 for multiprocessing; uses 1 core per mass bin, 0 for sequential. Default is 1.')

args = parser.parse_args()

mass_range = np.arange(args.lower_mass, args.upper_mass + args.step, args.step).astype(np.float32)
mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])


BASEPATH = '/disks/cosmodm/vdvuurst'
data_dir = os.path.join(BASEPATH,f'data/OneHalo_{args.step}dex')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

overwrite = bool(args.overwrite)
verbose = bool(args.verbose)
multiprocess = bool(args.multiprocess) and len(mass_bins) > 1 #if we have only 1 mass bin we do not need to go through the hassle of multiprocessing


def _create_iterable_input(**kwargs):
    fitters = []
    for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
        filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
        filepath =  os.path.join(data_dir,filename)

        filehead = filename.split('.hdf5')[0]

        fitter = ONEHALO_fitter(PATH = filepath,
                                initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize/{filehead}.json', joint = False)

        fitters.append(fitter)

    return [[f, m, kwargs] for f,m in zip(fitters,mass_bins)]

def _run_experiment_radial_bins(inpt):
    fitter, mass_bin, kwargs = inpt
    fitter.fit_to_radial_bins(**kwargs)
    tqdm.write(f'MASS BIN 1{mass_bin[0]}-1{mass_bin[1]} dex COMPLETED')


#This makes plots that are made in the processes pretty by changing global mpl settings
format_plot()

if multiprocess:
    #In multiprocess we set verbose to false since it will interfere too much with output
    # all kwargs given here explicitly will be passed on to ONEHALO_fitter.fit_to_radial_bins() 
    iterable_input = _create_iterable_input(method = args.method.lower(), mass_bins = mass_bins, verbose = False, save_params = True, plot = True,
                            overwrite = overwrite, nwalkers = args.num_walkers, nsteps = args.num_steps, return_values = False)

    NPROCS = len(mass_bins) # 1 per mass bin, so 7 for default run
    with Pool(NPROCS) as p, tqdm(total=len(iterable_input)) as pbar:
        for _ in p.imap_unordered(_run_experiment_radial_bins, iterable_input):
            pbar.update()


else:
    #TODO uncomment, maybe make an argument that can control whether we even catalogue at all? just to save the commenting and overhead
    # for mass_bin in tqdm(reversed(mass_bins)): # High mass bins first, since these have the least entries
    #     filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
    #     filepath =  os.path.join(data_dir,filename)
    #     file_exists = os.path.isfile(filename)

    #     if file_exists and not overwrite:
    #         print(f'{filename} already exists and --overwrite (-O) is set to false, skipping...\n')
    #         continue
    #     else:
    #         if file_exists:
    #             print(f'{filename} already exists, overwriting...\n')
        
    #         onehalo.create_catalogue(massbin = mass_bin, filename = filepath)


    match args.method.lower():
        case 'emcee':
            for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
                if verbose: print(f'WORKING ON MASS BIN M_1{mass_bin[0]}-1{mass_bin[1]}')
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath,
                                        initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize/{filehead}.json', joint = False)
                
                res,err = fitter.fit_to_radial_bins(method='emcee', verbose = verbose, save_params = True, plot = True,
                                                    overwrite = overwrite, nwalkers = args.num_walkers, nsteps = args.num_steps)
                print()

        case 'minimize':
            for mass_bin in tqdm(reversed(mass_bins)): # High mass bins first, since these have the least entries
                if verbose: print()
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                #TODO create similar overwrite structure

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize/{filehead}.json', joint = False)
                
                res,err = fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True,
                                                    plot = True, bins = args.bins, overwrite = overwrite)

        case 'both':
            for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
                filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
                filepath =  os.path.join(data_dir,filename)
                file_exists = os.path.isfile(filename)

                #TODO create similar overwrite structure

                filehead = filename.split('.hdf5')[0]

                fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize/{filehead}.json', joint = False)
                res,err = fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True, plot = True, bins = args.bins, overwrite = overwrite)
                print(f'MINIMIZE FINISHED ON {filename}')
                res,err = fitter.fit_to_radial_bins(method='emcee', verbose = verbose, save_params = True, plot = True,
                                                    overwrite = overwrite, nwalkers = args.num_walkers, nsteps = args.num_steps)
                print()


                



