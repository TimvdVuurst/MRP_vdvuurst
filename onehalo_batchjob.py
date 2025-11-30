import os
import numpy as np
from ONEHALO import ONEHALO, ONEHALO_fitter
from plotting import format_plot
import argparse
from tqdm import tqdm

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

args = parser.parse_args()

mass_range = np.arange(args.lower_mass, args.upper_mass + args.step, args.step).astype(np.float32)
mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])

BASEPATH = '/disks/cosmodm/vdvuurst'
data_dir = os.path.join(BASEPATH,f'data/OneHalo_{args.step}dex')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# onehalo = ONEHALO(PATH = args.path_to_soap) #change filename in loop structure
overwrite = bool(args.overwrite)
verbose = bool(args.verbose)

#TODO uncomment, maybe make an argument that can control whether we even catalogue at all? just to save the overhead
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

format_plot()

match args.method.lower():
    case 'emcee':
        for mass_bin in reversed(mass_bins): # High mass bins first, since these have the least entries
            if verbose: print(f'WORKING ON MASS BIN M_1{mass_bin[0]}-1{mass_bin[1]}')
            filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
            filepath =  os.path.join(data_dir,filename)
            file_exists = os.path.isfile(filename)

            filehead = filename.split('.hdf5')[0]

            fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize_binned/{filehead}.json', joint = False)
            
            res,err = fitter.fit_to_radial_bins(method='emcee', verbose = verbose, save_params = True, plot = True, overwrite = overwrite)
            print()

    case 'minimize':
        for mass_bin in tqdm(reversed(mass_bins)): # High mass bins first, since these have the least entries
            if verbose: print()
            filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
            filepath =  os.path.join(data_dir,filename)
            file_exists = os.path.isfile(filename)

            #TODO create similar overwrite structure

            filehead = filename.split('.hdf5')[0]

            fitter = ONEHALO_fitter(PATH = filepath, initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize_binned/{filehead}.json', joint = False)
             
            res,err = fitter.fit_to_radial_bins(method='minimize', verbose = verbose, save_params = True, plot = True, bins = args.bins, overwrite = overwrite)

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
            res,err = fitter.fit_to_radial_bins(method='emcee', verbose = verbose, save_params = True, plot = True, overwrite = overwrite)
            print()

            



