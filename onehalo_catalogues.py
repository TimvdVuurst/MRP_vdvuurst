import os
import numpy as np
from ONEHALO import ONEHALO
import argparse
from tqdm import tqdm
from functions import modified_logspace, mkdir_if_non_existent

SOAP_PATH_DEFAULT = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5"

parser = argparse.ArgumentParser()
parser.add_argument('-C','--catalogue', type = str, default = 'rad',  help = 'Select whether to create mass, radial or both catalogues. Defaults to radial.')
parser.add_argument('-M1','--lower_mass', type = np.float32, default = 2, help = 'Lower bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 2.')
parser.add_argument('-M2','--upper_mass', type = np.float32, default = 5.5, help = 'Upper bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 5.5.') #always EXCLUSIVE upper bound

parser.add_argument('-r1','--lower_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!')
parser.add_argument('-r2','--upper_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!') 
parser.add_argument('-RB','--r_bins', type = int, default = 20, help = 'Numer of radial bins to use. Defaults to 20.')
parser.add_argument('-RU', '--rad_unit', type = str, default = 'rvir', choices = ['rvir', 'mpc', 'Mpc'], help = 'Unit to use for radial bins. Either Mpc or rvir (R200m)')

parser.add_argument('-S', '--step', type = np.float32, default = 0.5, help = 'Size of bins in dex. Defaults to 0.5.')
parser.add_argument('-P', '--path_to_soap', type = str, default = SOAP_PATH_DEFAULT, help = 'Path specifying the SOAP-HBT data to be used. Should point to SOAP hdf5 file. Defaults to L1000N1800 @ z= 0.')
parser.add_argument('-O', '--overwrite', type = int, default = 1, help = 'If a catalogue already exist, control whether to overwrite it. 1 for True, 0 for False.')
parser.add_argument('-V', '--verbose', type = int, default = 1, help = 'Whether to print diagnostics and timings. 1 for True, 0 for False.')

args = parser.parse_args()

choice = args.catalogue.lower()
if choice == 'radius': choice = 'rad'
choices = ['mass','rad', 'radius', 'both']
if choice not in choices:
    raise ValueError(f'Catalogue type "{choice}" not recognized. Choose from {*choices,} (not caps sensitive).')

r_unit = args.rad_unit
r_unit = r_unit[0].upper() + r_unit[1:].lower() #typeset


mass_range = np.arange(args.lower_mass, args.upper_mass + args.step, args.step).astype(np.float32)
mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])

BASEPATH = '/disks/cosmodm/vdvuurst'
data_dir = os.path.join(BASEPATH,f'data/OneHalo_{args.step}dex')
mkdir_if_non_existent(data_dir)

overwrite = bool(args.overwrite)
verbose = bool(args.verbose)

onehalo = ONEHALO(SOAP_PATH_DEFAULT)
print('SOAP data loaded in and preprocessed.')

def create_mass_catalogue(mass_bin, mass_filename, mass_filepath):
    mass_file_exists = os.path.isfile(mass_filepath)
    if mass_file_exists and not overwrite and verbose:
        tqdm.write(f'{mass_filename} already exists and --overwrite (-O) is set to false, skipping...\n')
        return
    else:
        onehalo.create_catalogue(massbin = mass_bin, filename = mass_filepath)

def create_rad_catalogue(mass_filename):
    
    if args.lower_radius and args.upper_radius:
        lr = float(args.lower_radius)
        ur = float(args.upper_radius)
    else:
        lr = 0.
        if r_unit == 'Mpc':
            ur = 5.
        elif r_unit == 'Rvir':
            ur = 2.5 
        else:
            raise ValueError('This should not have happened, how did you get another radial unit?')
        
    rbins = modified_logspace(lr, ur, args.r_bins) 

    mass_head = mass_filename.replace('.hdf5','')
    mass_path = os.path.join(data_dir, mass_head)
    tqdm.write(f'WORKING ON {mass_head}')

    iterable = tqdm(range(len(rbins) - 1)) if verbose and choice == 'rad' else range(len(rbins) - 1)
    for i in iterable:
        rbin = (rbins[i],rbins[i + 1])
        rad_filename = f'r_{rbin[0]:.2f}-{rbin[1]:.2f}.hdf5'

        mkdir_if_non_existent(mass_path)
        
        rad_filepath = os.path.join(mass_path, r_unit, rad_filename) # split data storage based on used radial unit
        rad_file_exists = os.path.isfile(rad_filepath)

        if rad_file_exists and not overwrite:
            tqdm.write(f'{mass_head}/{rad_filename} already exists and --overwrite (-O) is set to false, skipping...\n')
            return
        
        else:
            too_little = onehalo.create_radial_catalogue(rbin, rad_filepath, mass_filename = os.path.join(data_dir, mass_filename), r_unit = r_unit)
            if too_little and verbose:
                    tqdm.write(f'{mass_head}/{rad_filename} contains too little datapoints, skipping...')


if __name__ == '__main__':
    iterable = tqdm(reversed(mass_bins)) if verbose and choice in ['mass','both'] else reversed(mass_bins)
    for mass_bin in iterable: # High mass bins first, since these have the least entries
        mass_filename =  f'M_1{mass_bin[0]}-1{mass_bin[1]}.hdf5'
        mass_filepath =  os.path.join(data_dir, mass_filename)
        match choice:
            case 'mass':
                create_mass_catalogue(mass_bin, mass_filename, mass_filepath)

            case 'rad':
                create_rad_catalogue(mass_filename)
            
            case 'both':
                create_mass_catalogue(mass_bin, mass_filename, mass_filepath)
                create_rad_catalogue(mass_filename)