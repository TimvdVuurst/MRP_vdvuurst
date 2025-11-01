import os
import numpy as np
from TWOHALO import TWOHALO, format_plot
from itertools import combinations_with_replacement
import argparse

def create_filename_from_mass_range(datadir, mass_range_primary, mass_range_secondary):
    return f"{datadir}/velocity_data_M1_1{mass_range_primary[0]}-1{mass_range_primary[1]}_M2_1{mass_range_secondary[0]}-1{mass_range_secondary[1]}.hdf5"


SOAP_PATH_DEFAULT = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5"

parser = argparse.ArgumentParser()
parser.add_argument('-M1','--lower_mass', type = np.float32, default = 2, help = 'Lower bound of the mass range in dex above 10^10 Msun. This is inclusive! Defaults to 2.')
parser.add_argument('-M2','--upper_mass', type = np.float32, default = 6, help = 'Upper bound of the mass range in dex above 10^10 Msun. This is exclusive! Defaults to 6.') #always EXCLUSIVE upper bound
parser.add_argument('-S', '--step', type = np.float32, default = 0.5, help = 'Size of bins in dex. Defaults to 0.5.')
parser.add_argument('-P', '--path_to_soap', type = str, default = SOAP_PATH_DEFAULT, help = 'Path specifying the SOAP-HBT data to be used. Should point to SOAP hdf5 file. Defaults to L1000N1800 @ z= 0.')
parser.add_argument('-O', '--overwrite', type = int, default = 1, help = 'If a catalogue already exist, control whether to overwrite it.')
args = parser.parse_args()

mass_range = np.arange(args.lower_mass, args.upper_mass, args.step).astype(np.float32)
mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])
bin_combis = np.array(list(combinations_with_replacement(mass_bins,2)))

#TODO rename existing data folder
BASEPATH = '/disks/cosmodm/vdvuurst'
data_dir = os.path.join(BASEPATH,f'data/M1{args.lower_mass}-1{args.upper_mass-args.step}_{args.step}dex')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# initializing the class reads in the SOAP file, we only need to do this once
# but just set the filename to which we save to a different value every time
twohalo = TWOHALO(PATH = args.path_to_soap, filename = None)

for bin_prim, bin_sec in reversed(bin_combis): # Start from the high mass bins, these are much less of these pairs

    bin_prim, bin_sec = tuple(bin_prim), tuple(bin_sec) 
    catalogue_file = create_filename_from_mass_range(data_dir, bin_prim, bin_sec)
    twohalo.filename = catalogue_file
    file_exists = os.path.isfile(catalogue_file)

    print(f"WORKING ON MASS BINS: PRIMARY {bin_prim} -  SECONDARY {bin_sec}")

    if file_exists and not bool(args.overwrite):
        print(f'{catalogue_file} already exists and --overwrite (-O) is set to false, skipping...\n')
        continue
    else:
        if file_exists:
            print(f'{catalogue_file} already exists, overwriting...\n')

        if bin_prim[0] <= 13.5 and bin_sec[0] != 15.: #there many pairs in these cases
            twohalo.create_subsampled_catalogue(mass_range_primary = bin_prim, mass_range_secondary = bin_sec)
        else:  # no subsampling needed in the other cases
            twohalo.create_full_catalogue(mass_range_primary = bin_prim, mass_range_secondary = bin_sec)

    format_plot()
    # twohalo.plot_velocity_histograms()
    # twohalo.plot_moments()

    print('COMPLETED SUCCESFULLY\n')


    