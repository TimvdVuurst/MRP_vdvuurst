import numpy as np
import os
import h5py
from typing import Tuple
import argparse
from tqdm import tqdm


# def _extract_information_from_path(PATH: str):
#     sim = PATH.split('/')[4] #fourth subfolder is always the sim judging from /net

#     #of the form LxxxxNxxxx
#     boxsize = int(sim[1:4]) # units of Mpc
#     n_particles = int(sim[6:])

    # return boxsize, boxsize / 2, n_particles

def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32) -> np.ndarray:
    return (10**m_min <= mass) & (mass <= 10**m_max) # base 10 since FOF masses aren't in units of 10^10 Msol

def _make_nbound_mask(bound: np.ndarray, N_min: np.float32):
    return bound >= N_min

class TWOHALO:
    def __init__(self, PATH: str):
        self.PATH = PATH
        with h5py.File(PATH, "r") as handle:
            self.TotalMass = handle["ExclusiveSphere/100kpc/TotalMass"][:]
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
            self.FOFMass = handle["InputHalos/FOF/Masses"][:]
            self.NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
            self.IsCentral = handle["InputHalos/IsCentral"][:].astype(bool) #set to bool so it can be used as a mask
            self.boxsize = handle['Header'].attrs['BoxSize'][0]
        
        self.half_boxsize = self.boxsize / 2
        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

    def create_catalogue(self, mass_range_primary: Tuple[np.float32,np.float32],
                          mass_range_secondary: Tuple[np.float32,np.float32],
                          N_bound: int = 100, only_centrals: bool = True):
        """_summary_

        Args:
            mass_range_primary (tuple): Must be in order (MIN,MAX)
            mass_range_secondary (tuple): Must be in order (MIN,MAX)
            N_bound (int, optional): _description_. Defaults to 100.
            only_centrals (bool, optional): _description_. Defaults to True.
        """
        bound_mask = _make_nbound_mask(self.NoofBoundParticles, N_bound)
        mass_mask = _make_mass_mask(self.FOFMass, *mass_range_primary) # mass bin of the primaries
        central_selection = self.IsCentral if only_centrals else np.ones_like(bound_mask).astype(bool) # pick out the centrals if so desired
        
        # final mask for the primaries, select primaries
        primary_mask = bound_mask & mass_mask & central_selection
        primary_selection_size = np.sum(primary_mask)
        primary_pos = self.COM[primary_mask] % self.boxsize # Map all positions to periodic box
        primary_vel = self.COMvelocity[primary_mask]
        # primary_mass = self.FOFmass[primary_mask] # NOT USED
        primary_ID = self.HaloCatalogueIndex[primary_mask] 

        # selection of the secondaries, differs only in mass range
        secondary_mass_selection = _make_mass_mask(self.FOFMass, *mass_range_secondary) & bound_mask & central_selection
        secondary_selection_size = secondary_mass_selection.sum()
        secondary_pos = self.COM[secondary_mass_selection] % self.boxsize # Map all positions to periodic box
        secondary_vel = self.COMvelocity[secondary_mass_selection]
        secondary_mass = self.FOFMass[secondary_mass_selection]
        secondary_ID = self.HaloCatalogueIndex[secondary_mass_selection]

        intersection_length = np.intersect1d(primary_ID, secondary_ID).shape[0] 
        primaries_are_subset = (intersection_length == primary_selection_size)

        # TODO: Think of a better naming convention?
        filename = f"data/velocity_data_M{str(mass_range_primary[0]).replace('.','_')}_{str(mass_range_primary[1]).replace('.','_')}.hdf5"
        print(f'\nNow working on {self.PATH}, writing to {filename}...\n')

        dset_shape = (secondary_selection_size - intersection_length) * (primary_selection_size - intersection_length) + \
                     (intersection_length * (secondary_selection_size - 1))

        with h5py.File(filename, "w") as f:
            # both radial distances and velocities are projected so that they're 1D
            dset_radial = f.create_dataset("radial_distances", (dset_shape,), dtype=np.float32, compression="gzip")
            dset_velocities = f.create_dataset("velocity_differences", (dset_shape,), dtype=np.float32, compression="gzip")
            dset_masses = f.create_dataset("masses", (dset_shape,), dtype=np.float32, compression="gzip")

            # Keeping to a for loop since array manipulation would be too memory intensive and thus slower (tested)
            counter = 0
            number_of_comparisons = secondary_selection_size - 1 # Holds if primaries are subset
            for pos1, vel1, id1 in tqdm(zip(primary_pos, primary_vel, primary_ID), total = len(primary_pos)):
                self_compare_mask = secondary_ID != id1 # Exclude self-comparison

                if not primaries_are_subset:
                    number_of_comparisons = self_compare_mask.sum()

                # Positional differences with periodic boundary conditions, without self-comparison
                pos_diffs = (secondary_pos[self_compare_mask] - pos1 + self.half_boxsize) % self.boxsize - self.half_boxsize
                radial_distances = np.linalg.norm(pos_diffs, axis=1) 
                radial_unit_vectors = pos_diffs / radial_distances[:, np.newaxis] 

                # Project velocities to the connecting line between haloes
                vel_diffs = secondary_vel[self_compare_mask] - vel1
                projected_vels = np.einsum('ij,ij->i', vel_diffs, radial_unit_vectors)

                dset_radial[counter:counter+number_of_comparisons] = radial_distances
                dset_velocities[counter:counter+number_of_comparisons] = projected_vels
                dset_masses[counter:counter+number_of_comparisons] = secondary_mass[self_compare_mask]

                # Since the file is of size 5 - 10 GB  (this depends on mass-range) presumably it is okay to keep it in RAM
                # f.flush() #Not sure if this is advantageous runtime-wise

                counter += number_of_comparisons

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P','--path', type = str,
                         default = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5")
    parser.add_argument("-M1","--mass_range_primary", type = float, nargs = '+', default=[4,4.5])
    parser.add_argument("-M2","--mass_range_secondary", type = float, nargs = '+', default=[3.5,5.5])
    # These two probably won't be touched but for the sake of generalizing they're included
    parser.add_argument('-N','--number_of_bound_particles', type = int, default = 100)
    parser.add_argument('-C','--only_centrals', type = bool, default = True)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    twohalo = TWOHALO(args.path)
    twohalo.create_catalogue(mass_range_primary = tuple(args.mass_range_primary), 
                             mass_range_secondary = tuple(args.mass_range_secondary),
                             N_bound = args.number_of_bound_particles,
                             only_centrals = args.only_centrals)

    print(f'\n CODE EXITED SUCCESFULLY.')


    
    
