import numpy as np
import pandas as pd
import swiftsimio as sio
import os
import h5py
from typing import Tuple

def _extract_information_from_path(PATH: str):
    sim = PATH.split('/')[4] #fourth subfolder is always the sim judging from ./net/

    #of the form LxxxxNxxxx
    boxsize = int(sim[1:4]) # units of Mpc
    n_particles = int(sim[6:])

    return boxsize, boxsize / 2, n_particles

def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32) -> np.ndarray:
    return (m_min < mass) & (mass < m_max)

def _make_nbound_mask(bound: np.ndarray, N_min: np.float32):
    return bound >= N_min

class TWOHALO:
    def __init__(self, PATH: str):
        with h5py.File(PATH, "r") as handle:
            self.TotalMass = handle["ExclusiveSphere/100kpc/TotalMass"][:]
            # StellarMass = handle["ExclusiveSphere/100kpc/StellarMass"][:] ## NOT USED IN SOWMYA's CODE
            # COMstellarvelocity = handle["ExclusiveSphere/100kpc/StellarCentreOfMassVelocity"][:] ## NOT USED IN SOWMYA's CODE
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
            self.Trackid = handle["InputHalos/HBTplus/TrackId"][:]
            self.HOSTFOFID = handle["InputHalos/HBTplus/HostFOFId"][:]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
            self.HOSTHALOINDEX = handle["SOAP/HostHaloIndex"][:]
            self.FOFMass = handle["InputHalos/FOF/Masses"][:]
            self.NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
            self.NoofDMParticles = handle["ExclusiveSphere/100kpc/NumberOfDarkMatterParticles"][:]
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
            self.IsCentral = handle["InputHalos/IsCentral"][:].astype(bool) #set to bool so it can be used as a mask
        
        self.boxsize, self.half_boxsize, self.n_particles = _extract_information_from_path(PATH)
        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

    def create_catalogue(self, mass_range_primary: Tuple[np.float32,np.float32], mass_range_secondary: Tuple[np.float32,np.float32],
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
        
        # if primaries are *entirely* a subset of the secondaries, this should hold
        primaries_are_subset = (mass_range_secondary[0] <= mass_range_primary[0]) & (mass_range_secondary[1] >= mass_range_primary[1])

        # final mask for the primaries, select primaries
        primary_mask = bound_mask & mass_mask & central_selection
        primary_selection_size = np.sum(primary_mask)
                
        primary_pos = self.COM[primary_mask] % self.boxsize # Map all positions to periodic box
        primary_vel = self.COMvelocity[primary_mask]
        primary_mass = self.FOFmass[primary_mask]
        primary_ID = self.HaloCatalogueIndex[primary_mask] 

        # selection of the secondaries, differs only in mass range
        # select secondaries
        secondary_mass_selection = _make_mass_mask(self.FOFMass, *mass_range_secondary) & bound_mask & central_selection
        secondary_selection_size = secondary_mass_selection.sum()

        secondary_pos = self.COM[secondary_mass_selection] % self.boxsize # Map all positions to periodic box
        secondary_vel = self.COMvelocity[secondary_mass_selection]
        secondary_mass = self.FOFMass[secondary_mass_selection]
        secondary_id = self.HaloCatalogueIndex[secondary_mass_selection]

        # TODO: Think of a better naming convention
        filename = f"data/velocity_data_M{str(mass_range_primary[0]).replace('.','_')}_{str(mass_range_primary[1]).replace('.','_')}.hdf5"
        with h5py.File(filename, "w") as f:
            dset_shape = ((primary_selection_size - 1), secondary_selection_size) # store as 2D array, so that we can pick out 2-halo terms individually

            # both radial distances and velocities are projected so that they're 1D
            dset_radial = f.create_dataset("radial_distances", dset_shape, dtype=np.float32, compression="gzip")
            dset_velocities = f.create_dataset("velocity_differences", dset_shape, dtype=np.float32, compression="gzip")
            dset_masses = f.create_dataset("masses", dset_shape, dtype=np.float32, compression="gzip")

            # Keeping to the for loop since array manipulation would be too memory intensive and thus slower (tested)
            counter = 0
            # for primary_idx, (pos1, vel1, id1) in enumerate(zip(primary_pos, primary_vel, primary_ID)): #enumerate prob not needed
            for pos1, vel1, id1 in zip(primary_pos, primary_vel, primary_ID):

                self_compare_mask = secondary_id != id1 # Exclude self-comparison

                # It may still occur that a primary is also a secondary 
                # if the primaries are not entirely a subset of the secondaries
                # But if they are entirely a subset we always have self-comparison to exclude
                number_of_comparisons = (secondary_selection_size -1) if primaries_are_subset else self_compare_mask.sum()

                # Positional differences with periodic boundary conditions, without self-comparison
                pos_diffs = (secondary_pos[self_compare_mask] - pos1 + self.half_box) % self.box_size - self.half_box
                radial_distances = np.linalg.norm(pos_diffs, axis=1) 
                radial_unit_vectors = pos_diffs / radial_distances[:, np.newaxis] 

                vel_diffs = secondary_vel[self_compare_mask] - vel1
                projected_vels = np.einsum('ij,ij->i', vel_diffs, radial_unit_vectors) # Project velocity to line connecting halos

                # Save to dset
                dset_radial[counter:counter+number_of_comparisons] = radial_distances
                dset_velocities[counter:counter+number_of_comparisons] = projected_vels
                dset_radial[counter:counter+number_of_comparisons] = secondary_mass[self_compare_mask]

                counter += number_of_comparisons

