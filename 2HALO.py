import numpy as np
import pandas as pd
import swiftsimio as sio
import os
import h5py

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
        
        boxsize, self.half_boxsize, self.n_particles = _extract_information_from_path(PATH)
        self.COM = self.COM % boxsize #if any coordinate value is negative or larger than box size - it gets mapped into the box

    def create_catalogue(self, min_mass: np.float32, max_mass: np.float32, N_bound: int = 100, only_centrals: bool = True):
        bound_mask = _make_nbound_mask(self.NoofBoundParticles, N_bound)
        mass_selection = _make_mass_mask(self.FOFMass, min_mass, max_mass)
        central_selection = self.IsCentral if only_centrals else np.ones_like(bound_mask).astype(bool)
        
        mask = bound_mask & mass_selection & central_selection
        selection_size = np.sum(mask)

        # with h5py.File(f"data/velocity_data_M{str(min_mass).replace('.','_')}_{str(max_mass).replace('.','_')}.h5", "w") as f:
            # Second argument is shape and will just fill with zeroes. Since we blot out self comparison the shape of these will be selection_size - 1 right? All the halos - 1? No, this is wrong.
            # Running sowmya's code yields fields of size 767657375 - which I do not know how she got
            # dset_radial = f.create_dataset("radial_distances", (selection_size - 1,), dtype=np.float32, compression="gzip")
            # dset_velocities = f.create_dataset("velocity_differences", (selection_size - 1,), dtype=np.float32, compression="gzip")
            # dset_masses = f.create_dataset("masses", (selection_size - 1,), dtype=np.float32, compression="gzip")


        

