import numpy as np
import h5py
from typing import Tuple
import argparse
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize

from TWOHALO import _make_nbound_mask



class ONEHALO:
    def __init__(self, PATH: str, filename: str, massbin: Tuple[np.float32,np.float32]):
        self.PATH = PATH
        self.filename = filename
        self.lower_mass, self.upper_mass = massbin

        with h5py.File(PATH, "r") as handle:
            NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
            prelim_mask = _make_nbound_mask(NoofBoundParticles, 100) 
            
            self.IsCentral = handle["InputHalos/IsCentral"][:][prelim_mask].astype(bool) #set to bool so it can be used as a mask
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:][prelim_mask]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:][prelim_mask]
            self.HostHaloIndex = handle["SOAP/HostHaloIndex"][:][prelim_mask] # -1 for centrals
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:][prelim_mask]
            self.SOMass = handle['SO/200_mean/TotalMass'][:][prelim_mask]
            self.FOFMass=handle["InputHalos/FOF/Masses"][:][prelim_mask]

            self.boxsize = handle['Header'].attrs['BoxSize'][0]

        self.half_boxsize = self.boxsize / 2
        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

    @staticmethod
    def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32) -> np.ndarray:
        if m_min in [0,-1,np.nan, None]:
            return (mass <= 10**m_max)
        elif m_max in [0,-1,np.nan, None]:
            return (10**m_min <= mass)
        return (10**m_min <= mass) & (mass <= 10**m_max) 
    
    def velocity(self):
        #FOFMass of non-centrals = 0, so mass_mask implicitly picks out centrals
        mass_mask = self._make_mass_mask(self.FOFMass, self.lower_mass, self.upper_mass) 

        # gets all the subhaloes belonging to the centrals we picked out with the mass bin
        subhalo_mask = np.isin(self.HostHaloIndex, np.where(mass_mask)[0])  

        HostHaloIDs, subhalos_per_host = np.unique(self.HostHaloIndex[subhalo_mask], return_counts = True)
        
        # sorter = np.argsort(self.HostHaloIndex[subhalo_mask]) 

