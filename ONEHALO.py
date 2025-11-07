import numpy as np
import h5py
from typing import Tuple
import argparse
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
import os

from TWOHALO import _make_nbound_mask



class ONEHALO:
    def __init__(self, PATH: str):
        self.PATH = PATH

        with h5py.File(self.PATH, "r") as handle:
            # NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
            # # prelim_mask = _make_nbound_mask(NoofBoundParticles, 100) 
            # prelim_mask = np.ones_like(NoofBoundParticles).astype(bool)

            self.IsCentral = handle["InputHalos/IsCentral"][:].astype(bool) #set to bool so it can be used as a mask
            self.HaloIndices = np.arange(self.IsCentral.size)
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
            self.HostHaloIndex = handle["SOAP/HostHaloIndex"][:] # -1 for centrals
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
            self.SOMass = handle['SO/200_mean/TotalMass'][:]
            # Full_SOMAss =  handle['SO/200_mean/TotalMass'][:]
            # FOFMass=handle["InputHalos/FOF/Masses"][:]

            self.boxsize = handle['Header'].attrs['BoxSize'][0]

        self.half_boxsize = self.boxsize / 2
        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

        # ~IsCentral picks out sattelites, this picks out all centrals !that host sattelites! and how many sattelites they host
        self.HostHaloIDs, self.subhalos_per_host_tot = np.unique(self.HostHaloIndex[~self.IsCentral], return_counts = True) 

    @staticmethod
    def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32) -> np.ndarray:
        if m_min in [0,-1,np.nan, None]:
            return (mass <= 10**m_max)
        elif m_max in [0,-1,np.nan, None]:
            return (10**m_min <= mass)
        return (10**m_min <= mass) & (mass <= 10**m_max) 
    
    def create_catalogue(self, massbin: Tuple[np.float32,np.float32], filename: str):
        self.lower_mass, self.upper_mass = massbin
        self.filename = filename

        # Select the relevant mass range, since the lowest is 10**12 Msol and particle mass is 10^9 
        # we do not need to explicitly filter for particle number
        mass_mask = self._make_mass_mask(self.SOMass, self.lower_mass, self.upper_mass) #functions as a central mask implicitly
        HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central (?)
        # print(np.any(NoofBoundParticles[HostIndices] < 100)) #Check just in case

        # From the catalogue of sattelite hosting haloes, select only those we know to have data for  
        Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
        subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask]
        HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] # This now replaces the previous HostIndices

        SatteliteMask = np.isin(self.HostHaloIndex, HostIndices) #pick out all the sattelites from relevant hosts

        # Get the host COM pos and vel, same for sattelites
        HostCOMs, HostVels = self.COM[HostIndices], self.COMvelocity[HostIndices]
        SatSorter = np.argsort(self.HostHaloIndex[SatteliteMask]) #NOTE: verify if this is necessary
        SatCOMs, SatVels = self.COM[SatteliteMask][SatSorter], self.COMvelocity[SatteliteMask][SatSorter]

        relative_COMs = SatCOMs - np.repeat(HostCOMs, subhalos_per_host, axis = 0)
        relative_vels = SatVels - np.repeat(HostVels, subhalos_per_host, axis = 0)

        with h5py.File(self.filename, 'w') as file:
            file.create_dataset('rel_pos', data  = relative_COMs, dtype = np.float32)
            file.create_dataset('rel_vels', data  = relative_vels, dtype = np.float32)
