import numpy as np
import h5py
from typing import Tuple
import argparse
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize

from TWOHALO import _make_mass_mask, _make_nbound_mask

class ONEHALO:
    def __init__(self, PATH: str, filename: str):
        self.PATH = PATH
        self.filename = filename

        with h5py.File(PATH, "r") as handle:
            NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
            prelim_mask = _make_nbound_mask(NoofBoundParticles, 100) 
            
            self.IsCentral = handle["InputHalos/IsCentral"][:][prelim_mask].astype(bool) #set to bool so it can be used as a mask
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:][prelim_mask]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:][prelim_mask]
            self.HOSTHALOINDEX=handle["SOAP/HostHaloIndex"][:][prelim_mask]
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:][prelim_mask]
            self.SOMass = handle['SO/200_mean/TotalMass'][:][prelim_mask]
            self.boxsize = handle['Header'].attrs['BoxSize'][0]

        self.half_boxsize = self.boxsize / 2
        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box
