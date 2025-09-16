import numpy as np
import pandas as pd
import swiftsimio as sio
import os
import h5py

def load_in_data(PATH: str) -> pd.DataFrame:
    with h5py.File(PATH, "r") as handle:
        TotalMass = handle["ExclusiveSphere/100kpc/TotalMass"][:]
        # StellarMass = handle["ExclusiveSphere/100kpc/StellarMass"][:] ## NOT USED IN SOWMYA's CODE
        # COMstellarvelocity = handle["ExclusiveSphere/100kpc/StellarCentreOfMassVelocity"][:] ## NOT USED IN SOWMYA's CODE
        COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
        Trackid = handle["InputHalos/HBTplus/TrackId"][:]
        HOSTFOFID = handle["InputHalos/HBTplus/HostFOFId"][:]
        HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
        HOSTHALOINDEX = handle["SOAP/HostHaloIndex"][:]
        FOFMass = handle["InputHalos/FOF/Masses"][:]
        NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
        NoofDMParticles = handle["ExclusiveSphere/100kpc/NumberOfDarkMatterParticles"][:]
        COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
        IsCentral = handle["InputHalos/IsCentral"][:]
    
    df = pd.DataFrame({
        'HOST_FOF' :  HOSTFOFID,
        'HostHaloIndex':HOSTHALOINDEX, # -1 for central halos
        'Catalogue Index': HaloCatalogueIndex,
        'Track ID' : Trackid,
        'mass':TotalMass,
        'FOFMass':FOFMass,
        'COM v- x':COMvelocity[:,0],
        'COM v- y':COMvelocity[:,1],
        'COM v- z':COMvelocity[:,2],
        'COM - x':COM[:,0],
        'COM - y':COM[:,1],
        'COM - z':COM[:,2],
        'Bound Particles No':NoofBoundParticles,
        'DM Particles No': NoofDMParticles,
        'CENTRAL':IsCentral
    })

    # Why was this added
    df['INDEX_HOST_HALOS'] = np.asarray(df.index)
    
    return df