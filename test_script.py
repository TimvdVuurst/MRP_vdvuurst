import os
import numpy as np
import h5py 

#NOTE this function is only for subsampling already existing data
def subsample_existing_data(pair_data_path, number_of_bins = 20, max_radius = 300, num_per_bin = int(1e7), overwrite = False):
    # num_per_bin is how many pairs we want to keep per radial bin
    # max_radius is the radial distance cut-off in Mpc

    with h5py.File(pair_data_path,'r') as pair_data:
        radii = pair_data['radial_distances'][:]
        vels = pair_data['velocity_differences'][:]
        prim_masses = pair_data['primary_masses'][:]
        sec_masses = pair_data['secondary_masses'][:]
    
    # Overwrite prim_masses by a new array which is now filled with prim masses to keep the same shape across
    #NOTE this is only relevant for old data so in the future this might become obsolete
    if prim_masses.shape != sec_masses.shape:
        sec_per_prim = sec_masses.shape[0] // prim_masses.shape[0] #we can use integer division because we are sure that this is in fact an integer
        prim_masses = np.array([np.full(sec_per_prim, prim_masses[i]) for i in range(prim_masses.shape[0])]).flatten()
    print('data loaded in')

    radial_bins = np.logspace(0, np.log10(max_radius), number_of_bins)
    bin_idx =  np.digitize(radii, radial_bins) - 1 # -1 to correct index offset i.e. it will index to the lower bound of the bin

    unique_bins, radial_bin_counts = np.unique(bin_idx, return_counts = True)

    if overwrite:
        new_filepath = pair_data_path
    else:
        head, tail = os.path.split(pair_data_path)
        new_dir = os.path.join('/disks/cosmodm/vdvuurst/data', os.path.split(head)[1] + '_subsampled')
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        new_filepath = os.path.join(new_dir, tail) 

   # How many do we keep in the new dataset; we set a per radial bin ceiling
    dset_shape = radial_bin_counts[radial_bin_counts <= num_per_bin].sum() +\
                 radial_bin_counts[radial_bin_counts > num_per_bin].shape[0] * num_per_bin
    
    with h5py.File(new_filepath,'w') as f:
        dset_radial = f.create_dataset("radial_distances", (dset_shape,), dtype=np.float32)
        dset_velocities = f.create_dataset("velocity_differences", (dset_shape,), dtype=np.float32)
        dset_sec_masses = f.create_dataset("secondary_masses", (dset_shape,), dtype=np.float32)
        dset_prim_masses = f.create_dataset("primary_masses", (dset_shape,), dtype=np.float32)

        write_pointer = 0
        # for i,(bin,bin_count) in enumerate(zip(radial_bins,radial_bin_counts)):
        for i in range(number_of_bins):
            if i not in unique_bins: #i.e. there are no halo pairs in this radial bin
                continue

            bin_count = radial_bin_counts[unique_bins == i][0] #some low radius bins may not have any pairs so this is correct indexing
            if bin_count <= num_per_bin: 
                bin_mask = (bin_idx == i)
                dset_radial[write_pointer:write_pointer + bin_count] = radii[bin_mask]
                dset_velocities[write_pointer:write_pointer + bin_count] = vels[bin_mask]
                dset_sec_masses[write_pointer:write_pointer + bin_count] = sec_masses[bin_mask]
                dset_prim_masses[write_pointer:write_pointer + bin_count] = prim_masses[bin_mask]

                write_pointer += bin_count
                continue
            
            #if the above catches aren't activated we have to subsample
            bin_mask = bin_idx == i
            radii_in_bin = radii[bin_mask]
            velocities_in_bin = vels[bin_mask]
            sec_masses_in_bin = sec_masses[bin_mask]
            prim_masses_in_bin = prim_masses[bin_mask]

            # WE LOSE ORDERING! is this bad? not if we lose the ordering in the same way across the dsets
            subsample_idx = np.random.choice(radii_in_bin.shape[0], num_per_bin, replace = False)
            dset_radial[write_pointer:write_pointer + num_per_bin] = radii_in_bin[subsample_idx]
            dset_velocities[write_pointer:write_pointer + num_per_bin] = velocities_in_bin[subsample_idx]
            dset_sec_masses[write_pointer:write_pointer + num_per_bin] = sec_masses_in_bin[subsample_idx]
            dset_prim_masses[write_pointer:write_pointer + num_per_bin] = prim_masses_in_bin[subsample_idx]

            write_pointer += num_per_bin

#NOTE this function is only for subsampling non-written data, so the input is the datasets created in TWOHALO.py 
# and before writing they're altered (requires reshaping)
def subsample_data(radii,vels,prim_masses,sec_masses, number_of_bins = 20, max_radius = 300, num_per_bin = int(1e7), overwrite = False):
    # num_per_bin is how many pairs we want to keep per radial bin
    # max_radius is the radial distance cut-off in Mpc

    print('data loaded in')

    radial_bins = np.logspace(0, np.log10(max_radius), number_of_bins)
    bin_idx =  np.digitize(radii, radial_bins) - 1 # -1 to correct index offset i.e. it will index to the lower bound of the bin

    unique_bins, radial_bin_counts = np.unique(bin_idx, return_counts = True)

   # How many do we keep in the new dataset; we set a per radial bin ceiling
    dset_shape = radial_bin_counts[radial_bin_counts <= num_per_bin].sum() +\
                 radial_bin_counts[radial_bin_counts > num_per_bin].shape[0] * num_per_bin
    
    subsampled_radii = np.zeros(dset_shape)
    subsampled_vels = np.zeros(dset_shape)
    subsampled_prim_masses = np.zeros(dset_shape)
    subsampled_sec_masses = np.zeros(dset_shape)

    write_pointer = 0
    for i in range(number_of_bins):
        if i not in unique_bins: #i.e. there are no halo pairs in this radial bin
            continue

        bin_count = radial_bin_counts[unique_bins == i][0] #some low radius bins may not have any pairs so this is correct indexing
        if bin_count <= num_per_bin: 
            bin_mask = (bin_idx == i)
            subsampled_radii[write_pointer:write_pointer + bin_count] = radii[bin_mask]
            subsampled_vels[write_pointer:write_pointer + bin_count] = vels[bin_mask]
            subsampled_sec_masses[write_pointer:write_pointer + bin_count] = sec_masses[bin_mask]
            subsampled_prim_masses[write_pointer:write_pointer + bin_count] = prim_masses[bin_mask]

            write_pointer += bin_count
            continue
        
        #if the above catches aren't activated we have to subsample
        bin_mask = bin_idx == i
        radii_in_bin = radii[bin_mask]
        velocities_in_bin = vels[bin_mask]
        sec_masses_in_bin = sec_masses[bin_mask]
        prim_masses_in_bin = prim_masses[bin_mask]

        # WE LOSE ORDERING! is this bad? not if we lose the ordering in the same way across the dsets
        subsample_idx = np.random.choice(radii_in_bin.shape[0], num_per_bin, replace = False)
        subsampled_radii[write_pointer:write_pointer + num_per_bin] = radii_in_bin[subsample_idx]
        subsampled_vels[write_pointer:write_pointer + num_per_bin] = velocities_in_bin[subsample_idx]
        subsampled_sec_masses[write_pointer:write_pointer + num_per_bin] = sec_masses_in_bin[subsample_idx]
        subsampled_prim_masses[write_pointer:write_pointer + num_per_bin] = prim_masses_in_bin[subsample_idx]

        write_pointer += num_per_bin

    #here we have to resize which is a little slow
    radii.resize(dset_shape)
    vels.resize(dset_shape)
    prim_masses.resize(dset_shape)
    sec_masses.resize(dset_shape)

    radii[:dset_shape] = subsampled_radii
    vels[:dset_shape] = subsampled_vels
    prim_masses[:dset_shape] = subsampled_prim_masses
    sec_masses[:dset_shape] = subsampled_sec_masses

if __name__ == '__main__':
    pair_data_path = '/disks/cosmodm/vdvuurst/data/M12-15.5_0.5dex/velocity_data_M1_13.5-14.0_M2_14.5-15.0.hdf5'
    subsample_existing_data(pair_data_path, overwrite = False)