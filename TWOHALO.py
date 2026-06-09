import numpy as np
import h5py
from typing import Tuple
import argparse
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from supplementary.subsample import *
from MCMC import MCMC
from functions import *
from inspect import signature
from json import dump, load

def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32) -> np.ndarray:
    return (10**m_min <= mass) & (mass <= 10**m_max) # base 10 since FOF masses aren't in units of 10^10 Msol

def _make_nbound_mask(bound: np.ndarray, N_min: np.float32):
    return bound >= N_min

def _make_radial_mask(coordinate_differences: np.ndarray, cutoff: np.float32 = 300.):
    # Make a mask that is true only where not a signal coordinate difference is larger than cutoff, set to 300 Mpc by default.
    return (np.sum(coordinate_differences >= cutoff, axis = 1) == 0)

class TWOHALO:
    def __init__(self, PATH: str, filename: str):
        self.PATH = PATH
        self.filename = filename

        with h5py.File(PATH, "r") as handle:
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
            self.SOMass = handle['SO/200_mean/TotalMass'][:] 
            self.NoofBoundParticles = handle["InputHalos/NumberOfBoundParticles"][:]
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
            self.IsCentral = handle["InputHalos/IsCentral"][:].astype(bool) #set to bool so it can be used as a mask
            self.boxsize = handle['Header'].attrs['BoxSize'][0]
        
        self.half_boxsize = self.boxsize / 2
        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

    def create_full_catalogue(self, mass_range_primary: Tuple[np.float32,np.float32],
                          mass_range_secondary: Tuple[np.float32,np.float32],
                          N_bound: int = 100, only_centrals: bool = True):
        """_summary_

        Args:
            mass_range_primary (tuple): Must be in order (MIN,MAX)
            mass_range_secondary (tuple): Must be in order (MIN,MAX)
            N_bound (int, optional): _description_. Defaults to 100.
            only_centrals (bool, optional): _description_. Defaults to True.
        """
        self.mass_range_primary = mass_range_primary
        self.mass_range_secondary = mass_range_secondary

        bound_mask = _make_nbound_mask(self.NoofBoundParticles, N_bound)
        mass_mask = _make_mass_mask(self.SOMass, *mass_range_primary) # mass bin of the primaries
        central_selection = self.IsCentral if only_centrals else np.ones_like(bound_mask).astype(bool) # pick out the centrals if so desired
        
        # final mask for the primaries, select primaries
        primary_mask = bound_mask & mass_mask & central_selection
        primary_selection_size = np.sum(primary_mask)
        primary_pos = self.COM[primary_mask] 
        primary_vel = self.COMvelocity[primary_mask]
        primary_mass = self.SOMass[primary_mask] # NOT USED
        primary_ID = self.HaloCatalogueIndex[primary_mask] 

        # selection of the secondaries, differs only in mass range
        secondary_mass_selection = _make_mass_mask(self.SOMass, *mass_range_secondary) & bound_mask & central_selection
        secondary_selection_size = secondary_mass_selection.sum()
        secondary_pos = self.COM[secondary_mass_selection]
        secondary_vel = self.COMvelocity[secondary_mass_selection]
        secondary_mass = self.SOMass[secondary_mass_selection]
        secondary_ID = self.HaloCatalogueIndex[secondary_mass_selection]

        intersection_length = np.intersect1d(primary_ID, secondary_ID).shape[0] 
        primaries_are_subset = (intersection_length == primary_selection_size)

        # TODO: Think of a better naming convention?
        print(f'\nNow working on {self.PATH}, writing to {self.filename}...\n')

        # When avoiding self-comparison for primaries that might be a (partial) subset of the secondaries this generally holds
        dset_shape = (secondary_selection_size - intersection_length) * (primary_selection_size - intersection_length) + \
                     (intersection_length * (secondary_selection_size - 1))

        # Keeping to a for loop since array manipulation would be too memory intensive and thus slower (tested)
        # Save everything in memory to mitigate I/O usage throughout
        array_radial = np.zeros(dset_shape, dtype = np.float32)
        array_velocities = np.zeros(dset_shape, dtype = np.float32)
        array_prim_masses = np.zeros(dset_shape, dtype = np.float32)
        array_sec_masses = np.zeros(dset_shape, dtype = np.float32)

        counter = 0
        number_of_comparisons = secondary_selection_size - 1 # Holds if primaries are subset
        for pos1, vel1,mass1, id1 in tqdm(zip(primary_pos, primary_vel,primary_mass, primary_ID), total = len(primary_pos)):
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

            array_radial[counter:counter+number_of_comparisons] = radial_distances
            array_velocities[counter:counter+number_of_comparisons] = projected_vels
            array_sec_masses[counter:counter+number_of_comparisons] = secondary_mass[self_compare_mask]
            array_prim_masses[counter:counter+number_of_comparisons] = np.full(number_of_comparisons, mass1) #fill with the same value to keep the same dset shape

            counter += number_of_comparisons

        # make sure we don't save meaningless zeroes at the end
        nonzero_idx = np.nonzero(array_radial) #holds across all arrays so we need only calculate it once
        array_radial = array_radial[nonzero_idx]
        array_velocities = array_velocities[nonzero_idx]
        array_sec_masses = array_sec_masses[nonzero_idx]
        array_prim_masses = array_prim_masses[nonzero_idx]

        # This is the only IO
        with h5py.File(self.filename, "w") as f:
            _ = f.create_dataset("radial_distances", data = array_radial, dtype=np.float32)
            _ = f.create_dataset("velocity_differences", data = array_velocities, dtype=np.float32)
            _ = f.create_dataset("secondary_masses", data = array_sec_masses, dtype=np.float32)
            _ = f.create_dataset("primary_masses", data = array_prim_masses, dtype=np.float32)


    def create_subsampled_catalogue(self, mass_range_primary: Tuple[np.float32,np.float32],
                          mass_range_secondary: Tuple[np.float32,np.float32],
                          radial_cutoff: np.float32 = 300., #Mpc
                          N_bound: int = 100, only_centrals: bool = True):
        """_summary_

        Args:
            mass_range_primary (tuple): Must be in order (MIN,MAX)
            mass_range_secondary (tuple): Must be in order (MIN,MAX)
            radial_cutoff (np.float32): maximum distance along one cartesian coordinate for selection
            N_bound (int, optional): _description_. Defaults to 100.
            only_centrals (bool, optional): _description_. Defaults to True.
        """
        self.mass_range_primary = mass_range_primary
        self.mass_range_secondary = mass_range_secondary
        self.radial_cutoff = radial_cutoff
        # self.radial_bins = np.logspace(0, np.log10(radial_cutoff), bins = 20)

        bound_mask = _make_nbound_mask(self.NoofBoundParticles, N_bound)
        mass_mask = _make_mass_mask(self.SOMass, *mass_range_primary) # mass bin of the primaries
        central_selection = self.IsCentral if only_centrals else np.ones_like(bound_mask).astype(bool) # pick out the centrals if so desired
        
        # final mask for the primaries, select primaries
        primary_mask = bound_mask & mass_mask & central_selection
        primary_selection_size = np.sum(primary_mask)
        primary_pos = self.COM[primary_mask] 
        primary_vel = self.COMvelocity[primary_mask]
        primary_mass = self.SOMass[primary_mask] # NOT USED
        primary_ID = self.HaloCatalogueIndex[primary_mask] 

        # selection of the secondaries, differs only in mass range from the primaries
        secondary_mass_selection = _make_mass_mask(self.SOMass, *mass_range_secondary) & bound_mask & central_selection
        secondary_selection_size = secondary_mass_selection.sum()
        secondary_pos = self.COM[secondary_mass_selection]
        secondary_vel = self.COMvelocity[secondary_mass_selection]
        secondary_mass = self.SOMass[secondary_mass_selection]
        secondary_ID = self.HaloCatalogueIndex[secondary_mass_selection]

        intersection_length = np.intersect1d(primary_ID, secondary_ID).size

        # print(f'\nNow working on {self.PATH}, writing to {self.filename}...\n')

        # When avoiding self-comparison for primaries that might be a (partial) subset of the secondaries this generally holds
        dset_shape_full = (secondary_selection_size - intersection_length) * (primary_selection_size - intersection_length) + \
                     (intersection_length * (secondary_selection_size - 1))

        # both radial distances and velocities are projected so that they're 1D

        # Save everything in memory to mitigate I/O usage throughout
        array_radial = np.zeros(dset_shape_full//20, dtype = np.float32)
        array_velocities = np.zeros(dset_shape_full//20, dtype = np.float32)
        array_prim_masses = np.zeros(dset_shape_full//20, dtype = np.float32)
        array_sec_masses = np.zeros(dset_shape_full//20, dtype = np.float32)

        counter = 0
        for pos1, vel1,mass1, id1 in tqdm(zip(primary_pos, primary_vel,primary_mass, primary_ID), total = len(primary_pos)):
            self_compare_mask = secondary_ID != id1 # Exclude self-comparison

            # Positional differences with periodic boundary conditions, without self-comparison
            pos_diffs = (secondary_pos[self_compare_mask] - pos1 + self.half_boxsize) % self.boxsize - self.half_boxsize
            radial_distances = np.linalg.norm(pos_diffs, axis=1) 

            #SUBSAMPLING!
            #TODO: make rmin (and rmax?) parameters that can be controlled in this functionality.
            radial_distances, selected_indx = rejection_sample(radial_distances, dropoff = power_dropoff, rmin = 20.13, rmax = 300) 

            pos_diffs = pos_diffs[selected_indx]
            number_of_comparisons = selected_indx.size

            # Project to line connecting the haloes
            radial_unit_vectors = pos_diffs / radial_distances[:, np.newaxis]

            # Project velocities to the connecting line between haloes, use all relevant selections
            vel_diffs = secondary_vel[self_compare_mask][selected_indx] - vel1
            projected_vels = np.einsum('ij,ij->i', vel_diffs, radial_unit_vectors)

            array_radial[counter:counter+number_of_comparisons] = radial_distances
            array_velocities[counter:counter+number_of_comparisons] = projected_vels
            array_sec_masses[counter:counter+number_of_comparisons] = secondary_mass[self_compare_mask][selected_indx]
            array_prim_masses[counter:counter+number_of_comparisons] = np.full(number_of_comparisons, mass1) #fill with the same value to keep the same dset shape

            counter += number_of_comparisons

        # make sure we don't save meaningless zeroes at the end
        nonzero_idx = np.nonzero(array_radial) #holds across all arrays so we need only calculate it once
        array_radial = array_radial[nonzero_idx]
        array_velocities = array_velocities[nonzero_idx]
        array_sec_masses = array_sec_masses[nonzero_idx]
        array_prim_masses = array_prim_masses[nonzero_idx]

        # This is the only
        with h5py.File(self.filename, "w") as f:
            _ = f.create_dataset("radial_distances", data = array_radial, dtype=np.float32)
            _ = f.create_dataset("velocity_differences", data = array_velocities, dtype=np.float32)
            _ = f.create_dataset("secondary_masses", data = array_sec_masses, dtype=np.float32)
            _ = f.create_dataset("primary_masses", data = array_prim_masses, dtype=np.float32)


    def plot_velocity_histograms(self, savefig = True, showfig = False):
        # Define radial bins
        # TODO: verify this choice, does this need to be modifiable?
        self.radial_bins = np.logspace(0.5, 2.7, 20).astype(np.float32)

        with h5py.File(self.filename, "r") as f:
            radial_distances = f["radial_distances"][:]
            velocities = f["velocity_differences"][:]
            
        bin_indices = np.digitize(radial_distances, bins = self.radial_bins) - 1
        num_bins = len(self.radial_bins) - 1  

        num_cols = 3
        num_rows = int(np.ceil(num_bins / num_cols))  

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
        axes = axes.flatten()  

        self.mean, self.dispersion, self.skews, self.kurt = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)

        for bin_idx in range(num_bins):
            bin_mask = bin_indices == bin_idx
            bin_velocities = velocities[bin_mask]

            ax = axes[bin_idx]
            ax.set_xlabel("Velocity Difference ")
            ax.set_ylabel("Count")
            ax.set_title(f"Bin {bin_idx + 1}: {self.radial_bins[bin_idx]:.2f} - {self.radial_bins[bin_idx + 1]:.2f} Mpc")

            if len(bin_velocities) == 0:
                continue

            self.mean[bin_idx] = np.mean(bin_velocities)
            self.dispersion[bin_idx] = np.std(bin_velocities)
            self.skews[bin_idx] = skew(bin_velocities, bias = False)
            self.kurt[bin_idx] = kurtosis(bin_velocities, fisher = False, bias = False)

            ax.hist(bin_velocities, bins=50, alpha=0.75, color='b', edgecolor='black')

        # No clue what this does yet
        for i in range(num_bins, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if savefig:
            # TODO: find better filename structure
            filename = f"/disks/cosmodm/vdvuurst/figures/vel_hist_2halo_M1_1{self.mass_range_primary[0]}-1{self.mass_range_primary[1]}_M2_1{self.mass_range_secondary[0]}-1{self.mass_range_secondary[1]}"+".png"
            plt.savefig(filename, dpi=200,bbox_inches = 'tight')
        if showfig:
            plt.show()
        
        plt.close()
    
    def plot_moments(self, savefig = True, showfig = False):
        fig,axes = plt.subplots(nrows = 4,figsize=(8,12), sharex=True)

        axes[0].plot(self.radial_bins[:-1], self.mean, marker='s')
        axes[0].set(ylabel = r'Mean $\mu$')
        axes[1].plot(self.radial_bins[:-1], self.dispersion, marker = 's')
        axes[1].set(ylabel = r'Dispersion $\sigma$')
        axes[2].plot(self.radial_bins[:-1], self.skews, marker = 's')
        axes[2].set(ylabel = r'Skewness $s$')
        axes[3].plot(self.radial_bins[:-1], self.kurt, marker='s')
        axes[3].set(xlabel = r'radial distance $r$', ylabel = r'Kurtosis $k$')

        plt.subplots_adjust(wspace = 0, hspace=0) #NOTE: this might not actually do anything lol
        if savefig:
            filename = f"/disks/cosmodm/vdvuurst/figures/moments_2halo_M1_1{self.mass_range_primary[0]}-1{self.mass_range_primary[1]}_M2_1{self.mass_range_secondary[0]}-1{self.mass_range_secondary[1]}"+".png"
            plt.savefig(filename, dpi = 200, bbox_inches = 'tight')
        if showfig:
            plt.show()
        
        plt.close()


class TWOHALO_fitter:
    def __init__(self, PATH = '/disks/cosmodm/vdvuurst/data/M12-15.5_0.5dex/velocity_data_M1_13.0-13.5_M2_13.5-14.0.hdf5'):
        self.massbin = PATH.split('velocity_data_')[-1].split('.hdf5')[0]

        with h5py.File(PATH,'r') as file:
            self.radial_distances = file['radial_distances'][:]
            self.velocities = file['velocity_differences'][:]
            self.prim_masses = file['primary_masses'][:]
            self.sec_masses = file['secondary_masses'][:]
        
        print('DATA LOADED')

        self.max_radius = 300. #Mpc
        self.number_of_bins = 20 #hardcoded, should make input

        self.radial_bins = np.logspace(0, np.log10(self.max_radius), self.number_of_bins)
        self.bin_indices = np.digitize(self.radial_distances, bins = self.radial_bins) - 1


    def run_two_halo_emcee(self, bin_idx, likelihood_func = skew_gaussian_log_likelihood, func = skewnorm_func, nwalkers = 20, nsteps = 500):
        bin_mask = self.bin_indices == bin_idx
        bin_velocities = self.velocities[bin_mask]

        ndim = len(signature(func).parameters) - 1

        if func is skewnorm_func:
            pos = np.random.uniform(low = [-2.,-250., 100.,], high = [2., 0., 2000.], size = (nwalkers, ndim)).T
            step_sizes = np.array([0.05, 2.5, 100.])[:, np.newaxis]

        elif func is skew_t_pdf:
            pos = np.random.uniform(low = [-5., -400, 150, 2.], high = [0, 400, 350, 8], size = (nwalkers, ndim)).T
            step_sizes = np.array([0.25, 25., 5., 0.1])[:, np.newaxis]
            
        sampler = MCMC(nwalkers, likelihood_func , args = (bin_velocities,), step_sizes= step_sizes)
        sampler.run_mcmc(pos, nsteps, verbose = False)

        best_params, best_likelihood, title = self.refine_and_plots_twohalo(bin_velocities, bin_idx, sampler, ndim, func, likelihood_func)

        datapath = f'/disks/cosmodm/vdvuurst/data/TwoHalo_param_fits/{title}/'
        
        mkdir_if_non_existent(datapath)
        param_dict = {'parameters':list(best_params), 'likelihood':float(best_likelihood), 'nsteps':nsteps, 'nwalkers':nwalkers, 'function':title}
        with open(os.path.join(datapath, f'{self.massbin}-rbin{bin_idx}.json'), 'w') as f:
            dump(param_dict, f, indent = 1)
        
        # plot both distributions together if possible
        try:
            self.plot_dist_from_result(os.path.join(datapath, f'{self.massbin}-rbin{bin_idx}.json'), bin_idx)
        except FileNotFoundError as e:
            print(f'Could not plot both distributions because of the following error:\n{e}')

    def refine_and_plots_twohalo(self, bin_velocities, bin_idx, sampler, ndim, func = skewnorm_func, ll_func = skew_gaussian_log_likelihood):
        title = 'Skew-normal' if func is skewnorm_func else 'Skew-t'
        fig, axes = plt.subplots(ndim + 1, figsize=(12, 10), sharex=True)
        axes[0].set_title(title)
        samples = sampler.get_chain()

        burnin = 0
        if func is skewnorm_func:
            param_labels = [r'$\alpha$',r'$\mu$', r'$\sigma$']
        else:
            param_labels = [r'$\alpha$', r'$\xi$', r'$\omega$', r'$\nu$']
        for i in range(ndim):
            ax:plt.Axes = axes[i]
            ax.plot(samples[i,...].T, alpha=0.3)
            ax.set_xlim(0, samples.shape[-1])
            ax.set_ylabel(param_labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

            ymin, ymax = ax.get_ylim()
            ax.vlines(burnin, ymin, ymax, colors = 'black', linestyles = '--')
            ax.set_ylim(ymin, ymax)

        likelihoods = sampler.get_likelihoods()
        likelihoods_plot = np.log10(likelihoods).T
        likelihoods_plot[:burnin] = np.nan # so that the lines are better discernable in the plot
        
        axes[-1].plot(likelihoods_plot, alpha = 0.3)
        axes[-1].set(ylabel = r'$\log\left(-\log(\mathcal{L})\right)$')
        axes[-1].set_xlabel("Step number")
        ymin, ymax = axes[-1].get_ylim()
        axes[-1].vlines(burnin, ymin, ymax, colors = 'black', linestyles = '--')
        axes[-1].set_ylim(ymin, ymax)
        fig.tight_layout()
        # subpath = f/TWOHALO/{title}/'
        plt.savefig(os.path.join('/disks/cosmodm/vdvuurst/figures/TWOHALO', title, f'{self.massbin}-rbin{bin_idx}_walkers.png'), dpi = 300)
        plt.close()

        # Refining
        best_arg = np.unravel_index(np.argmin(likelihoods), likelihoods.shape)
        best_params = np.array([samples[i,*best_arg] for i in range(ndim)])
        # print('Performing BFGS refinement')
        best_params = minimize(ll_func, best_params, args = (bin_velocities)).x
        best_likelihood = ll_func(best_params, bin_velocities)

        bin_mask = self.bin_indices == bin_idx
        bin_velocities = self.velocities[bin_mask]

        plt.figure()
        plt.title(f'{self.radial_bins[bin_idx]:.2f} - {self.radial_bins[bin_idx + 1]:.2f} Mpc ({title})')

        bins = rice_bins(bin_velocities.size)

        bin_heights, bin_edges = np.histogram(bin_velocities, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width= bin_edges[1] - bin_edges[0] 
        bin_widths = np.diff(bin_edges)  # The width of each bin
        number_density = bin_heights / bin_widths  # Normalize by bin width

        plt.bar(bin_centers, number_density, width=bin_width, align='center', edgecolor = 'black',
                alpha=0.75, color='b', label = "N" + r'$_\mathrm{b}$' + f" = {rice_bins(bin_velocities.size)}")

        DAT=np.linspace(np.min(bin_velocities),np.max(bin_velocities),1000)
        hist_area=np.sum(bin_heights)
        plt.plot(DAT, hist_area * func(DAT, *best_params), c='black')
        plt.legend()

        plt.xlabel(r'Two-halo velocity $v$ [km/s]')
        plt.ylabel('Density')

        plt.tight_layout()
        plt.savefig(os.path.join('/disks/cosmodm/vdvuurst/figures/TWOHALO', title, f'{self.massbin}-rbin{bin_idx}_fit.png'), dpi = 300)
        plt.close()

        return best_params, best_likelihood, title


    def plot_dist_from_result(self, param_dict_path, bin_idx):
        if 'Skew-t' in param_dict_path:
            with open(param_dict_path, 'r') as f:
                param_dict_skew_t = load(f)
            
            param_dict_path = param_dict_path.replace('Skew-t', 'Skew-normal')
            with open(param_dict_path, 'r') as f:
                param_dict_skew_norm = load(f)
        else:
            with open(param_dict_path, 'r') as f:
                param_dict_skew_norm = load(f)
            
            param_dict_path = param_dict_path.replace('Skew-normal', 'Skew-t')
            with open(param_dict_path, 'r') as f:
                param_dict_skew_t = load(f)

        best_params_skew_t = np.asarray(param_dict_skew_t['parameters'])
        best_params_skew_norm = np.asarray(param_dict_skew_norm['parameters'])

        bin_mask = self.bin_indices == bin_idx
        bin_velocities = self.velocities[bin_mask]

        plt.figure()
        splitstring = self.massbin.split('_')
        firsthalf = splitstring[0].replace('M1',r'$M_{\mathrm{2h,p}}$') + ' = ' + splitstring[1] + ' dex'
        sechalf = splitstring[2].replace('M2','$M_{\mathrm{2h,s}}$') + ' = ' + splitstring[3] + ' dex'
        plt.title(f'{firsthalf}, {sechalf}' + '\n' + r'$r_{\mathrm{2h}}$'+ f' = {self.radial_bins[bin_idx]:.2f} - {self.radial_bins[bin_idx + 1]:.2f} Mpc')

        bins = rice_bins(bin_velocities.size)

        bin_heights, bin_edges = np.histogram(bin_velocities, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width= bin_edges[1] - bin_edges[0] 
        bin_widths = np.diff(bin_edges)  # The width of each bin
        number_density = bin_heights / bin_widths  # Normalize by bin width

        plt.bar(bin_centers, number_density, width=bin_width, align='center', edgecolor = 'black',
                alpha=1, color='b', label = "N" + r'$_\mathrm{b}$' + f" = {rice_bins(bin_velocities.size)}")

        DAT=np.linspace(np.min(bin_velocities),np.max(bin_velocities),1000)
        hist_area=np.sum(bin_heights)

        plt.plot(DAT, hist_area * skewnorm_func(DAT, *best_params_skew_norm), c='red', label = r'Skew normal-distribution', lw = 2)
        plt.plot(DAT, hist_area * skew_t_pdf(DAT, *best_params_skew_t), c='forestgreen', label = r'Skew $t$-distribution', lw = 2)
        plt.legend()

        plt.xlabel(r'Two-halo velocity $v_{\mathrm{2h}}$ [km/s]')
        plt.ylabel('Density')

        plt.tight_layout()
        mkdir_if_non_existent('/disks/cosmodm/vdvuurst/figures/TWOHALO/Combined')
        plt.savefig(os.path.join('/disks/cosmodm/vdvuurst/figures/TWOHALO/Combined',
                                  f'{self.massbin}-rbin{bin_idx}_fit.png'), dpi = 300)
        plt.close()




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P','--path', type = str,
                         default = "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0077.hdf5")
    parser.add_argument("-M1","--mass_range_primary", type = float, nargs = '+', default=[3.5,4])
    parser.add_argument("-M2","--mass_range_secondary", type = float, nargs = '+', default=[4.5,5])
    parser.add_argument('-F','--make_figures', type = bool, default = False)
    parser.add_argument('-SF','--show_figures', type = bool, default = True)
    # These two probably won't be touched but for the sake of generalizing they're included
    parser.add_argument('-N','--number_of_bound_particles', type = int, default = 100)
    parser.add_argument('-C','--only_centrals', type = bool, default = True)
    
    return parser.parse_args()

if __name__ == '__main__':
    from twohalo_plotting import format_plot

    args = parse_args()

    twohalo = TWOHALO(args.path, '/disks/cosmodm/vdvuurst/data/twohalotest_13.5-14_14.5-15.hdf5')
    twohalo.create_subsampled_catalogue(mass_range_primary = tuple(args.mass_range_primary), 
                             mass_range_secondary = tuple(args.mass_range_secondary),
                             N_bound = args.number_of_bound_particles,
                             only_centrals = args.only_centrals)
    
    if args.make_figures:
        format_plot()
        twohalo.plot_velocity_histograms(savefig = True, showfig = args.show_figures)
        twohalo.plot_moments(savefig = True, showfig = args.show_figures)

    print(f'\n CODE EXITED SUCCESFULLY.')


    
    
