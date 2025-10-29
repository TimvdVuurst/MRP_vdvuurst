import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import skew, kurtosis
import os
from tqdm import tqdm

class Plotter:
    def __init__(self, path: str):
        with h5py.File(path,'r') as file:
            self.radial_distances = file['radial_distances'][:]
            self.velocities = file['velocity_differences'][:]
            self.prim_masses = file['primary_masses'][:]
            self.sec_masses = file['secondary_masses'][:]
        
        self.is_subsample = 'subsample' in path
        info = path.split('_')
        self.mass_range_primary = np.array(info[-3].split('-')).astype(np.float32)
        self.mass_range_secondary = np.array(info[-1].split('.hdf5')[0].split('-')).astype(np.float32)

    # watch out that the max_radius and number_of_bins should be equal to those set in the subsampling
    def plot_velocity_histograms(self, max_radius = 300, number_of_bins = 20, savefig = True, showfig = False):
        # Define radial bins
        self.radial_bins = np.logspace(0, np.log10(max_radius), number_of_bins)

        bin_indices = np.digitize(self.radial_distances, bins = self.radial_bins) - 1
        num_bins = len(self.radial_bins) - 1  

        num_cols = 3
        num_rows = int(np.ceil(num_bins / num_cols))  

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
        axes = axes.flatten()  

        self.mean, self.dispersion, self.skews, self.kurt = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)

        for bin_idx in range(num_bins):
            bin_mask = bin_indices == bin_idx
            bin_velocities = self.velocities[bin_mask]

            ax = axes[bin_idx]
            ax.set_xlabel("Velocity Difference ")
            ax.set_ylabel("Count")
            ax.set_title(f"{self.radial_bins[bin_idx]:.2f} - {self.radial_bins[bin_idx + 1]:.2f} Mpc")

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

        # plt.suptitle(f'M1: {self.mass_range_primary[0]}-{self.mass_range_primary}[1], M2: {self.mass_range_secondary[0]}-self.mass')
        plt.tight_layout()
        if savefig:
            # TODO: find better filename structure
            if self.is_subsample:
                filename = f"/disks/cosmodm/vdvuurst/figures/vel_hist_2halo_subsampled_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            else:
                filename = f"/disks/cosmodm/vdvuurst/figures/full_data/vel_hist_2halo_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            plt.savefig(filename, dpi=300,bbox_inches = 'tight')
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

        for ax in axes:
            ax.grid()

        plt.subplots_adjust(wspace = 0, hspace=0) #NOTE: this might not actually do anything lol
        if savefig:
            if self.is_subsample:
                filename = f"/disks/cosmodm/vdvuurst/figures/moments_2halo_subsample_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            else:
                filename = f"/disks/cosmodm/vdvuurst/figures/moments_2halo_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            plt.savefig(filename, dpi = 200, bbox_inches = 'tight')
        if showfig:
            plt.show()
        
        plt.close()


if __name__ == '__main__':
    dir = '/disks/cosmodm/vdvuurst/data/M12-15.5_0.5dex_subsampled'
    # datapath = '/disks/cosmodm/vdvuurst/data/M12-15.5_0.5dex/velocity_data_M1_12.5-13.0_M2_13.0-13.5.hdf5'
    
    for file in tqdm(sorted(os.listdir(dir))[1:]):
        datapath = os.path.join(dir,file)
        plotter = Plotter(datapath)
        plotter.plot_velocity_histograms()
        plotter.plot_moments()