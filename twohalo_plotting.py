from scipy.stats import skew, kurtosis
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
import argparse
from functions import *
from json import load

class TwoHaloPlotter:
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

        for bin_idx in range(num_bins):
            bin_mask = bin_indices == bin_idx
            bin_velocities = self.velocities[bin_mask]

            ax = axes[bin_idx]
            ax.set_xlabel("Velocity Difference ")
            ax.set_ylabel("Count")
            ax.set_title(f"{self.radial_bins[bin_idx]:.2f} - {self.radial_bins[bin_idx + 1]:.2f} Mpc")

            if len(bin_velocities) == 0:
                continue

            ax.hist(bin_velocities, bins=50, alpha=0.75, color='b', edgecolor='black')
            # ax.ticklabel_format(useOffset=False, style='plain') # no scientific notation

        # No clue what this does yet
        for i in range(num_bins, len(axes)):
            fig.delaxes(axes[i])

        # plt.suptitle(f'M1: {self.mass_range_primary[0]}-{self.mass_range_primary}[1], M2: {self.mass_range_secondary[0]}-self.mass')
        plt.tight_layout()
        if savefig:
            # TODO: find better filename structure
            if self.is_subsample:
                filename = f"/disks/cosmodm/vdvuurst/figures/vel_hists/vel_hist_2halo_subsampled_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            else:
                filename = f"/disks/cosmodm/vdvuurst/figures/vel_hists/vel_hist_2halo_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            plt.savefig(filename, dpi=300,bbox_inches = 'tight')
        if showfig:
            plt.show()
        
        plt.close()

    @staticmethod
    #from: https://en.wikipedia.org/wiki/Skewness#Sample_skewness
    def SES(n):
        return np.sqrt(6*n *(n-1) / ((n-2)*(n+1)*(n+3)))
    
    @staticmethod
    #From: https://en.wikipedia.org/wiki/Kurtosis
    def SEK(n, SES):
        return np.sqrt(24*n*np.square(n-1) / ((n-3) * (n-2) * (n+3) * (n+5)))


    def plot_moments(self, max_radius = 300, number_of_bins = 20, savefig = True, showfig = False, filename = None):
        self.radial_bins = np.logspace(0, np.log10(max_radius), number_of_bins)

        bin_indices = np.digitize(self.radial_distances, bins = self.radial_bins) - 1
        num_bins = len(self.radial_bins) - 1  

        self.mean, self.dispersion, self.skews, self.kurt = np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins), np.zeros(num_bins)
        self.skew_error = np.zeros(num_bins)
        self.kurt_error = np.zeros(num_bins)
        
        fig,axes = plt.subplots(nrows = 4,figsize=(8,12), sharex=True)

        for bin_idx in range(num_bins):
            bin_mask = bin_indices == bin_idx
            bin_velocities = self.velocities[bin_mask]
            N = bin_velocities.size
            # print(f'LOOK HERE!!!! {N}')
            if N == 0:
                continue

            self.mean[bin_idx] = np.mean(bin_velocities)
            self.dispersion[bin_idx] = np.std(bin_velocities)
            self.skews[bin_idx] = skew(bin_velocities, bias = False)
            self.kurt[bin_idx] = kurtosis(bin_velocities, fisher = False, bias = False)

            # Standard error for skewness and kurtosis
            if N >= 4:
                self.skew_error[bin_idx] = self.SES(N)
                self.kurt_error[bin_idx] = self.SEK(N, self.skew_error[bin_idx])

        axes[0].plot(self.radial_bins[:-1], self.mean, marker='s', c = 'black')
        axes[0].set(ylabel = r'Mean $\mu$')

        axes[1].plot(self.radial_bins[:-1], self.dispersion, marker = 's', c = 'forestgreen')
        axes[1].set(ylabel = r'Dispersion $\sigma$')

        axes[2].plot(self.radial_bins[:-1], self.skews, marker = 's', c= 'red')
        nonzero_error = np.nonzero(self.skew_error)
        axes[2].errorbar(self.radial_bins[:-1][nonzero_error], self.skews[nonzero_error], yerr = self.skew_error[nonzero_error], fmt =',', capsize=3, color = 'red')
        axes[2].set(ylabel = r'Skewness $s$')

        axes[3].plot(self.radial_bins[:-1], self.kurt, marker='s', c = 'blue')
        nonzero_error = np.nonzero(self.kurt_error)
        axes[3].errorbar(self.radial_bins[:-1][nonzero_error], self.kurt[nonzero_error], yerr = self.kurt_error[nonzero_error], fmt =',', capsize=3, color = 'blue')
        axes[3].set(xlabel = r'radial distance $r$', ylabel = r'Kurtosis $k$')

        for ax in axes:
            ax.grid()

        plt.subplots_adjust(wspace = 0, hspace=0) #NOTE: this might not actually do anything lol
        if savefig:
            # if self.is_subsample:
            #     filename = f"/disks/cosmodm/vdvuurst/figures/moments/moments_2halo_subsample_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            # else:
            if filename is None:
                filename = f"/disks/cosmodm/vdvuurst/figures/moments/moments_2halo_M1_{self.mass_range_primary[0]}-{self.mass_range_primary[1]}_M2_{self.mass_range_secondary[0]}-{self.mass_range_secondary[1]}"+".png"
            plt.savefig(filename, dpi = 200, bbox_inches = 'tight')
        if showfig:
            plt.show()
        
        plt.close()


