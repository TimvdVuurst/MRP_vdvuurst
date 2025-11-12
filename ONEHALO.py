import numpy as np
import h5py
from typing import Tuple
import argparse
from tqdm import tqdm
from plotting import format_plot
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
import os
from scipy.integrate import quad
from functions import Romberg
from TWOHALO import _make_nbound_mask
from json import load, dump

import emcee
from corner import corner

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

class ONEHALO_fitter:
    def __init__(self, PATH: str, initial_param_file: str = None, joint: bool = False):
        """_summary_

        Args:
            PATH (str): _description_
            joint (bool, optional): Whether to use parameterizations of the three gaussian parameters (9 params in total). 
                                    Defaults to False.
        """
        self.PATH = PATH

        massbin = os.path.split(self.PATH)[-1].split('_')[-1].split('.hdf5')[0].split('-')
        self.lower_mass, self.upper_mass = np.float32(massbin[0]), np.float32(massbin[1])

        if initial_param_file:
            if joint:
                with open(initial_param_file, 'r') as f:
                    self.initial_param_dict = load(f)[f'M_{self.lower_mass}-{self.upper_mass}']
            else:
                with open(initial_param_file, 'r') as f:
                    self.initial_param_dict = load(f)
        else:
            # some random stuff in the ballpark of where we want them to kickstart
            if joint:
                self.initial_param_dict = {"p": -227.7,"n": 0.22,"q": 37.6,"b": 396.4,"m": 7.8,"c": 70.9,"A": 0.69,"B": 20,"C": -0.005}
            else:
                self.initial_param_dict = {'sigma_1':100.0, 'sigma_2': 100.0, 'lambda':0.1} 

        with h5py.File(self.PATH, 'r') as handle:
            self.rel_pos = handle['rel_pos'][:]
            self.rel_vels = handle['rel_vels'][:]
        
        self.rel_sq_dist = np.square(self.rel_pos).sum(axis = 1)

    @staticmethod
    def mod_gaussian(x,sigma1,sigma2,lambda_):
        return ((1-lambda_)* (np.exp(-(x)**2 / (2 * sigma1**2)) /sigma1 ) + \
                lambda_*np.exp(-((x)**2/(2*sigma2**2)))/sigma2)/ (np.sqrt(2*np.pi))
   
    def mod_gaussian_integral(self,sigma1,sigma2,lambda_,x_i,x_f):
        # integral,_= quad(lambda x: self.mod_gaussian(x,sigma1,sigma2,lambda_), x_i, x_f)
        integral, err = Romberg(x_i, x_f, lambda x: self.mod_gaussian(x,sigma1,sigma2,lambda_))
        return integral
        
    def mod_gaussian_log_likelihood_binned(self, params, bin_edges, bin_heights, bin_width):
        sigma1, sigma2, lambda_ = params
        hist_area = np.sum(bin_heights) 
        # I do not know where this comes from, integral of mod_gaussian dv from -inf to inf is 1.  
        # fit_integral = (sigma1 * np.sqrt(2 * np.pi)) *(3*sigma2 + 1 + 105*lambda_) 
        fit_integral = 1.
        A = hist_area / fit_integral
        
        log_L = 0
        for i in range(1,len(bin_edges)):
            f_b = A * self.mod_gaussian_integral(sigma1,sigma2,lambda_,bin_edges[i-1],bin_edges[i])
            
            #penalize negative values and zero
            if f_b <= 0:
                return 10**11
            
            n_b = bin_heights[i-1] * bin_width
            log_L += (n_b * np.log(f_b)) - f_b  # herein lies the only difference with neg_log_likelihood
        
        return log_L
    
    @staticmethod
    def log_prior(theta):
        sigma_1, sigma_2, lambda_ = theta
        if 0.01 < sigma_1 and 0.0001 < sigma_2 and -0.09 < lambda_ < 1.0:   # Standard values taken from Sowmya's code
            return 0.0
        return -np.inf

    def mod_gaussian_log_likelihood(self, params, bin_edges, bin_heights, bin_width): #full with prior
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return self.mod_gaussian_log_likelihood_binned(params, bin_edges, bin_heights, bin_width)
    
    
    @staticmethod
    def _fit_modified_gaussian_emcee(data, bins, initial_guess: dict, log_likelihood_func,
                                     nwalkers = 32, nsteps = 5000, param_labels = [r'$\sigma_1$',r'$\sigma_2$',r'$\lambda$'],
                                     plot = True, verbose = False, filename = 'Mfit', save_params = False):
        
        init_guess, param_names = np.array(list(initial_guess.values())), list(initial_guess.keys()) #param_names not to be confused with param_labels
        ndim = init_guess.shape[0]
        pos = init_guess + 1e-4 * np.random.randn(nwalkers, ndim)

        # Compute histogram
        bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_width = bin_edges[1] - bin_edges[0]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_func, args = (bin_edges, bin_heights, bin_width))
        sampler.run_mcmc(pos, nsteps, progress = verbose)

        if plot:
            # chain plots
            fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
            samples = sampler.get_chain()
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(param_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number")
            # plt.show()
            plt.savefig(f'/disks/cosmodm/vdvuurst/figures/emcee_results/{filename}_walkers.png', dpi = 200)
            plt.close()

            #corner plot
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True) # modify a little but this is probably fine

            fig = corner(flat_samples, labels = param_labels)

            fig.savefig(f'/disks/cosmodm/vdvuurst/figures/emcee_results/{filename}_corner.png', dpi = 200)
            plt.close(fig)
        
        result = np.zeros(ndim)
        errs = np.zeros((3,2))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            err = np.diff(mcmc)
            
            result[i] = mcmc[1]
            errs[i,0] = err[0]
            errs[i,1] = err[1]
        
        if save_params:
            param_dict = dict(zip(param_names, result.tolist()))
            param_dict['errors'] = errs.tolist()
            param_dict['nwalkers'] = nwalkers
            param_dict['nsteps'] = nsteps
            with open(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee/{filename}.json', 'w') as f:
                dump(param_dict, f, indent = 1)

        return result, errs

    def mod_gaussian_neg_log_likelihood_binned(self, params, bin_edges, bin_heights, bin_width):
        sigma1, sigma2, lambda_ = params
        hist_area = np.sum(bin_heights) 
        # I do not know where this comes from, integral of mod_gaussian dv from -inf to inf is 1.  
        # fit_integral = (sigma1 * np.sqrt(2 * np.pi)) *(3*sigma2 + 1 + 105*lambda_) 
        fit_integral = 1.
        A = hist_area / fit_integral
        
        neg_log_L = 0
        for i in range(1,len(bin_edges)):
            f_b = A * self.mod_gaussian_integral(sigma1,sigma2,lambda_,bin_edges[i-1],bin_edges[i])
            
            #penalize negative values and zero
            if f_b <= 0:
                return 10**11
            
            n_b = bin_heights[i-1] * bin_width
            neg_log_L += f_b - (n_b * np.log(f_b))
        
        return neg_log_L
    

    @staticmethod
    def _fit_modified_gaussian_minimize(data, bins, initial_guess,bounds, neg_log_likelihood_func, method = 'emcee',
                              plot_func = None, dist_func = None, distname = 'Modified Gaussian', binno = 1,
                              verbose = False, filename = 'Mfit', save_params = False):
        """
        Fits a modified Gaussian distribution using minimize (scipy.optimize) to binned data and plots the result.

        Input:
        data : Input peculiar velocity difference data to fit.
        bins : [int] Number of histogram bins.
        initial_guess : Initial guess for the parameters.
        bounds :  Bounds for the parameters .
        neg_log_likelihood_func : Negative log_likelihood function to minimize (negative log-likelihood). Must use (bin_edges, bin_heights, bin_width, result.x).
        plot_func : Function to plot the results. Must use (dist_func, params, data, bins, distname, binno).
        dist_func :  optional. Distribution function to be used while plotting.
        distname : [str] optional.  Name of the distribution to display in the plot.
        binno : [int or str] optional.  Identifier of the bin for labeling.

        Returns:
        result : The optimization result object from minimize.
        """
        # Compute histogram
        bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_width = bin_edges[1] - bin_edges[0]

        init_guess, param_names = np.array(list(initial_guess.values())), list(initial_guess.keys()) #param_names not to be confused with param_labels

        # Optimize
        result = minimize(
            neg_log_likelihood_func,
            init_guess,
            args=(bin_edges, bin_heights, bin_width),
            bounds=bounds
        )

        if verbose:
          print("Optimized parameters:", result.x)
          print(result)
        
        if save_params:
            param_dict = dict(zip(param_names, result.x)) 
            tail = os.path.split(filename)[1].strip('_fit.png')
            with open(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/{tail}.json', 'w') as f:
                dump(param_dict, f, indent = 1) #write to json
            pass

        # Plot if function provided
        if plot_func and dist_func:
            plot_func(dist_func, result.x, data, bins=bins, distname=distname, filename = filename)

        return result
    
    # Standard values taken from Sowmya's code
    def fit_to_data(self, method: str = 'emcee', bins = 70,
                     bounds = [(0.01, None), (0.0001, None), (-0.09, 1)], plot_func = None, dist_func = None,
                     nwalkers = 32, nsteps = 5000,
                     distname = 'Modified Gaussian', binno = 1, verbose = False, save_params = False):
        
        method = method.lower()
        if method not in ['emcee','minimize']:
            raise ValueError(f'Method name "{method}" not recognized. Specify either "emcee" or "minimize".')
        
        # init_params = np.array(list(self.initial_param_dict.values())) #from dict to np.array

        if method == 'minimize':
            filename = f'/disks/cosmodm/vdvuurst/figures/minimize_fits/M_{self.lower_mass}-{self.upper_mass}_fit.png'

            return self._fit_modified_gaussian_minimize(self.rel_vels, bins, self.initial_param_dict, bounds, 
                                            self.mod_gaussian_neg_log_likelihood_binned,
                                            plot_func, dist_func, distname, binno, verbose, filename, save_params)
        else:
            filename = f'M_{self.lower_mass}-{self.upper_mass}'
            return self._fit_modified_gaussian_emcee(self.rel_vels, bins, self.initial_param_dict, self.mod_gaussian_log_likelihood,
                                                     nwalkers = nwalkers, nsteps = nsteps,
                                                     verbose = verbose, save_params = save_params, plot = True, filename = filename) #change plot to a variable
    
    def fit_to_radial_bins(self, rbin = None, bins = 70, bounds = [(0.01, None), (0.0001, None), (-0.09, 1)], plot_func = None, dist_func = None,
                     distname = 'Modified Gaussian', binno = 1, verbose = False):
        
        if rbin is None:
            #TODO: do it for all radial bins, somehow define the radial bins w/o hardcoding (at least motivated unlike Sowmya did)
            return 
        
        else:
            radial_mask = (rbin[0]**2 <= self.rel_sq_dist) & (self.rel_sq_dist <= rbin[1]**2)
            masked_data = self.rel_vels[radial_mask]
            filename = f'/disks/cosmodm/vdvuurst/figures/M_{self.lower_mass}-{self.upper_mass}_r_{rbin[0]}-{rbin[1]}_fit.png'
            return self._fit_modified_gaussian_minimize(masked_data, bins, self.initial_param_dict, bounds, 
                                    self.mod_gaussian_neg_log_likelihood_binned,
                                    plot_func, dist_func, distname, binno, verbose, filename)

    #TODO: make this prettier
    @staticmethod
    def plot_distribution_gaussian_mod(f,params,data,bins,distname, filename = 'Mfit'):
        
        bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width= bin_edges[1] - bin_edges[0] 
        bin_widths = np.diff(bin_edges)  # The width of each bin
        number_density = bin_heights / bin_widths  # Normalize by bin width
        # Plot the histogram
        fig = plt.figure(figsize=(7,7))
        frame=fig.add_subplot(1,1,1)
        frame.set_xlabel('Velocity difference v', fontsize=16)
        frame.set_ylabel('Number of galaxies per v', fontsize=16)
        frame.bar(bin_centers, number_density, width=bin_width, align='center')
        DAT=np.linspace(np.min(data),np.max(data),1000)
        sigma,sigma1,lambda_=params 
        hist_area=np.sum(bin_heights)
        fit_integral = 1 #since we're using a normalized function
        A=hist_area/ fit_integral
        frame.plot(DAT,A*f(DAT,sigma,sigma1,lambda_),'-', label=f"{distname},\nN={hist_area:.0f}",color='red')
        # frame.set_yscale("log")
        frame.legend(fontsize=12.5, loc="upper right")
        frame.tick_params(axis='both', which='major',length=6, width=2,labelsize=14)
        fig.savefig(filename, dpi=300)
        plt.close()


if __name__ == '__main__':

    path = r'/disks/cosmodm/vdvuurst/data/OneHalo_0.5dex/M_13.5-14.0.hdf5'
    # fitter.initial_params = [412.6, 72.8, 0.21] #should work as a kickstart
    # fitter.fit_to_radial_bins(rbin = [0.2,0.23],verbose=True, plot_func= fitter.plot_distribution_gaussian_mod, dist_func = fitter.mod_gaussian)

    format_plot()

    param_fits_dir = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee'
    savepath = '/disks/cosmodm/vdvuurst/figures/emcee_results'
    for param_file in tqdm(os.listdir(param_fits_dir)):
        param_path = os.path.join(param_fits_dir, param_file)
        param_tail = param_file.split('.json')[0]

        path = f'/disks/cosmodm/vdvuurst/data/OneHalo_0.5dex/{param_tail}.hdf5'
        fitter = ONEHALO_fitter(PATH = path, initial_param_file = None)

        #  plot_func(dist_func, result.x, data, bins=bins, distname=distname, filename = filename)
        with open(param_path, 'r') as f:
            param_dict = load(f)

        # param_vals = np.array(param_dict[('sigma_1','sigma_2','lambda')])
        param_vals = np.array([param_dict[x] for x in ['sigma_1','sigma_2','lambda']])

        ONEHALO_fitter.plot_distribution_gaussian_mod(ONEHALO_fitter.mod_gaussian, param_vals, fitter.rel_vels,
                                                       bins = 70, distname = 'Modified Gaussian',
                                                        filename = os.path.join(savepath, f'{param_tail}_fit.png'))









    # dir = '/disks/cosmodm/vdvuurst/data/OneHalo_0.5dex'
    # for file in os.listdir(dir): 
    #     filehead = file.split('.hdf5')[0]

    #     if os.path.isfile(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/{filehead}.json'):
    #         initial_param_file = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/{filehead}.json'
    #     else:
    #         initial_param_file = None
        
    #     path = os.path.join(dir, file)
    #     fitter = ONEHALO_fitter(PATH = path, initial_param_file = initial_param_file, joint = False) #joint set to false for now since we haven't generalized to that yet
    #     fitter.initial_params = [412.6, 72.8, 0.21] #should work
    #     # fitter.fit_to_data(save_params = True)
    #     fitter.fit_to_data(verbose = True, plot_func= fitter.plot_distribution_gaussian_mod, dist_func = fitter.mod_gaussian)