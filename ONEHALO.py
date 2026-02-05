import numpy as np
import h5py
from typing import Tuple
from tqdm import tqdm
from onehalo_plotter import format_plot
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from functions import *
from onehalo_plotter import plot_distribution_gaussian_mod
from json import load, dump
import emcee
from corner import corner
from Etc.shrinking_gaussian_move import ShrinkingGaussianMove


#TODO: expand accordingly
latex_formatter = {'sigma_1':r'$\sigma_1$', 'sigma_2': r'$\sigma_2$', 'lambda':r'$\lambda$', 'loglambda':r'$\log_{10}\left(\lambda\right)$'}


class ONEHALO:
    def __init__(self, PATH: str):
        self.PATH = PATH #SOAP path

        with h5py.File(self.PATH, "r") as handle:
            self.IsCentral = handle["InputHalos/IsCentral"][:].astype(bool) #set to bool so it can be used as a mask
            self.HaloIndices = np.arange(self.IsCentral.size)
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
            self.HostHaloIndex = handle["SOAP/HostHaloIndex"][:] # -1 for centrals
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
            self.SOMass = handle['SO/200_mean/TotalMass'][:]
            self.Rvir = handle['SO/200_mean/SORadius'][:]

            self.boxsize = handle['Header'].attrs['BoxSize'][0]

        # self.half_boxsize = self.boxsize / 2
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
        # we do not need to explicitly filter for number of particles, it will always be at least 1000
        mass_mask = self._make_mass_mask(self.SOMass, self.lower_mass, self.upper_mass) #functions as a central mask implicitly
        self.mass_mask = mass_mask
        HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central
        
        # From the catalogue of sattelite hosting haloes, select only those we know actually HAVE subhaloes and thus are relevant for this dataset
        Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
        
        subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask]
        # self.subhalos_per_host = subhalos_per_host
        HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] # This now replaces the previous HostIndices
        # self.HostIndices = HostIndices

        SatteliteMask = np.isin(self.HostHaloIndex, HostIndices) #pick out all the sattelites from relevant hosts

        # Get the host COM pos and vel, same for sattelites
        HostCOMs, HostVels = self.COM[HostIndices], self.COMvelocity[HostIndices]
        SatSorter = np.argsort(self.HostHaloIndex[SatteliteMask]) # sorting so that we are sure to compare every sattelite to the right host below
        SatCOMs, SatVels = self.COM[SatteliteMask][SatSorter], self.COMvelocity[SatteliteMask][SatSorter]
        
        # Get the virial radii for all sattelites in order
        self.SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)

        relative_COMs = SatCOMs - np.repeat(HostCOMs, subhalos_per_host, axis = 0)
        relative_vels = SatVels - np.repeat(HostVels, subhalos_per_host, axis = 0)

        with h5py.File(self.filename, 'w') as file:
            file.create_dataset('rel_pos', data  = relative_COMs, dtype = np.float32)
            file.create_dataset('rel_vels', data  = relative_vels, dtype = np.float32)
    
    def create_radial_catalogue(self, radial_bin: Tuple[np.float32, np.float32], rad_filename:str, mass_filename:str, r_unit: str = 'Rvir'):
        with h5py.File(mass_filename, 'r') as handle:
            rel_vels = handle['rel_vels'][:] 
            rel_pos = handle['rel_pos'][:] 

        if r_unit == 'Rvir':
            # Change distance units to R200m (Rvir) instead of Mpc on a per halo basis
            try:
                rel_sq_dist = np.square(rel_pos).sum(axis = 1) / np.square(self.SatRvirs)

            except AttributeError: # We have not called the mass catalogue and need some stuff from it, so we calc it here
                mass_mask = self._make_mass_mask(self.SOMass, self.lower_mass, self.upper_mass) #functions as a central mask implicitly
                self.mass_mask = mass_mask
                HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central                
                # From the catalogue of sattelite hosting haloes, select only those we know actually HAVE subhaloes and thus are relevant for this dataset
                Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
                HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] # This now replaces the previous HostIndices
    
                subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask] # Get the virial radii for all sattelites in order
                self.SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)

                rel_sq_dist = np.square(rel_pos).sum(axis = 1) / np.square(self.SatRvirs)

        elif r_unit == 'Mpc':
            rel_sq_dist = np.square(rel_pos).sum(axis = 1)

        radial_mask = (radial_bin[0]**2 <= rel_sq_dist) & (rel_sq_dist <= radial_bin[1]**2)
        masked_data = rel_vels[radial_mask]

        if masked_data.size < 100: #TODO: check minimum, maybe even more - like 1000? make modifiable? also check in fit_to_radial_bins if changed
            return True # return that bin has insufficient data

        with h5py.File(rad_filename, 'w') as file:
            file.create_dataset('rel_vels', data = masked_data, dtype = np.float32)


class ONEHALO_fitter:
    def __init__(self, PATH: str, initial_param_file: str = None, joint: bool = False, loglambda: bool = False):
        """_summary_

        Args:
            PATH (str): _description_
            initial_param_file (str): _description_. Defaults to None.
            joint (bool, optional): Whether to use parameterizations of the three gaussian parameters (9 params in total). 
                                    Defaults to False.
        """
        self.PATH = PATH

        massbin = os.path.split(self.PATH)[-1].split('_')[-1].split('.hdf5')[0].split('-')
        self.lower_mass, self.upper_mass = np.float32(massbin[0]), np.float32(massbin[1])

        self.joint = joint

        if initial_param_file:
            if self.joint:
                with open(initial_param_file, 'r') as f:
                    self.initial_param_dict = load(f)[f'M_{self.lower_mass}-{self.upper_mass}']
                    #TODO: same as below presumably, with removing errors
            else:
                with open(initial_param_file, 'r') as f:
                    self.initial_param_dict = load(f)
                #NOTE: some dicts are bigger, we want only the param data, we dont care about error values
                self.initial_param_dict = {"sigma_1": self.initial_param_dict['sigma_1'],
                                            "sigma_2": self.initial_param_dict['sigma_2'], 
                                            "lambda": 0.} 

        else:
            # some random stuff in the ballpark of where we want them to kickstart
            if self.joint:
                # this will not work with an ensemble of function combinations
                raise ValueError("No initial parameter file specified.")
            # These values are ok to jumpstart the process across mass and radial bins
            else:
                self.initial_param_dict = {'sigma_1':500.0, 'sigma_2': 150.0, 'lambda':0 if not loglambda else -100} 

        with h5py.File(self.PATH, 'r') as handle:
            self.rel_pos = handle['rel_pos'][:]
            self.rel_vels = handle['rel_vels'][:]#.flatten()
        
        self.rel_sq_dist = np.square(self.rel_pos).sum(axis = 1)
        self.rel_vel_sq = np.square(self.rel_vels) * -0.5


    @staticmethod
    def _fit_modified_gaussian_emcee(data, bins: int | Callable, initial_guess: dict, log_likelihood_func, use_binned = False,
                                     nwalkers = 32, nsteps = 5000, param_labels = [r'$\sigma_1$',r'$\sigma_2$',r'$\lambda$'],
                                     plot = True, verbose = False, filename = 'Mfit', save_params = False, loglambda = False, **kwargs):
        
        #param_names not to be confused with param_labels; latter is latex formatted
        init_guess, param_names = np.array(list(initial_guess.values())), list(initial_guess.keys()) 
        ndim = init_guess.shape[0]

        random_pos = True #TODO: change to be modifiable from terminal 
        if not random_pos:
            
            #TODO: if this will even be used henceforth, it is now dependent on three parameters not some n
            noise = np.random.randn(nwalkers, ndim)
            noise[:,-1] = np.abs(noise[:,-1]) * 1e-4 #lambda must take positive values and be smaller
            if loglambda:
                noise[:, -1] *= -1
            pos = init_guess + noise

        else:
            # uniformly random starting positions in the prior range
            pos = np.random.uniform(low = [1., 1., 0.], high = [1500., 1500., 0.5], size = (nwalkers, ndim))

        loglambda_str = '_log_lambda' if loglambda else ''

        # If bins is passed as a function (e.g. ricebins in functions.py), calculate the amount of bins
        if hasattr(bins, '__call__'):
            bins = bins(data.size)

        if not use_binned:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_func, args = (data, loglambda, kwargs['single_gauss']))
            sampler.run_mcmc(pos, nsteps, progress = verbose)

        else:
            # Compute histogram
            bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_func, args = (bin_edges, bin_heights))
            sampler.run_mcmc(pos, nsteps, progress = verbose)

        burnin = 250 # does not matter so much

        if plot:
            # chain plots
            # if ndim > 3:
            #     nrows = np.ceil(ndim / 2)
            # TODO: with joint there's gonna be a lot of parameters which may clutter a walker plot if they are all plotted
            # vertically
            fig, axes = plt.subplots(ndim + 1, figsize=(12, 10), sharex=True)
            samples = sampler.get_chain()

            for i in range(ndim):
                ax:plt.Axes = axes[i]
                ax.plot(samples[:, :, i], alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(param_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

                ymin, ymax = ax.get_ylim()
                ax.vlines(burnin, ymin, ymax, colors = 'black', linestyles = '--')
                ax.set_ylim(ymin, ymax)

            likelihoods = sampler.get_log_prob()
            # we want the MAXIMUM likelihood here, not the minimum negative likelihood (which would be equivalent)
            best_arg = np.unravel_index(np.argmax(likelihoods), likelihoods.shape) 
            best_likelihood = likelihoods[*best_arg]
            best_params = np.array([samples[*best_arg, i] for i in range(ndim)])
  
            likelihoods_plot = np.log10(-1 * likelihoods)
            likelihoods_plot[:burnin] = np.nan # so that the lines are better discernable in the plot

            axes[-1].plot(likelihoods_plot, alpha = 0.3)
            axes[-1].set(ylabel = r'$\log\left(-\log(\mathcal{L})\right)$')
            axes[-1].set_xlabel("Step number")

            ymin, ymax = axes[-1].get_ylim()
            axes[-1].vlines(burnin, ymin, ymax, colors = 'black', linestyles = '--')
            axes[-1].set_ylim(ymin, ymax)

            plt.savefig(f'{filename}_walkers{loglambda_str}.png', dpi = 200)
            plt.close()

            #corner plot
            if not kwargs['single_gauss']:
                flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True) # this is probably fine

                fig = corner(flat_samples, labels = param_labels, quiet = True,
                                quantiles=[0.16, 0.5, 0.84])

                fig.savefig(f"{filename}_corner{loglambda_str}.png", dpi = 200)
                plt.close(fig)
        
        else: # If we do not plot we still need these values
            samples = sampler.get_chain()
            likelihoods = sampler.get_log_prob()
            best_arg = np.argmax(likelihoods) #doublecheck if max or min
            best_likelihood = likelihoods[best_arg]
            best_params = np.array([samples[best_arg, i] for i in range(ndim)])
        
 
        # Perturb the best found parameter set for an error estimate
        naive_perturbation = True
        if naive_perturbation:
            param_dict = perturb_around_likelihood(best_likelihood, best_params, lambda x: log_likelihood_func(x, data, loglambda, kwargs['single_gauss']), single_gauss = kwargs['single_gauss'])
        
        else: # start new, short MCMC chain to estimate the error
            pos = np.random.normal(loc = best_params, scale = 0.001 * best_params, size = (nwalkers, ndim))
            new_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_func, args = (data, loglambda, kwargs['single_gauss']))
            new_sampler.run_mcmc(pos, nsteps = 100, progress = False) # no progress lest it clutter up the output
            
            flat_samples_perturb = new_sampler.get_chain(discard=0, thin=5, flat=True) 
            
            errs = np.zeros((3,2))
            for i in range(ndim):
                mcmc = np.percentile(flat_samples_perturb[:, i], [16, 50, 84])
                err = np.diff(mcmc)
                errs[i,0] = err[0]
                errs[i,1] = err[1]
            
            # We ONLY use this short chain for error estimates
            param_dict = dict(zip(param_names, best_params.tolist()))
            param_dict['errors'] = errs.tolist()

            # Uncomment in case a diagnostic plot of the walkers is desired
            # fig, axes = plt.subplots(ndim + 1, figsize=(12, 10), sharex=True)
            # new_samples = new_sampler.get_chain()

            # for i in range(ndim):
            #     ax:plt.Axes = axes[i]
            #     ax.plot(new_samples[:, :, i], alpha=0.3)
            #     ax.set_xlim(0, len(new_samples))
            #     ax.set_ylabel(param_labels[i])
            #     ax.yaxis.set_label_coords(-0.1, 0.5)

            #     ymin, ymax = ax.get_ylim()
            #     ax.set_ylim(ymin, ymax)

            # likelihoods = new_sampler.get_log_prob()
            # # we want the MAXIMUM likelihood here, not the minimum negative likelihood (which would be equivalent)
            # best_arg = np.unravel_index(np.argmax(likelihoods), likelihoods.shape) 
            # best_likelihood = likelihoods[*best_arg]
            # best_params = np.array([new_samples[*best_arg, i] for i in range(ndim)])
  
            # likelihoods_plot = np.log10(-1 * likelihoods)

            # axes[-1].plot(likelihoods_plot, alpha = 0.3)
            # axes[-1].set(ylabel = r'$\log\left(-\log(\mathcal{L})\right)$')
            # axes[-1].set_xlabel("Step number")

            # ymin, ymax = axes[-1].get_ylim()
            # axes[-1].set_ylim(ymin, ymax)

            # plt.savefig(f'{filename}_PERTURBTEST.png', dpi = 200)
            # plt.close()


        # Add some more diagnostics to the dictionaries for later use and ease
        param_dict['nwalkers'] = nwalkers
        param_dict['nsteps'] = nsteps
        param_dict['N'] = data.size
        param_dict['likelihood'] = best_likelihood

        if save_params:
            head, rtail = os.path.split(filename)
            if not kwargs['single_gauss']:
                masstail = os.path.split(head)[1]
            else:
                head = os.path.split(head)[0]
                masstail = os.path.split(head)[1]

            param_path = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee'
            mkdir_if_non_existent(os.path.join(param_path, masstail))
            
            param_path = os.path.join(param_path, masstail, kwargs['r_unit'])            
            mkdir_if_non_existent(param_path)

            single_gauss_str = '' if not kwargs['single_gauss'] else '_single_gaussian'
            param_path = os.path.join(param_path, rtail + f'{loglambda_str}{single_gauss_str}.json')
            
            with open(param_path, 'w') as f:
                dump(param_dict, f, indent = 1)
        
        # Plot if function provided
        if plot:
            if loglambda:
                if not kwargs['single_gauss']: 
                    plot_distribution_gaussian_mod(mod_gaussian_loglambda, param_dict, data, bins=bins, distname='Double Gaussian', filename = filename + f'_fit{loglambda_str}.png', loglambda = True)
                else: 
                    plot_distribution_gaussian_mod(mod_gaussian_loglambda, param_dict, data, bins=bins, distname="Single Gaussian", filename = filename + f'_fit{loglambda_str}.png', loglambda = True)
            else:
                if not kwargs['single_gauss']: 
                    plot_distribution_gaussian_mod(mod_gaussian, param_dict, data, bins=bins, distname='Double Gaussian', filename = filename + f'_fit{loglambda_str}.png', loglambda = False)
                else: 
                    plot_distribution_gaussian_mod(mod_gaussian, param_dict, data, bins=bins, distname="Single Gaussian", filename = filename + f'_fit{loglambda_str}.png', loglambda = False, single_gauss = True)
        return param_dict
        # return samples

    @staticmethod
    def _fit_modified_gaussian_minimize(data, bins, initial_guess,bounds, neg_log_likelihood_func,
                              plot: bool = False, distname = 'Double Gaussian', use_binned = True,
                              verbose = False, filename = 'Mfit', save_params = False):
        """
        Fits a modified Gaussian distribution using minimize (scipy.optimize) to binned data and plots the result.

        Input:
        data : Input peculiar velocity difference data to fit.
        bins : [int] Number of histogram bins.
        initial_guess : Initial guess for the parameters.
        bounds :  Bounds for the parameters .
        neg_log_likelihood_func : Negative log_likelihood function to minimize (negative log-likelihood). Must use (bin_edges, bin_heights, result.x).
        plot_func : Function to plot the results. Must use (dist_func, params, data, bins, distname, binno).
        dist_func :  optional. Distribution function to be used while plotting.
        distname : [str] optional.  Name of the distribution to display in the plot.
        binno : [int or str] optional.  Identifier of the bin for labeling.

        Returns:
        result : The optimization result object from minimize.
        """

        init_guess, param_names = np.array(list(initial_guess.values())), list(initial_guess.keys()) #param_names not to be confused with param_labels
        
        # If bins is passed as a function (e.g. ricebins in functions.py), calculate the amount of bins
        if hasattr(bins, '__call__'):
            bins = bins(data.size)

        if use_binned:
            bin_heights, bin_edges = np.histogram(data, bins=bins, density=False) # Compute histogram

            # Optimize
            result = minimize(
                neg_log_likelihood_func,
                init_guess,
                args=(bin_edges, bin_heights),
                bounds=bounds
            )

        else: #non-binned likelihood takes less args
            result = minimize(
                neg_log_likelihood_func,
                init_guess,
                args=(data),
                bounds=bounds
            )
        
        if save_params:
            param_dict = dict(zip(param_names, result.x)) 
            head, rtail = os.path.split(filename)
            masstail = os.path.split(head)[1]

            param_path = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize'
            mkdir_if_non_existent(os.path.join(param_path, masstail))
                        
            param_path = os.path.join(param_path, masstail, rtail + '.json')
            
            with open(param_path, 'w') as f:
                dump(param_dict, f, indent = 1) #write to json
        
        # Plot if function provided
        if plot:
            plot_distribution_gaussian_mod(mod_gaussian, result.x, data, bins=bins, distname=distname, filename = filename + '_fit.png')

        return result
    
    # Standard values taken from Sowmya's code
    def fit_to_data(self, method: str, bins: int = 200, bounds: list = [(0.01, None), (0.0001, None), (-0.09, 1)],
                     plot: bool = True, nwalkers: int = 8, nsteps: int = 1000, non_bin_threshold: int = -1,
                     distname: str = 'Double Gaussian', verbose: bool = False, save_params: bool = False, **kwargs):
        
        method = method.lower()
        allowed = ['emcee','minimize']
        if method not in allowed:
            raise ValueError(f'Method name "{method}" not recognized. Choose from {*allowed,}.')
        
        if non_bin_threshold != -1: #default, then we never bin in the likelihood
            use_binned = self.rel_vels.size > non_bin_threshold
        else:
            use_binned = False    
        filename = f'/disks/cosmodm/vdvuurst/figures/{method}_results/M_{self.lower_mass}-{self.upper_mass}_fit.png'

        if method == 'minimize':
            likelihood_func = mod_gaussian_neg_log_likelihood_binned if use_binned else mod_gaussian_neg_log_likelihood

            return self._fit_modified_gaussian_minimize(data = self.rel_vels, bins = bins, initial_guess= self.initial_param_dict, bounds = bounds, 
                                                            neg_log_likelihood_func = likelihood_func, plot = plot, distname = distname, use_binned = False, verbose = verbose, filename = filename,
                                                              save_params = save_params)

        else:
            likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood

            return self._fit_modified_gaussian_emcee(self.rel_vels, bins, self.initial_param_dict, likelihood_func,
                                                     nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                     verbose = verbose, save_params = save_params, plot = plot, filename = filename)
    

    def _fit_to_catalogued_radial_bins(self, method:str, rbins: np.array, datapath, **kwargs):
        #datapath should go to massbin folder with hdf5 files
        method = method.lower()
        allowed = ['emcee','minimize']
        if method not in allowed:
            raise ValueError(f'Method name "{method}" not recognized. Choose from {*allowed,}.')
        
        if kwargs['return_values']:
            results = np.zeros((len(rbins), 3))
            errors = np.zeros((len(rbins), 3, 2))
        
        for i,rbin in enumerate(rbins):
            rpath = os.path.join(datapath, kwargs['r_unit'], f'r_{rbin[0]:.2f}-{rbin[1]:.2f}.hdf5')

            try:
                with h5py.File(rpath, 'r') as file:
                    masked_data = file['rel_vels'][:]
            except FileNotFoundError:
                if kwargs['verbose']:
                    tqdm.write(f'{rpath} does not exist, skipping...')
                continue

            filename = f'/disks/cosmodm/vdvuurst/figures/{method}_results_radial_bins_{kwargs['r_unit']}/M_{self.lower_mass}-{self.upper_mass}'
            mkdir_if_non_existent(f'/disks/cosmodm/vdvuurst/figures/{method}_results_radial_bins_{kwargs['r_unit']}')
            mkdir_if_non_existent(filename)

            if kwargs['single_gauss']:
                filename += f'/single_gaussian'
                mkdir_if_non_existent(filename)

            filename += f'/r_{rbin[0]:.2f}-{rbin[1]:.2f}'

            if kwargs['non_bin_threshold'] != -1: #default, then we never bin in the likelihood
                use_binned = masked_data.size > kwargs['non_bin_threshold']
            else:
                use_binned = False

            if method == 'emcee':
                likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
                output = self._fit_modified_gaussian_emcee(data = masked_data, initial_guess = self.initial_param_dict, log_likelihood_func = likelihood_func,
                                                                filename = filename, use_binned = use_binned, **kwargs)
                
                if kwargs['return_values']:
                    result, err = output   
                    results[i] = result
                    errors[i] = err

                if kwargs['verbose']:
                    print(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} completed.')

            else: # minimize
                likelihood_func = mod_gaussian_neg_log_likelihood
                result = self._fit_modified_gaussian_minimize(data = masked_data, initial_guess= self.initial_param_dict, filename = filename,
                                                            neg_log_likelihood_func = likelihood_func, use_binned = use_binned, **kwargs)
                if kwargs['return_values']:
                    results[i] = result.x
        
        if kwargs['return_values']:
            return results, errors


    def _fit_to_non_catalogued_radial_bins(self, method:str, rbins = None, r_start: float = 0., r_stop:float = 5., r_steps: int = 18, 
                            bins: int | Callable = 200, bounds: list = [(50, 1000), (50, 1000), (0, 1)], plot: bool = True,
                            nwalkers: int = 16, nsteps: int = 1000, non_bin_threshold: int = -1,
                            distname: str = 'Double Gaussian', verbose: bool = False, save_params: bool = False, overwrite: bool = True,
                            return_values = False, loglambda = False, **kwargs):
        """_summary_

        Args:
            method (str): _description_
            rbins (Tuple or array_like of tuples, optional): either a 2-tuple or a list of 2-tuples specifiying the bin_edges. 
                                                             If None, r_start, r_stop and r_step will be used to log-space the bins. Defaults to None.
            r_start (float, optional): _description_. Defaults to 0.
            r_stop (float, optional): _description_. Defaults to 5.
            r_steps (int, optional): _description_. Defaults to 18.
            bins (int, optional): _description_. Defaults to 200.
            bounds (list, optional): _description_. Defaults to [(0.01, None), (0.0001, None), (-0.09, 1)].
            plot (bool, optional): _description_. Defaults to True.
            nwalkers (int, optional): _description_. Defaults to 8.
            nsteps (int, optional): _description_. Defaults to 500.
            non_bin_threshold (int, optional): _description_. Defaults to 1000.
            distname (str, optional): _description_. Defaults to 'Double Gaussian'.
            verbose (bool, optional): _description_. Defaults to False.
            save_params (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        method = method.lower()
        allowed = ['emcee','minimize']
        if method not in allowed:
            raise ValueError(f'Method name "{method}" not recognized. Choose from {*allowed,}.')
        
        if rbins is not None:
            if len(rbins) > 2:
                results = np.zeros((len(rbins), 3))
                errors = np.zeros_like(results)
                for rbin in rbins:
                    radial_mask = (rbin[0]**2 <= self.rel_sq_dist) & (self.rel_sq_dist <= rbin[1]**2)
                    masked_data = self.rel_vels[radial_mask]
                    if non_bin_threshold != -1: #default, then we never bin in the likelihood
                        use_binned = masked_data.size > non_bin_threshold
                    else:
                        use_binned = False                
                    filename = f'/disks/cosmodm/vdvuurst/figures/{method}_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
                
                    if method == 'emcee':
                        likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
                        result, err = self._fit_modified_gaussian_emcee(self.rel_vel_sq, bins, self.initial_param_dict, mod_gaussian_log_likelihood,
                                                            nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                            verbose = verbose, save_params = save_params, plot = plot, filename = filename, **kwargs)
                        results[i] = result
                        errors[i] = err

                    else:
                        likelihood_func = mod_gaussian_neg_log_likelihood_binned if use_binned else mod_gaussian_neg_log_likelihood
                        result = self._fit_modified_gaussian_minimize(data = masked_data, bins = bins, initial_guess= self.initial_param_dict, bounds = bounds, 
                                                                    neg_log_likelihood_func = likelihood_func, plot = plot,
                                                                    distname = distname, binned = False, verbose = verbose, filename = filename,
                                                                    save_params = save_params)
                        results[i] = result.x

                return results, errors

            else:
                radial_mask = (rbins[0]**2 <= self.rel_sq_dist) & (self.rel_sq_dist <= rbins[1]**2)
                masked_data = self.rel_vels[radial_mask]

                if non_bin_threshold != -1: #default, then we never bin in the likelihood
                    use_binned = masked_data.size > non_bin_threshold
                else:
                    use_binned = False            

                filename = f'/disks/cosmodm/vdvuurst/figures/{method}_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
                if os.path.isfile(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/{method}/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}.json') and not overwrite:
                    print(f'M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f} already done, skipping...')
            
                if method == 'emcee':
                    likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
                    result, err = self._fit_modified_gaussian_emcee(masked_data, bins, self.initial_param_dict, likelihood_func,
                                                     nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                     verbose = verbose, save_params = save_params, plot = plot, filename = filename, loglambda = loglambda, **kwargs)
                    if return_values:
                        return result, err

                else:
                    likelihood_func = mod_gaussian_neg_log_likelihood_binned if use_binned else mod_gaussian_neg_log_likelihood
                    result = self._fit_modified_gaussian_minimize(data = masked_data, bins = bins, initial_guess= self.initial_param_dict, bounds = bounds, 
                                                                neg_log_likelihood_func = likelihood_func, plot = plot,
                                                                distname = distname, binned = False, verbose = verbose, filename = filename,
                                                                save_params = save_params)
                    if return_values:
                        return result, np.zeros_like(result)
        
        # if an rbin is not specified, we log-space the bins in the function manually
        rbins = modified_logspace(r_start, r_stop, r_steps) 
        if return_values:
            results = np.zeros((r_steps - 1, 3))
            errors = np.zeros((r_steps - 1, 3, 2))
        for i in range(r_steps - 1): 
            rbin = (rbins[i], rbins[i+1])

            filename = f'/disks/cosmodm/vdvuurst/figures/{method}_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
            if os.path.isfile(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/{method}/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}.json') and not overwrite:
                print(f'M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f} already done, skipping...')
                continue
            
            radial_mask = (rbin[0]**2 <= self.rel_sq_dist) & (self.rel_sq_dist <= rbin[1]**2)
            masked_data = self.rel_vels[radial_mask]


            if masked_data.size < 100: #TODO: check minimum, maybe even more - like 1000? make modifiable?
                if verbose:
                    print(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} contains too little datapoints, skipping...')
                continue

            if verbose:
                print(f"Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} contains {masked_data.size} datapoints")

            if non_bin_threshold != -1: #default, then we never bin in the likelihood
                use_binned = masked_data.size > non_bin_threshold
            else:
                use_binned = False
           
            if method == 'emcee':
                likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
                result, err = self._fit_modified_gaussian_emcee(masked_data, bins, self.initial_param_dict, likelihood_func,
                                                     nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                     verbose = verbose, save_params = save_params, plot = plot, filename = filename, loglambda = loglambda)
                if return_values:   
                    results[i] = result
                    errors[i] = err
                if not verbose:
                    print(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} completed.')

            else: # minimize
                likelihood_func = mod_gaussian_neg_log_likelihood
                result = self._fit_modified_gaussian_minimize(data = masked_data, bins = bins, initial_guess= self.initial_param_dict, bounds = bounds, 
                                                            neg_log_likelihood_func = likelihood_func, plot = plot,
                                                            distname = distname, use_binned = use_binned, verbose = verbose, filename = filename,
                                                            save_params = save_params)
                if return_values:
                    results[i] = result.x
        
        if return_values:
            return results, errors


    def fit_to_radial_bins(self, catalogued: bool = True, **kwargs):
        if catalogued:
            self._fit_to_catalogued_radial_bins(**kwargs)
        else:
            self._fit_to_non_catalogued_radial_bins(**kwargs)


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
                                                       bins = 70, distname = 'Double Gaussian',
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