import numpy as np
import h5py
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from functions import *
from onehalo_plotter import plot_distribution_gaussian_mod
from json import load, dump
import emcee
from corner import corner
from functional_forms import * # also runs some code to create the functional_form catalogue in the all_combis variable
from time import time

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
                lower_mass, upper_mass = os.path.split(mass_filename)[1].split('.hdf5')[0].split('M_')[1].split('-')
                lower_mass, upper_mass = float(lower_mass) -10., float(upper_mass) -10. # -10. because of units

                mass_mask = self._make_mass_mask(self.SOMass, lower_mass, upper_mass) #functions as a central mask implicitly

                HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central                
                # From the catalogue of sattelite hosting haloes, select only those we know actually HAVE subhaloes and thus are relevant for this dataset
                Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
                HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] # This now replaces the previous HostIndices
    
                subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask] # Get the virial radii for all sattelites in order
                SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)

                rel_sq_dist = np.square(rel_pos).sum(axis = 1) / np.square(SatRvirs)

        elif r_unit == 'Mpc':
            rel_sq_dist = np.square(rel_pos).sum(axis = 1)

        radial_mask = (radial_bin[0]**2 <= rel_sq_dist) & (rel_sq_dist <= radial_bin[1]**2)
        masked_data = rel_vels[radial_mask]

        if masked_data.size < 100: #TODO: check minimum, maybe even more - like 1000? make modifiable? also check in fit_to_radial_bins if changed
            return True # return that bin has insufficient data

        with h5py.File(rad_filename, 'w') as file:
            file.create_dataset('rel_vels', data = masked_data, dtype = np.float32)


class ONEHALO_fitter:
    def __init__(self, PATH: str, initial_param_file: str = None, loglambda: bool = False):
        """_summary_

        Args:
            PATH (str): Path to hdf5 file specifying the data in a given massbin
            initial_param_file (str): path to .json file holding initial parameter values for the fitting process. If None, empirically good default values are used. Defaults to None.
        """
        self.PATH = PATH

        # Exctract massbin from the file path
        massbin = os.path.split(self.PATH)[-1].split('_')[-1].split('.hdf5')[0].split('-')
        self.lower_mass, self.upper_mass = np.float32(massbin[0]), np.float32(massbin[1])

        if initial_param_file: # if some file with initial parameters is specified
            with open(initial_param_file, 'r') as f:
                self.initial_param_dict = load(f)
            #NOTE: some dicts are bigger, we want only the param data, we dont care about error values
            self.initial_param_dict = {"sigma_1": self.initial_param_dict['sigma_1'],
                                        "sigma_2": self.initial_param_dict['sigma_2'], 
                                        "lambda": 0.} 
        else:
            # some random stuff in the ballpark of where we want them to kickstart, works well
            self.initial_param_dict = {'sigma_1':500.0, 'sigma_2': 150.0, 'lambda':0 if not loglambda else -100} 

        # Load in the data
        with h5py.File(self.PATH, 'r') as handle:
            self.rel_pos = handle['rel_pos'][:]
            self.rel_vels = handle['rel_vels'][:]
        
        # Some calculations to store for marginal speed-up in the MCMC process
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

        burnin = 50 # does not matter so much

        samples = sampler.get_chain()
        likelihoods = sampler.get_log_prob()
        best_arg = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
        best_likelihood = likelihoods[best_arg]
        best_params = np.array([samples[*best_arg, i] for i in range(ndim)])

        now = time()
        # Perturb the best found parameter set for an error estimate
        param_dict = perturb_around_likelihood(best_likelihood, best_params, lambda x: log_likelihood_func(x, data, loglambda, kwargs['single_gauss']), single_gauss = kwargs['single_gauss'])
        duration = time() - now
        # duration = 0.
        # # NOTE: stand-in to save purely the results from MCMC
        # param_dict = {'sigma_1':best_params[0], 'sigma_2':best_params[1], 'lambda': best_params[2], 'errors':[[] for _ in range(len(best_params))]}
        # for i in range(3):
        #     param_dict['errors'][i] = [np.abs(best_params[1] - GLOBAL_PRIOR_RANGE[i][0]), np.abs(best_params[i] - GLOBAL_PRIOR_RANGE[i][1])]


        # Sometimes, sigma_1 is seen as the small contribution. For consistency we flip this and break the prior
        if param_dict['sigma_1'] < param_dict['sigma_2'] and kwargs['flip_sigmas']:

            rbin = kwargs['rbin']
            tqdm.write(f' r {rbin[0]:.2f}-{rbin[1]:.2f}: SIGMA1 < SIGMA2, FLIPPING')
            #Switch sigma_1 and sigma_2 (and errors) and flip lambda to 1 - lambda (thus switching the errors)
            param_dict['sigma_1'], param_dict['sigma_2'] = param_dict['sigma_2'], param_dict['sigma_1']
            param_dict['errors'][0], param_dict['errors'][1] = param_dict['errors'][1], param_dict['errors'][0]
            param_dict['lambda'] = 1 - param_dict['lambda']
            param_dict['errors'][2][0], param_dict['errors'][2][1] = param_dict['errors'][2][1], param_dict['errors'][2][0]


        # Add some more diagnostics to the dictionaries for later use and ease
        param_dict['nwalkers'] = nwalkers
        param_dict['nsteps'] = nsteps
        param_dict['N'] = data.size
        param_dict['likelihood'] = best_likelihood

        if plot:
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
        
        return param_dict, duration
        # return samples

    def fit_to_catalogued_bin(self, rbin: np.array, datapath: str, **kwargs):
        #datapath should go to massbin folder with hdf5 files
    
        rpath = os.path.join(datapath, kwargs['r_unit'], f'r_{rbin[0]:.2f}-{rbin[1]:.2f}.hdf5')

        try:
            with h5py.File(rpath, 'r') as file:
                masked_data = file['rel_vels'][:]
        except FileNotFoundError:
            if kwargs['verbose']:
                tqdm.write(f'{rpath} does not exist, skipping...')
            return

        filename = f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins_{kwargs['r_unit']}/M_{self.lower_mass}-{self.upper_mass}'
        mkdir_if_non_existent(f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins_{kwargs['r_unit']}')
        mkdir_if_non_existent(filename)

        if kwargs['single_gauss']:
            filename += f'/single_gaussian'
            mkdir_if_non_existent(filename)

        filename += f'/r_{rbin[0]:.2f}-{rbin[1]:.2f}'

        likelihood_func = mod_gaussian_log_likelihood
        output, duration = self._fit_modified_gaussian_emcee(data = masked_data, initial_guess = self.initial_param_dict, log_likelihood_func = likelihood_func,
                                                        filename = filename, rbin = rbin, **kwargs)
        
        with open('timing.txt', 'a') as tfile:
            tfile.write(f'{self.lower_mass}-{self.upper_mass},{rbin[0]:.2f}-{rbin[1]:.2f},{duration}\n')
            

        if kwargs['verbose']:
            print(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} completed.')

        if kwargs['return_values']:
            result, err = output   
            return result, err

    def _fit_to_catalogued_radial_bins_sequential(self, rbins: np.array, datapath, **kwargs):
        #datapath should go to massbin folder with hdf5 files
        
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

            filename = f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins_{kwargs['r_unit']}/M_{self.lower_mass}-{self.upper_mass}'
            mkdir_if_non_existent(f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins_{kwargs['r_unit']}')
            mkdir_if_non_existent(filename)

            if kwargs['single_gauss']:
                filename += f'/single_gaussian'
                mkdir_if_non_existent(filename)

            filename += f'/r_{rbin[0]:.2f}-{rbin[1]:.2f}'

            if kwargs['non_bin_threshold'] != -1: #default, then we never bin in the likelihood
                use_binned = masked_data.size > kwargs['non_bin_threshold']
            else:
                use_binned = False

            likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
            output = self._fit_modified_gaussian_emcee(data = masked_data, initial_guess = self.initial_param_dict, log_likelihood_func = likelihood_func,
                                                            filename = filename, use_binned = use_binned, rbin = rbin, **kwargs)
            
            if kwargs['return_values']:
                result, err = output   
                results[i] = result
                errors[i] = err

            if kwargs['verbose']:
                print(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} completed.')

        
        if kwargs['return_values']:
            return results, errors


    def _fit_to_non_catalogued_radial_bins(self, rbins = None, r_start: float = 0., r_stop:float = 5., r_steps: int = 18, 
                            bins: int | Callable = 200, bounds: list = [(50, 1000), (50, 1000), (0, 1)], plot: bool = True,
                            nwalkers: int = 16, nsteps: int = 1000, non_bin_threshold: int = -1,
                            distname: str = 'Double Gaussian', verbose: bool = False, save_params: bool = False, overwrite: bool = True,
                            return_values = False, loglambda = False, **kwargs):
        """_summary_

        Args:
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
                    filename = f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
                
                    likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
                    result, err = self._fit_modified_gaussian_emcee(self.rel_vels, bins, self.initial_param_dict, mod_gaussian_log_likelihood,
                                                        nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                        verbose = verbose, save_params = save_params, plot = plot, filename = filename, **kwargs)
                    results[i] = result
                    errors[i] = err

                if return_values:
                    return results, errors

            else:
                radial_mask = (rbins[0]**2 <= self.rel_sq_dist) & (self.rel_sq_dist <= rbins[1]**2)
                masked_data = self.rel_vels[radial_mask]

                if non_bin_threshold != -1: #default, then we never bin in the likelihood
                    use_binned = masked_data.size > non_bin_threshold
                else:
                    use_binned = False            

                filename = f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
                if os.path.isfile(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}.json') and not overwrite:
                    print(f'M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f} already done, skipping...')
            
                likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
                result, err = self._fit_modified_gaussian_emcee(masked_data, bins, self.initial_param_dict, likelihood_func,
                                                    nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                    verbose = verbose, save_params = save_params, plot = plot, filename = filename, loglambda = loglambda, **kwargs)
                if return_values:
                    return result, err
        
        # if an rbin is not specified, we log-space the bins in the function manually
        rbins = modified_logspace(r_start, r_stop, r_steps) 
        if return_values:
            results = np.zeros((r_steps - 1, 3))
            errors = np.zeros((r_steps - 1, 3, 2))
        for i in range(r_steps - 1): 
            rbin = (rbins[i], rbins[i+1])

            filename = f'/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
            if os.path.isfile(f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}.json') and not overwrite:
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
           
            likelihood_func = mod_gaussian_log_likelihood_binned if use_binned else mod_gaussian_log_likelihood
            result, err = self._fit_modified_gaussian_emcee(masked_data, bins, self.initial_param_dict, likelihood_func,
                                                    nwalkers = nwalkers, nsteps = nsteps, use_binned = use_binned,
                                                    verbose = verbose, save_params = save_params, plot = plot, filename = filename, loglambda = loglambda)
            if return_values:   
                results[i] = result
                errors[i] = err
            if not verbose:
                print(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} completed.')

        if return_values:
            return results, errors


    def fit_to_radial_bins(self, catalogued: bool = True, **kwargs):
        if catalogued:
            self._fit_to_catalogued_radial_bins_sequential(**kwargs)
        else:
            self._fit_to_non_catalogued_radial_bins(**kwargs)




# def create_function_combinations(funclist): 
#     func_combis = [list(product([rfunc], list(combinations_with_replacement(m_funcs, no_params(rfunc))))) for rfunc in funclist]
#     return flatten(func_combis)

class ONEHALO_joint_fitter:
    def __init__(self, PATH: str = '/disks/cosmodm/vdvuurst/data/OneHalo_0.5dex', init_param_file: str = '/disks/cosmodm/vdvuurst/data/initial_params.json'):
        self.PATH = PATH

        # with open(init_param_file, 'r') as f:
        #     self.init_param_dict = load(f)

        # TODO: hardcoded now, make function argument
        mass_edges = [12 + 0.5*i for i in range(8)]
        self.massbins = [f'M_{mass_edges[m]}-{mass_edges[m+1]}' for m in range(len(mass_edges)-1)]

        #TODO: same as above
        r_range = modified_logspace(0, 2.5, 20)
        self.r_bins = [f'{r_range[i]:.2f}-{r_range[i+1]:.2f}' for i in range(len(r_range)-1)]

        self.data_dict = {m:{r:[] for r in self.r_bins} for m in self.massbins}
        # Load in the data for all mass- and radial bins
        for mbin in self.massbins:
            for rbin in self.r_bins:
                path_to_data = os.path.join(self.PATH, mbin, 'Rvir', 'r_'+ rbin + '.hdf5')
                with h5py.File(path_to_data, 'r') as handle:
                    # rel_pos = handle['rel_pos'][:]
                    rel_vels = handle['rel_vels'][:]
                
                # Some calculations to store for marginal speed-up in the MCMC process
                # rel_sq_dist = np.square(rel_pos).sum(axis = 1)
                
                # self.data_dict[mbin]['rel_sq_dist'] = rel_sq_dist
                self.data_dict[mbin][rbin] = rel_vels

    def joint_likelihood(self, params: list, r_func: list, m_funcs: list, n_params_r: list, n_params_m: list):
        # lambda_
        # Can this be vectorized?
        L = 0
        for mbin in self.massbins:
            for rbin in self.r_bins:
                vel_data = self.data_dict[mbin][rbin]
                



    def fit_to_data(self, function_combi: list, nwalkers: int = 10, nsteps: int = 500, **kwargs):
        # lambda_combi, sigma1_combi, sigma2_combi = function_combi
        n_params_r = []
        n_params_m = [[], [], []]
        for i,param_combi in enumerate(function_combi):
            r_function, m_parametrizations = param_combi # unpack the function combination for this specific parameter
            n_params_r.append(len(m_parametrizations))
            n_params_m[i] = [len(signature(m_func).parameters) - 1 for m_func in m_parametrizations]
        
        n_params = sum(flatten(n_params_m))

        return n_params_r, n_params_m

        # #TODO: tweak based on results and see if it can be made function specific
        # init_pos = np.random.uniform(low = [-1000 for _ in range(n_params)], high = [1000 for _ in range(n_params)], size = (nwalkers, n_params))

        # sampler = emcee.EnsembleSampler(nwalkers, n_params, self.joint_likelihood, args = (self.data_dict))
        # sampler.run_mcmc(init_pos, nsteps, progress = kwargs['verbose'])

        

        
    #     init_guess, param_names = np.array(list(initial_guess.values())), list(initial_guess.keys()) 
    #     ndim = init_guess.shape[0]



    ## Below is wrong, since these are initial parameters based ONLY on r. I have no idea where the real ballpark
    ## is, so just try smth out and we'll see how it goes. Same with the priors.
    # def _get_init_params(self, function_name: str):
    #     match function_name.lower():
    #         case 'power_linear':
    #             return np.array([self.init_param_dict[x] for x in ['p', 'n', 'q', 'b']])
    #         case 'linear':
    #             return np.array([self.init_param_dict[x] for x in ['m', 'c']])
    #         case 'exponential':
    #             return np.array([self.init_param_dict[x] for x in ['A', 'B', 'C']])
    #         case 'exponential_squared':
    #             return np.array([self.init_param_dict[x] for x in ['A', 'B', 'C']])
    #         case 'power_law':
    #             return np.array([self.init_param_dict[x] for x in ['p', 'n', 'q']])
    #         case 'constant':
    #             return 400.
    #         case 'parabola':
    #             return np.hstack(([1], np.array([self.init_param_dict[x] for x in ['m', 'c']])))
    #         #the three below empircally deduced to work decently
    #         case 'inverse':
    #             return np.array([1., 0.2])
    #         case 'poly_3':
    #             return np.array([10, -50, 100, 50])
    #         case 'poly_4':
    #             return np.array([10, -50, 100, 0.5, 10])

    #     raise ValueError(f"{function_name} not recognized, select a valid function.")