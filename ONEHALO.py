import numpy as np
import h5py
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from functions import *
from onehalo_plotter import plot_distribution_gaussian_mod, format_plot
from json import dump
import emcee
from corner import corner
from functional_forms import * # also runs some code to create the functional_form catalogue in the all_combis variable
from time import time

#TODO: expand accordingly
latex_formatter = {'sigma_1':r'$\sigma_1$', 'sigma_2': r'$\sigma_2$', 'lambda':r'$\lambda$', 'loglambda':r'$\log_{10}\left(\lambda\right)$'}

def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32) -> np.ndarray:
    if m_min in [0,-1,np.nan, None]:
        return (mass <= 10**m_max)
    elif m_max in [0,-1,np.nan, None]:
        return (10**m_min <= mass)
    return (10**m_min <= mass) & (mass <= 10**m_max) 

def _make_radial_mask(radii: np.ndarray, r_min: np.float32, r_max: np.float32) -> np.ndarray:
    return (r_min <= radii) & (r_max >= radii)

def str_from_mbin(mbin):
    return f'M_1{mbin[0]}-1{mbin[1]}'

def str_from_rbin(rbin):
    return f'r_{rbin[0]:.2f}-{rbin[1]:.2f}'

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

        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

        # ~IsCentral picks out sattelites, this picks out all centrals !that host sattelites! and how many sattelites they host
        self.HostHaloIDs, self.subhalos_per_host_tot = np.unique(self.HostHaloIndex[~self.IsCentral], return_counts = True) 

    def create_full_dataset(self, mass_range: Tuple[np.float32,np.float32], filename: str, r_max: np.float32 = 2.5, verbose = False):
        """ Create a full catalogue of halo mass, relative velocities and positions in a given mass range (so NOT binning!).

        Args:
            mass_range (Tuple[np.float32,np.float32]): Massbin in units of 10^10 Msol. Lower, upper.
            filename (str): string specifying the full path to which the data is saved (must end on .hdf5)
            r_max (float) : maximum radius to allow. Defualts to 2.5
        """
        lower_mass, upper_mass = mass_range
        mass_mask = _make_mass_mask(self.SOMass, lower_mass, upper_mass) #functions as a central mask implicitly
        
        # halo_masses = self.SOMass[mass_mask] # get all the masses
        
        HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central
        
        # From the catalogue of sattelite hosting haloes, select only those we know actually HAVE subhaloes and thus are relevant for this dataset
        Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
        
        subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask]
        HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] # This now replaces the previous HostIndices

        SatteliteMask = np.isin(self.HostHaloIndex, HostIndices) # Pick out all the sattelites from relevant hosts

        if verbose: print('Sattelites found')

        # Get the host COM pos, vel and mass, same for sattelites
        HostCOMs, HostVels = self.COM[HostIndices], self.COMvelocity[HostIndices]
        HostMasses = self.SOMass[HostIndices]

        SatSorter = np.argsort(self.HostHaloIndex[SatteliteMask]) # sorting so that we are sure to compare every sattelite to the right host below
        SatCOMs, SatVels = self.COM[SatteliteMask][SatSorter], self.COMvelocity[SatteliteMask][SatSorter]
        if verbose: print('Sattelites sorted')

        # Get the virial radii for all sattelites in order
        self.SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)
        relative_COMs = SatCOMs - np.repeat(HostCOMs, subhalos_per_host, axis = 0)
        relative_vels = SatVels - np.repeat(HostVels, subhalos_per_host, axis = 0)
        
        # Get the virial radii for all sattelites in order
        SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)
        rel_dist = np.sqrt(np.square(relative_COMs).sum(axis = 1)) / SatRvirs # Calculate the distance from coordinates and set in units of virial radius
        HostMasses = np.repeat(HostMasses, subhalos_per_host, axis = 0)

        if verbose: print('All sattelite data found, applying masks...')

        radial_mask = rel_dist < r_max # We set a maximum distance, beyond this we do not consider it to be an effect of the one halo term

        # Apply radial mask
        relative_COMs = relative_COMs[radial_mask]
        rel_dist = rel_dist[radial_mask]
        relative_vels = relative_vels[radial_mask, :]
        HostMasses = HostMasses[radial_mask]

        # Some datapoints have such velocities as to incur overflow errors (exponential of -0.5v is returned as 0), so we prune those
        # HOWEVER we prune it in such a way that the entire velocity vector is thrown, otherwise we get a lot of issues later on
        # TODO: verify with Marcel that this is ok
        # usable_mask = np.full_like(HostMasses, True, dtype = bool)
        usable_mask = np.invert(np.any(np.exp((-0.5 * np.square(relative_vels)) / (2 * np.min((GLOBAL_PRIOR_RANGE[0][1], GLOBAL_PRIOR_RANGE[1][1]))**2)) == 0, axis = 1))
        if verbose: print(f'There are {usable_mask.sum()} usuable velocity vectors, that is {usable_mask.sum()/HostMasses.size * 100:.1f}%, removing the rest...')

        with h5py.File(filename, 'w') as file:
            file.create_dataset('rel_pos', data  = relative_COMs[usable_mask], dtype = np.float32)
            file.create_dataset('rel_dist', data = rel_dist[usable_mask], dtype = np.float32)
            file.create_dataset('rel_vels', data  = relative_vels[usable_mask, :], dtype = np.float32)
            file.create_dataset('mass', data = HostMasses[usable_mask], dtype = np.float32)

    
    def create_catalogue(self, massbin: Tuple[np.float32,np.float32], filename: str):
        """ Create a catalogue of relative velocities and positions in a given massbin.

        Args:
            massbin (Tuple[np.float32,np.float32]): Massbin in units of 10^10 Msol. Lower, upper.
            filename (str): string specifying the full path to which the data is saved (must end on .hdf5)
        """
        self.lower_mass, self.upper_mass = massbin
        self.filename = filename

        # Select the relevant mass range, since the lowest is 10**12 Msol and particle mass is 10^9 
        # we do not need to explicitly filter for number of particles, it will always be at least 1000
        mass_mask = _make_mass_mask(self.SOMass, self.lower_mass, self.upper_mass) #functions as a central mask implicitly
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

                mass_mask = _make_mass_mask(self.SOMass, lower_mass, upper_mass) #functions as a central mask implicitly

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
    def __init__(self, PATH: str, initial_param_file: str = None, loglambda: bool = False, load: bool= False):
        """_summary_

        Args:
            PATH (str): Path to hdf5 file specifying the data in a given massbin
            initial_param_file (str): path to .json file holding initial parameter values for the fitting process. If None, empirically decent default values are used. Defaults to None.
            TODO: finish
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

        if load:
            self.load_data()

    def load_data(self):
        # Load in the data
        with h5py.File(self.PATH, 'r') as handle:
            self.rel_pos = handle['rel_pos'][:]
            self.rel_vels = handle['rel_vels'][:]
        
        # Some calculations to store for marginal speed-up in the MCMC process
        self.rel_sq_dist = np.square(self.rel_pos).sum(axis = 1)
        # self.rel_vel_sq = np.square(self.rel_vels) * -0.5

    @staticmethod
    def _fit_modified_gaussian_emcee(data, bins: int | Callable, initial_guess: dict, log_likelihood_func, use_binned = False,
                                     nwalkers = 32, nsteps = 5000, param_labels = [r'$\sigma_1$',r'$\sigma_2$',r'$\lambda$'],
                                     plot = True, verbose = False, filename = 'Mfit', save_params = False, loglambda = False, **kwargs):
        if kwargs['timeit']: now = time()


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

        # Perturb the best found parameter set for an error estimate
        param_dict = perturb_around_likelihood(params = best_params, 
                                               log_likelihood_func = lambda x: log_likelihood_func(x, data, loglambda, kwargs['single_gauss']), 
                                               single_gauss = kwargs['single_gauss'])
 
        # Sometimes, sigma_1 is seen as the small contribution. For consistency we flip this and break the prior
        if param_dict['sigma_1'] < param_dict['sigma_2'] and kwargs['flip_sigmas']:
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
        
        if kwargs['timeit']: 
            duration = time() - now
            return param_dict, duration

        return param_dict
        # return samples

    def fit_to_catalogued_bin(self, rbin: np.array, datapath: str, **kwargs):
        #datapath should go to massbin folder with hdf5 files

        # only flip the sigmas if the argument
        if f'r_{rbin[0]:.2f}-{rbin[1]:.2f}' not in kwargs['flip_bins']:
            kwargs['flip_sigmas'] = False
        else:
            kwargs['flip_sigmas'] = True
    
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

        output = self._fit_modified_gaussian_emcee(data = masked_data, initial_guess = self.initial_param_dict, log_likelihood_func = likelihood_func,
                                                        filename = filename, rbin = rbin, **kwargs)

        # extra split needed if timed        
        if kwargs['timeit']: 
            output, duration = output
            with open('timing.txt', 'a') as tfile:
                tfile.write(f'{self.lower_mass}-{self.upper_mass},{rbin[0]:.2f}-{rbin[1]:.2f},{duration}\n')
                

        if kwargs['verbose']:
            tqdm.write(f'Radial bin {rbin[0]:.2f} - {rbin[1]:.2f} completed.')

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


class ONEHALO_joint_fitter:
    def __init__(self, 
                 PATH: str = '/disks/cosmodm/vdvuurst/data/Onehalo_M_12-15.5_subsampled.hdf5',
                 init_condition_path: str = '/disks/cosmodm/vdvuurst/data/onehalo_joint_initial_conditions'
                 ):
        self.PATH = PATH
        self.init_condition_path = init_condition_path

        with h5py.File(self.PATH) as handle:
            self.halo_masses = handle['mass'][:]
            self.rel_vels = handle['rel_vels'][:]
            self.rel_dist = handle['rel_dist'][:]
        
        # pre calculate this
        self.min_half_v_sq_arr = -0.5 *np.square(self.rel_vels) 

        self.Ndata = self.rel_vels.shape[0] # Take the 0 of the shape since shape[1] = 3 (3 indep. velocity points per datum)

    @staticmethod
    def param_info(function_combi):
        # Generate parameter information lists (pointers) given a function combination
        n_params_r = []
        n_params_m = [] # every list is for a double gauss parameter, containing a number of parameters needed to parametrize an r-parameter in terms of M for every r-parameter
        for param_combi in function_combi:
            _, m_parametrizations = param_combi # unpack the function combination for this specific parameter
            n_params_r.append(len(m_parametrizations))
            n_params_m.append([len(signature(m_func).parameters) - 1 for m_func in m_parametrizations])

        n_params_m = flatten(n_params_m)

        return n_params_r, n_params_m, sum(n_params_m)

    @staticmethod
    def split_parameters(params, n_params_m):
        # Split array of parameters into jagged array to be used with pointers
        return np.split(params, np.cumsum(n_params_m)[:-1]) 

    def get_double_gauss_parameters(self, split_params, function_combi, n_params_r,
                                    halo_masses = None, rel_dist = None):
        # Add so that we can also use this function for subsets of the data
        if halo_masses is None:
            halo_masses = self.halo_masses
        if rel_dist is None:
            rel_dist = self.rel_dist

        r_pointers = np.concat(([0],np.cumsum(n_params_r))) # indexers for the r parameters
        
        #TODO: do I want to move this outside? to save time on initializing this every time
        double_gauss_params = np.zeros((3, halo_masses.size), dtype = np.float32)

        for i in range(3): # iterate over double gauss parameters
            # Isolate the r_function for the given DG parameter and the functions of M that paramettrize that further
            param_r_func, param_m_funcs = function_combi[i] 

            # Get the parameter values for the function of r (by applying the mass functions). Has shape (N_{rparams}, Ndata)
            parameters_for_r_func = [param_m_funcs[k](halo_masses, *split_params[j]) for k,j in enumerate(range(r_pointers[i], r_pointers[i+1]))]

            # Now apply those parameter values for all distances. Returns array of shape Ndata
            param_values = param_r_func(rel_dist, *parameters_for_r_func) 
            double_gauss_params[i] = param_values

        return double_gauss_params
    
    def get_joint_likelihood(self, params, n_params_m, n_params_r, function_combi):
        # TODO make function input? idk if that fucks with mcmc, i think it does
        split_params = self.split_parameters(params, n_params_m)
        
        # Update DG parameters from parameter set
        DG_params = self.get_double_gauss_parameters(split_params, function_combi, n_params_r)
        
        # Calculate the likelihood of the data + parameter set
        L = mod_gaussian_log_likelihood_vec(DG_params, self.min_half_v_sq_arr)

        return L

    #WRAPPER
    def fit_function_combi_to_data(self, function_combi: list, function_combi_names: list, combi_number: int,
                                    nwalkers: int = 50, nsteps: int = 500,
                                    filepath: str = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/joint_subsample',
                                    **kwargs) -> None:
        n_params_r, n_params_m, ndim = self.param_info(function_combi)

        #read in initial conditions and mcmc step sizes form pre-ran conditions (see ONEHALO_initial_conditions.py)
        #and add small amount of noise for every walker
        initial_params, MCMC_scales = np.load(os.path.join(self.init_condition_path, f'function_combi_{combi_number}.npy'))
        noise = np.random.normal(0, MCMC_scales, size = (nwalkers, MCMC_scales.size))
        initial_params = initial_params[np.newaxis, :] + noise

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.get_joint_likelihood,
                                         args = (n_params_m, n_params_r, function_combi))
        sampler.run_mcmc(initial_params, nsteps, progress = kwargs['verbose'])
        
        samples = sampler.get_chain()
        likelihoods = sampler.get_log_prob()
        best_arg = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
        best_likelihood = likelihoods[best_arg]
        best_params = np.array([samples[*best_arg, i] for i in range(ndim)])
       
        #TODO: update perturb param function to work for joint model

        BIC_score = BIC(best_likelihood, self.Ndata, ndim)

        # Create dict of all relevant information of the fit

        param_dict = {'parameters':list(best_params), 'likelihood':best_likelihood,
                       'BIC':BIC_score, 'functional_form':function_combi_names,
                       'nwalkers': nwalkers, 'nsteps': nsteps}
    
        # Save best parameter dictionary
        with open(os.path.join(filepath, f'function_combi_{combi_number}'), 'w') as f:
            dump(param_dict, f, indent = 1)

        if kwargs['plot']:
            fpath_for_plot = '/disks/cosmodm/vdvuurst/figures/onehalo_joint/subsampled' if 'subsampled' in filepath else '/disks/cosmodm/vdvuurst/figures/onehalo_joint'
            self.plot_in_bin(best_params, function_combi, combi_number, n_params_r, n_params_m,
                            BIC_score, kwargs['mbin'], kwargs['rbin'], show = kwargs['show_plot'],
                            filepath = fpath_for_plot)


    def plot_in_bin(self, best_params: np.ndarray, function_combi: list, combi_number: int,
                    n_params_r: int, n_params_m: list,
                    BIC_score: np.float32, mbin: list | tuple, rbin: list | tuple, show: bool = False,
                    filepath: str = '/disks/cosmodm/vdvuurst/figures/onehalo_joint/subsampled') -> None:
        
        # Define the bin
        mbin_mask = _make_mass_mask(self.halo_masses, *mbin)
        rbin_mask = _make_radial_mask(self.rel_dist, *rbin)
        bin_mask = np.logical_and(mbin_mask, rbin_mask)

        # Get binned data
        vel_data_in_bin = self.rel_vels[bin_mask].flatten() # first apply mask to all 3-vectors, then flatten
        min_half_v_sq_in_bin = self.min_half_v_sq_arr[bin_mask]
        masses_in_bin = self.halo_masses[bin_mask]
        rel_dist_in_bin = self.rel_dist[bin_mask]

        bins = rice_bins(vel_data_in_bin.size)
        filename = os.path.join(filepath, f'function_combi_{combi_number}')
        mkdir_if_non_existent(filename)
        filename = os.path.join(filename, f'{str_from_mbin(mbin)}_{str_from_rbin(rbin)}_fit.png')

        DG_params = self.get_double_gauss_parameters(self.split_parameters(best_params, n_params_m), function_combi,
                                                      n_params_r, masses_in_bin, rel_dist_in_bin)

        # Plotting
        fig, ax = plt.subplots(figsize = (7,7))
        ax.set_xlabel('Velocity difference v', fontsize=16)
        ax.set_ylabel('Number of galaxies per v', fontsize=16)
        ax.tick_params(axis='both', which='major',length=6, width=2,labelsize=14)

        # Bin velocity histogram and plot it
        bin_heights, bin_edges = np.histogram(vel_data_in_bin, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width= bin_edges[1] - bin_edges[0] 
        bin_widths = np.diff(bin_edges)  # The width of each bin
        number_density = bin_heights / bin_widths  # Normalize by bin width
        hist_area=np.sum(bin_heights)
        ax.bar(bin_centers, number_density, width=bin_width, align='center', edgecolor = 'black')

        # Add BIC score in textbox
        ax.text(0.155, 0.83, f'BIC = {BIC_score:.2e}', transform=plt.gcf().transFigure,
                backgroundcolor='white',zorder=-1,
                bbox = {'boxstyle':'round','facecolor':'white'}, fontsize = 12)
        
        # Plot fitted distribution
        sigma_1_sq, sigma_2_sq = 2 * np.square(DG_params[0]), 2 * np.square(DG_params[1])
        one_min_lambda = 1 - DG_params[2]

        norm = 1 / (((one_min_lambda * DG_params[0]) + (DG_params[2] * DG_params[1]))* np.sqrt(2 * np.pi)) 

        # ax.plot(vel_data_in_bin, hist_area*mod_gaussian_vec_for_plot(min_half_v_sq_in_bin, sigma_1_sq, sigma_2_sq, DG_params[2], one_min_lambda),
        #         '-', label = f"Double Gaussian (joint)\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}",
        #         color='red')
        DAT = np.linspace(np.min(vel_data_in_bin),np.max(vel_data_in_bin), min_half_v_sq_in_bin.size).reshape(min_half_v_sq_in_bin.shape)

        plot_data =  hist_area * mod_gaussian_vec_for_plot(-0.5*np.square(DAT), sigma_1_sq, sigma_2_sq, DG_params[2], one_min_lambda, norm)
        print(plot_data)
        ax.plot(vel_data_in_bin, plot_data, '-', label = f"Double Gaussian (joint)\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}",
                color='red')
        
        # ax.scatter(DAT, hist_area* mod_gaussian_vec_for_plot(min_half_v_sq_in_bin, sigma_1_sq, sigma_2_sq, DG_params[2], one_min_lambda, norm),
        #             label = f"Double Gaussian (joint)\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}",
        #             color='red')

        ax.legend(fontsize=12.5, loc="upper right")
        
        if not show:
            fig.savefig(filename, dpi=200)
            plt.close()
        else:
            plt.show()