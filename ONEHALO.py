import numpy as np
import h5py
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from functions import *
from onehalo_plotter import plot_distribution_gaussian_mod
from json import dump
from json import load as jsonload
import emcee
from corner import corner
from functional_forms import * # also runs some code to create the functional_form catalogue in the all_combis variable
from time import time
from MCMC import MCMC

latex_formatter = {'sigma_1':r'$\sigma_1$', 'sigma_2': r'$\sigma_2$', 'lambda':r'$\lambda$', 'loglambda':r'$\log_{10}\left(\lambda\right)$'}

def _make_mass_mask(mass: np.ndarray, m_min: np.float32, m_max: np.float32, logmass: bool = False) -> np.ndarray:
    if not logmass:
        if m_min in [0,-1,np.nan, None]:
            return (mass <= 10**m_max)
        elif m_max in [0,-1,np.nan, None]:
            return (10**m_min <= mass)
        return (10**m_min <= mass) & (mass <= 10**m_max) 
   
    else:
        if m_min in [0,-1,np.nan, None]:
            return (mass <= m_max)
        elif m_max in [0,-1,np.nan, None]:
            return (m_min <= mass)
        return (m_min <= mass) & (mass <= m_max) 

def _make_radial_mask(radii: np.ndarray, r_min: np.float32, r_max: np.float32) -> np.ndarray:
    return (r_min <= radii) & (r_max >= radii)

def str_from_mbin(mbin):
    return f'M_1{mbin[0]}-1{mbin[1]}'

def str_from_rbin(rbin):
    return f'r_{rbin[0]:.2f}-{rbin[1]:.2f}'

class ONEHALO:
    def __init__(self, PATH: str):
        self.PATH = PATH # Path to SOAP Catalogue

        with h5py.File(self.PATH, "r") as handle:
            self.IsCentral = handle["InputHalos/IsCentral"][:].astype(bool) 
            self.HaloIndices = np.arange(self.IsCentral.size)
            self.COMvelocity = handle["ExclusiveSphere/100kpc/CentreOfMassVelocity"][:]
            self.SubHaloStellarMass = handle["ExclusiveSphere/100kpc/StellarMass"][:] # TODO: impose minimum, make histogram , je verwacht powerlaw met exponeential cut-off, cut bij de piek van de power law
            self.HaloCatalogueIndex = handle["InputHalos/HaloCatalogueIndex"][:]
            self.HostHaloIndex = handle["SOAP/HostHaloIndex"][:] # -1 for centrals
            self.COM = handle["ExclusiveSphere/100kpc/CentreOfMass"][:]
            self.SOMass = handle['SO/200_mean/TotalMass'][:]
            self.Rvir = handle['SO/200_mean/SORadius'][:]

            self.boxsize = handle['Header'].attrs['BoxSize'][0]

        self.COM = self.COM % self.boxsize #if any coordinate value is negative or larger than box size - map into the box

        # ~IsCentral picks out sattelites, this picks out all centrals !that host sattelites! and how many sattelites they host
        self.HostHaloIDs, self.subhalos_per_host_tot = np.unique(self.HostHaloIndex[~self.IsCentral], return_counts = True) 

    def create_full_dataset(self, mass_range: Tuple[np.float32,np.float32], filename: str,
                             r_max: np.float32 = 2.5, verbose: bool = False, create_subsample: bool = True):
        """ Create a full catalogue of halo mass, relative velocities and positions in a given mass range (so NOT binning!).

        Args:
            mass_range (Tuple[np.float32,np.float32]): Massbin in units of 10^10 Msol. Lower, upper.
            filename (str): string specifying the full path to which the data is saved (must end on .hdf5)
            r_max (float) : maximum radius to allow. Defualts to 2.5
            verbose (bool): whether or not to output intermittent updates. Defaults to False.
            create_subsample(bool, optional): whether to create a uniformly subsampled (1%) dataset as well. Defaults to True.

        """
        lower_mass, upper_mass = mass_range
        mass_mask = _make_mass_mask(self.SOMass, lower_mass, upper_mass)
                
        HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central
        
        # From the catalogue of sattelite hosting haloes, select only those we know actually HAVE subhaloes and thus are relevant for this dataset
        Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
        
        subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask]
        HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] 

        SatteliteMask = np.isin(self.HostHaloIndex, HostIndices) 
        if verbose: print('Sattelites found')

        HostCOMs, HostVels = self.COM[HostIndices], self.COMvelocity[HostIndices]
        HostMasses = self.SOMass[HostIndices]

        SatSorter = np.argsort(self.HostHaloIndex[SatteliteMask]) # sorting so that we are sure to compare every sattelite to the right host below
        SatCOMs, SatVels = self.COM[SatteliteMask][SatSorter], self.COMvelocity[SatteliteMask][SatSorter]
        if verbose: print('Sattelites sorted')

        # Get the virial radii for all sattelites in order
        self.SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)
        relative_COMs = SatCOMs - np.repeat(HostCOMs, subhalos_per_host, axis = 0)
        relative_vels = SatVels - np.repeat(HostVels, subhalos_per_host, axis = 0)
        
        # Get the virial radii for all sattelites in order and transform coordinates
        SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)
        rel_dist = np.sqrt(np.square(relative_COMs).sum(axis = 1)) / SatRvirs 
        relative_COMS = relative_COMS / SatRvirs
        HostMasses = np.repeat(HostMasses, subhalos_per_host, axis = 0)

        if verbose: print('All sattelite data found, applying masks...')

        radial_mask = rel_dist < r_max # We set a maximum distance, beyond this we do not consider it to be an effect of the one halo term

        relative_COMs = relative_COMs[radial_mask]
        rel_dist = rel_dist[radial_mask]
        relative_vels = relative_vels[radial_mask, :]
        HostMasses = np.log10(HostMasses[radial_mask]) # Save in dex!

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
        
                
        if create_subsample:
            masses = HostMasses[usable_mask]
            rel_pos = relative_COMs[usable_mask]
            rel_dist=  rel_dist[usable_mask]
            rel_vels = relative_vels[usable_mask, :]

            subsample_idx = np.random.choice(masses.size, size = masses.size // 100, replace = False)

            subsample_mass = masses[subsample_idx]
            subsample_dist = rel_dist[subsample_idx]
            subsample_pos = rel_pos[subsample_idx, :]
            subsample_vels = rel_vels[subsample_idx, :]

            subsample_filename = filename.replace('.hdf5', '_subsampled.hdf5')
            with h5py.File(subsample_filename, 'w') as file:
                file.create_dataset('rel_pos', data  = subsample_pos, dtype = np.float32)
                file.create_dataset('rel_vels', data  = subsample_vels, dtype = np.float32)
                file.create_dataset('mass', data = subsample_mass, dtype = np.float32)
                file.create_dataset('rel_dist', data = subsample_dist, dtype = np.float32)

    
    def create_catalogue(self, massbin: Tuple[np.float32,np.float32], filename: str):
        """ Create a catalogue of relative velocities and positions in a given massbin.

        Args:
            massbin (Tuple[np.float32,np.float32]): Massbin in units of 10^10 Msol. Lower, upper.
            filename (str): string specifying the full path to which the data is saved (must end on .hdf5)
        """
        self.lower_mass, self.upper_mass = massbin
        self.filename = filename

        mass_mask = _make_mass_mask(self.SOMass, self.lower_mass, self.upper_mass)
        self.mass_mask = mass_mask
        HostIndices = self.HaloIndices[mass_mask] #if it has non-zero mass it must be a central
        
        # From the catalogue of sattelite hosting haloes, select only those we know actually HAVE subhaloes and thus are relevant for this dataset
        Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
        
        subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask]
        HostIndices = self.HostHaloIDs[Relevant_Hosts_mask]

        SatteliteMask = np.isin(self.HostHaloIndex, HostIndices) #pick out all the sattelites from relevant hosts

        # Get the host COM pos and vel, same for sattelites
        HostCOMs, HostVels = self.COM[HostIndices], self.COMvelocity[HostIndices]
        SatSorter = np.argsort(self.HostHaloIndex[SatteliteMask]) # sorting so that we are sure to compare every sattelite to the right host below
        SatCOMs, SatVels = self.COM[SatteliteMask][SatSorter], self.COMvelocity[SatteliteMask][SatSorter]
        
        relative_COMs = SatCOMs - np.repeat(HostCOMs, subhalos_per_host, axis = 0)
        relative_vels = SatVels - np.repeat(HostVels, subhalos_per_host, axis = 0)

        # Get the virial radii for all sattelites in order and transform coordinates
        self.SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)
        relative_COMs = relative_COMs / self.SatRvirs 

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

                mass_mask = _make_mass_mask(self.SOMass, lower_mass, upper_mass)

                HostIndices = self.HaloIndices[mass_mask]          

                Relevant_Hosts_mask = np.isin(self.HostHaloIDs,HostIndices)
                HostIndices = self.HostHaloIDs[Relevant_Hosts_mask] 

                subhalos_per_host = self.subhalos_per_host_tot[Relevant_Hosts_mask] 
                SatRvirs = np.repeat(self.Rvir[HostIndices], subhalos_per_host)

                rel_sq_dist = np.square(rel_pos).sum(axis = 1) / np.square(SatRvirs)

        elif r_unit == 'Mpc':
            rel_sq_dist = np.square(rel_pos).sum(axis = 1)

        radial_mask = (radial_bin[0]**2 <= rel_sq_dist) & (rel_sq_dist <= radial_bin[1]**2)
        masked_data = rel_vels[radial_mask]

        if masked_data.size < 100: 
            return True # return that bin has insufficient data

        with h5py.File(rad_filename, 'w') as file:
            file.create_dataset('rel_vels', data = masked_data, dtype = np.float32)

class ONEHALO_fitter:
    def __init__(self, PATH: str, initial_param_file: str = None, loglambda: bool = False, load: bool= False, enforce_sigma_2_smaller: bool = False):
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

        self.enforce_sigma_2_smaller = enforce_sigma_2_smaller

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
            pos_ranges = np.copy(GLOBAL_PRIOR_RANGE).T
            pos_ranges[1][1] = pos_ranges[1][0] / 4 # set sigma_2 to at most quarter the max of sigma_1
            pos_ranges[1][2] = 0.5 # set the upper bound for lambda to 0.5 instead of 1.
            pos_ranges = pos_ranges[..., np.newaxis]
            pos = np.random.uniform(*pos_ranges, size = (ndim, nwalkers))
            # Assure sigma_1 > sigma_2 for every walker at the starting point
            mask = pos[0] < pos[1]
            pos[0][mask], pos[1][mask] = pos[1][mask], pos[0][mask]

        loglambda_str = '_log_lambda' if loglambda else ''

        # If bins is passed as a function (e.g. ricebins in functions.py), calculate the amount of bins
        if hasattr(bins, '__call__'):
            bins = bins(data.size)

        step_sizes = np.array([20., 15., 5e-3])[:, np.newaxis]
        if not use_binned:
            sampler = MCMC(nwalkers, log_likelihood_func,
                            args = (data, loglambda, kwargs['single_gauss'], kwargs['enforce_sigma_2_smaller']), step_sizes= step_sizes)
            sampler.run_mcmc(pos, nsteps, verbose = verbose)
        else:
            # Compute histogram
            bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
            sampler = MCMC(nwalkers, log_likelihood_func, args = (bin_edges, bin_heights), step_sizes= step_sizes)
            sampler.run_mcmc(pos, nsteps, verbose = verbose)

        burnin = 50 # does not matter so much

        samples = sampler.get_chain()
        likelihoods = sampler.get_likelihoods()
        # likelihoods = sampler.get_log_prob()
        best_arg = np.unravel_index(np.argmin(likelihoods), likelihoods.shape)
        best_likelihood = likelihoods[best_arg]
        best_params = np.array([samples[i, *best_arg] for i in range(ndim)])

        # We refine the found optimum with a BFGS minimization to get a marginally better result for little time investment
        ll_func_for_minimize = lambda x: log_likelihood_func(x, data, loglambda, kwargs['single_gauss'], kwargs['enforce_sigma_2_smaller'])
        new_best_params = minimize(ll_func_for_minimize, x0 = best_params, bounds = pos_ranges.squeeze().T).x

        # Check that we are still in the prior range and didn't diverge out, if we did keep the old result
        prior_reached = log_prior(new_best_params)
        if np.isfinite(prior_reached):
            best_params = new_best_params
            best_likelihood = ll_func_for_minimize(new_best_params)
        else:
            best_likelihood = likelihoods[best_arg]

        if not kwargs['single_gauss']:
            if not kwargs['is_rerun']:
                # Create or open the log-file to track poor fits which we only do in initial run, NOT rerun
                with open('./logs/bad_fits_onehalo.txt', 'a+') as f:
                    # We want to run these again after the batch job with different constraints, so note down the mass and rad bin that is bad
                    if best_params[2] == 0.5:
                        mstr, rstr = filename.split('/')[-2:]
                        f.write(f'{mstr}/{rstr}\n')

            else:
                # In these cases, the fit is best described by a single gaussian but only sigma_1 or sigma_2 then matters, not both
                # So find out which and select it. These cases will regardless be re-ran as per the check above, but even for the re-run this may happen
                if best_params[2] == 0.5 or np.abs(best_params[0] - best_params[1]) <= 10:
                    test_params = best_params.copy()
                    test_params[2] = 0.
                    lambda_0_likelihood = ll_func_for_minimize(test_params)

                    test_params[2] = 1.
                    lambda_1_likelihood = ll_func_for_minimize(test_params)

                    if lambda_0_likelihood < lambda_1_likelihood:
                        test_params[2] = 0.
                    
                    else:
                       test_params[1], test_params[0] = test_params[0], test_params[1] #flip sigmas, EVEN IF that means sigma1 < sigma2 now since sigma1 is always the more important one

                    best_params = test_params.copy()


        # Perturb the best found parameter set for an error estimate
        param_dict = perturb_around_likelihood(params = best_params, 
                                               log_likelihood_func = lambda x: -1 *log_likelihood_func(x, data, loglambda, kwargs['single_gauss'], False), #for the error measure, we do not want to enforce sigma_1 > sigma_2, otherwise the error is just the difference between the values 
                                               single_gauss = kwargs['single_gauss'])
 
        if kwargs['flip_sigmas']:
            #Switch sigma_1 and sigma_2 (and errors) and flip lambda to 1 - lambda (thus switching the errors)
            param_dict['sigma_1'], param_dict['sigma_2'] = param_dict['sigma_2'], param_dict['sigma_1']
            param_dict['errors'][0], param_dict['errors'][1] = param_dict['errors'][1], param_dict['errors'][0]
            param_dict['lambda'] = 1 - param_dict['lambda']
            param_dict['errors'][2][0], param_dict['errors'][2][1] = param_dict['errors'][2][1], param_dict['errors'][2][0]


        # Add some more diagnostics to the dictionaries for later use and ease
        param_dict['nwalkers'] = nwalkers
        param_dict['nsteps'] = nsteps
        param_dict['N'] = data.size
        param_dict['likelihood'] = best_likelihood #minlogL

        if plot:
            # Walkers
            fig, axes = plt.subplots(ndim + 1, figsize=(12, 10), sharex=True)

            for i in range(ndim):
                ax:plt.Axes = axes[i]
                ax.plot(samples[i, ...].T, alpha=0.3)
                ax.set_xlim(0, samples.shape[-1])
                # ax.set_xlim(0, len(samples))
                ax.set_ylabel(param_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

                ymin, ymax = ax.get_ylim()
                ax.vlines(burnin, ymin, ymax, colors = 'black', linestyles = '--')
                ax.set_ylim(ymin, ymax)
  
            likelihoods_plot = np.log10(likelihoods).T
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
                # based on emcee implementation of flat
                v = sampler.get_chain(discard=burnin, thin=15)
                s = [v.shape[2], v.shape[0]]
                s[0] = np.prod((v.shape[2], v.shape[1]))
                flat_samples = v.T.reshape(s)

                # flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True) 
                # print(flat_samples.shape)
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
                    plot_distribution_gaussian_mod(double_gaussian_loglambda, param_dict, data, bins=bins, distname='Double Gaussian', filename = filename + f'_fit{loglambda_str}.png', loglambda = True)
                else: 
                    plot_distribution_gaussian_mod(double_gaussian_loglambda, param_dict, data, bins=bins, distname="Single Gaussian", filename = filename + f'_fit{loglambda_str}.png', loglambda = True)
            else:
                if not kwargs['single_gauss']: 
                    plot_distribution_gaussian_mod(double_gaussian, param_dict, data, bins=bins, distname='Double Gaussian', filename = filename + f'_fit{loglambda_str}.png', loglambda = False)
                else: 
                    plot_distribution_gaussian_mod(double_gaussian, param_dict, data, bins=bins, distname="Single Gaussian", filename = filename + f'_fit{loglambda_str}.png', loglambda = False, single_gauss = True)
        
        if kwargs['timeit']: 
            duration = time() - now
            return param_dict, duration

        return param_dict
        # return samples

    def fit_to_catalogued_bin(self, rbin: np.array, datapath: str, **kwargs):
        #datapath should go to massbin folder with hdf5 files
        # always flip the sigmas in these specific bins
        if f'r_{rbin[0]:.2f}-{rbin[1]:.2f}' in kwargs['flip_bins'] and self.lower_mass in [12.0, 12.5]:
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

        # keep everything variable except the enforcement of sigma_2 < sigma_1 in the likelihood
        # likelihood_func = lambda x,y,z,w :double_gaussian_log_likelihood(x,y,z,w, enforce_sigma_2_smaller = self.enforce_sigma_2_smaller)
        likelihood_func = double_gaussian_log_likelihood

        output = self._fit_modified_gaussian_emcee(data = masked_data, initial_guess = self.initial_param_dict, log_likelihood_func = likelihood_func,
                                                        filename = filename, rbin = rbin, enforce_sigma_2_smaller = self.enforce_sigma_2_smaller, **kwargs)

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

            # in case of non-binned L: keep everything variable except the enforcement of sigma_2 < sigma_1 in the likelihood
            likelihood_func = double_gaussian_log_likelihood if not use_binned else double_gaussian_log_likelihood_binned

            output = self._fit_modified_gaussian_emcee(data = masked_data, initial_guess = self.initial_param_dict, log_likelihood_func = likelihood_func,
                                                            filename = filename, use_binned = use_binned, rbin = rbin, enforce_sigma_2_smaller = self.enforce_sigma_2_smaller,
                                                              **kwargs)
            
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
                
                    likelihood_func = double_gaussian_log_likelihood_binned if use_binned else double_gaussian_log_likelihood
                    result, err = self._fit_modified_gaussian_emcee(self.rel_vels, bins, self.initial_param_dict, double_gaussian_log_likelihood,
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
            
                likelihood_func = double_gaussian_log_likelihood_binned if use_binned else double_gaussian_log_likelihood
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
           
            likelihood_func = double_gaussian_log_likelihood_binned if use_binned else double_gaussian_log_likelihood
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


def param_info(function_combi: list):
    """ Generate parameter information lists (pointers) given a function combination. 
        Example: if sigma_1 is parametrized in r as a third order polynomial, the first entry in the n_params_r list will be 4 (the amount of parameters in the functional form). 
        If the first parameter of this function of r is then sub-parameterized as a linear function, the first instance of the n_params_m list will be 2 (the amount of parameters in that functional form).

    Args:
        function_combi (list): list of function instances. Should be as generated in functional_forms.py.

    Returns:
        list: n_params_r, a list of integers specifying the amount of parameters for the parameterization of a double-gauss parameter as a function of r.
        list: n_params_m, a list of integers specifying the amount of parameters for the sub-parametetrization of a parameter of the function of r. 
        int: The total number of parameters. 
    """
    n_params_r = []
    n_params_m = [] 
    for param_combi in function_combi:
        _, m_parametrizations = param_combi # unpack the function combination for this specific parameter
        n_params_r.append(len(m_parametrizations))
        n_params_m.append([len(signature(m_func).parameters) - 1 for m_func in m_parametrizations])

    n_params_m = flatten(n_params_m)

    return n_params_r, n_params_m, sum(n_params_m)

class ONEHALO_MADD_fitter:
    def __init__(self, 
                 PATH: str = '/disks/cosmodm/vdvuurst/data/Onehalo_M_12-15.5_subsampled.hdf5',
                 ):
        """

        Class that loads in the one-halo data and is able to fit using an MCMC procedure. Different initial conditions are possible.

        Args:
            PATH (str, optional): Path to the data, also controls whether to use subsampled data or not. Defaults to '/disks/cosmodm/vdvuurst/data/Onehalo_M_12-15.5_subsampled.hdf5'.
        """
        
        self.PATH = PATH

        with h5py.File(self.PATH) as handle:
            self.halo_masses = handle['mass'][:]
            self.rel_vels = handle['rel_vels'][:]
            self.rel_dist = handle['rel_dist'][:]
        
        self.min_half_v_sq_arr = -0.5 *np.square(self.rel_vels) 

        self.Ndata = self.rel_vels.shape[0] # Take the 0 of the shape since shape[1] = 3 (3 indep. velocity points per datum)

    @staticmethod
    def split_parameters(params, n_params_m):
        # Split array of parameters into jagged array to be used with pointers created by param_info()
        return np.split(params, np.cumsum(n_params_m)[:-1]) 

    def get_double_gauss_parameters(self, split_params:list, function_combi:list, n_params_r:list,
                                    halo_masses: np.ndarray | None = None, rel_dist: np.ndarray | None = None,
                                    upper_sigma_bound: float | None = None, flip_sigmas: bool = True) -> np.ndarray:
        
        """ From a given set of parameters (split by the split_parameter() function) and corresponding function combination, generate the 
            double gauss parameter values for each datapoint. Note that sigma_1 > sigma_2 is explicitly enforced by flipping the values if need be,
            and changing the corresponding lambda values to 1 - lambda. Parameter values are clamped to roughly the GLOBAL_PRIOR_RANGE (defined in functions.py).
            Note that the upper_sigma_bound argument can control the upper bound for both sigma_1 and sigma_2 (used for initial conditions). Moreover note that lambda is clamped
            to 0.5 and not 1. 

            
        Args:
            split_params (list): Parameter values, split according to the split_parameters() function
            function_combi (list): list of function instances to be passed to the likelihood function. Should be as generated in functional_forms.py.
            n_params_r (list): integers specifying the amount of parameters for the parameterization of a double-gauss parameter as a function of r. As generated by split_parameters()
            halo_masses (np.array or None, optional): Halo masses in the data. Defaults to None, i.e. the whole dataset is used.
            rel_dist (np.array or None, optional): Radial distances from the COM in the data. Defaults to None, i.e. the whole dataset is used.
            upper_sigma_bound(int or None, optional): Controls the cap for the sigma parameters, if None will set it to the value in GLOBAL_PRIOR_RANGE. Only used for initial conditions. Defaults to None.


        Returns:
            np.ndarray: Shape (3, Ndata) array holding the parameter values for the double gauss model.
        """

        # we can also use this function for subsets of the data
        if halo_masses is None:
            halo_masses = self.halo_masses
        if rel_dist is None:
            rel_dist = self.rel_dist

        r_pointers = np.concat(([0],np.cumsum(n_params_r))) # indexers for the r parameters
     
        double_gauss_params = np.zeros((3, halo_masses.size), dtype = np.float32)

        for i in range(3): # iterate over double gauss parameters
            param_r_func, param_m_funcs = function_combi[i] 

            parameters_for_r_func = [param_m_funcs[k](halo_masses, *split_params[j]) for k,j in enumerate(range(r_pointers[i], r_pointers[i+1]))]

            param_values = param_r_func(rel_dist, *parameters_for_r_func) 
            double_gauss_params[i, :] = param_values

        if upper_sigma_bound is None:
            upper_sigma_bound = GLOBAL_PRIOR_RANGE[0][1]
            sigma_2_min = GLOBAL_PRIOR_RANGE[1][0]
        else: # i.e. upper_sigma_bound got a value which only happens during initial conditions finding
            sigma_2_min = 50.
      
        if flip_sigmas: 
            sigma_flip_mask = double_gauss_params[0] < double_gauss_params[1] # sigma_1 < sigma_2

            lower_bounds = np.array([GLOBAL_PRIOR_RANGE[0][0], sigma_2_min, GLOBAL_PRIOR_RANGE[2][0]])[:, np.newaxis]
            upper_bounds = np.array([upper_sigma_bound, upper_sigma_bound, 0.5])[:, np.newaxis]

        else: 
            lower_bounds = [np.repeat(GLOBAL_PRIOR_RANGE[0][0], double_gauss_params.shape[1]),
                            np.repeat(sigma_2_min, double_gauss_params.shape[1]),
                            np.repeat(GLOBAL_PRIOR_RANGE[2][0], double_gauss_params.shape[1])]
            
            upper_bounds = [np.repeat(upper_sigma_bound, double_gauss_params.shape[1]),
                            np.clip(double_gauss_params[0,:] - 5., 51., None), #clamp sigma_2 upper_bound (i.e. the corresponding value of sigma_1 - 5) to a minimum of 51
                            np.repeat(0.5, double_gauss_params.shape[1])] 
            
        double_gauss_params = np.clip(double_gauss_params, lower_bounds, upper_bounds)

        if flip_sigmas: 
            double_gauss_params[0][sigma_flip_mask], double_gauss_params[1][sigma_flip_mask] = double_gauss_params[1][sigma_flip_mask], double_gauss_params[0][sigma_flip_mask] 
            double_gauss_params[2][sigma_flip_mask] = 1 - double_gauss_params[2][sigma_flip_mask]

        return double_gauss_params
    
    def get_MADD_likelihood(self, params, n_params_m, n_params_r, function_combi, upper_sigma_bound: int | None = None, flip_sigmas: bool = True) -> np.float64:
        """ Given parameters, parameter info (as per the param_info() function) and a function combination, calculate the likelihood of the data.

        Args:
            params (array-like): Parameter values
            n_params_m (list): integers specifying the amount of parameters for the sub-parametetrization of a parameter of the function of r. 
            n_params_r (list): integers specifying the amount of parameters for the parameterization of a double-gauss parameter as a function of r. As generated by split_parameters()
            function_combi (list): list of function instances to be passed to the likelihood function. Should be as generated in functional_forms.py.
            upper_sigma_bound(int or None, optional): Controls the cap for the sigma parameters, if None will set it to the value in GLOBAL_PRIOR_RANGE. Only used for initial conditions. Defaults to None.

        Returns:
            float: The negative log-likelihood of the data with the given parameters and function combination.
        """
        split_params = self.split_parameters(params, n_params_m)
        
        # Update DG parameters from parameter set
        DG_params = self.get_double_gauss_parameters(split_params, function_combi, n_params_r, upper_sigma_bound = upper_sigma_bound, flip_sigmas = flip_sigmas)

        minlogL = double_gaussian_log_likelihood_vec(DG_params, self.min_half_v_sq_arr)

        return minlogL 

    def fit_function_combi_to_data(self, function_combi: list, function_combi_names: list, combi_number: int,
                                    nwalkers: int = 50, nsteps: int = 1000,
                                    n_params_r: list | None = None, n_params_m: list | None = None, ndim: int | None = None,
                                    filepath: str = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/MADD_subsample',
                                    init_condition_method: str = 'Nelder-Mead',
                                    **kwargs) -> None:
        """ Perform Markov-Chain Monte-Carlo procedure on a function combination, finding the optimum parameters that fit the data best.

        Args:
            function_combi (list): list of function instances to be passed to the likelihood function. Should be as generated in functional_forms.py.
            function_combi (list): strings specifying the names of functions passed in fucntion_combi. Should be as generated in functional_forms.py.
            combi_number (int): Function combination identifier. Should be as generated in functional_forms.py.
            nwalkers (int, optional):Number of walkers to use in the MCMC procedure. Defaults to 50.
            nsteps (int, optional): Number of steps to take in the MCMC procedure. Defaults to 1000.
            n_params_r (list | None, optional): integers specifying the amount of parameters for the parameterization of a double-gauss parameter as a function of r. As generated by split_parameters(). Defaults to None, i.e. calculated from scratch.
            n_params_m (list | None, optional)): integers specifying the amount of parameters for the sub-parametetrization of a parameter of the function of r.  Defaults to None, i.e. calculated from scratch.
            ndim (int | None, optional): Total number of parameters for the given function combination. Defaults to None, i.e. calculated from scratch.
            filepath (str, optional): Path specifying the directory in which the results may be stored. Defaults to '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/MADD_subsample'.
            init_condition_method (str, optional):  Name of the method used to find the initial conditions. This should be pre-ran via ONEHALO_initial_conditions.py! Defaults to 'Nelder-Mead'.

        """
        # if used like this, make it subsample_results-METHOD
        if 'subsample_results' in init_condition_method:
            # Extract the initial conditions method from the input and read in the results from the subsampled data fit as initial conditions
            init_condition_method = init_condition_method.split('-')[-1]
            self.init_condition_path = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/MADD_subsample/{init_condition_method}'
            with open(os.path.join(self.init_condition_path, f'function_combi_{combi_number}.json')) as f:
                subsample_dict = jsonload(f)
                initial_params = np.array(subsample_dict['parameters'])
            
            # From the initial conditions result, extract only the MCMC scales to use
            _, MCMC_scales = np.load(os.path.join('/disks/cosmodm/vdvuurst/data/onehalo_MADD_initial_conditions', init_condition_method, f'function_combi_{combi_number}.npy'))

        else:
            self.init_condition_path = f'/disks/cosmodm/vdvuurst/data/onehalo_MADD_initial_conditions/{init_condition_method}'
            initial_params, MCMC_scales = np.load(os.path.join(self.init_condition_path, f'function_combi_{combi_number}.npy'))


        if any([n_params_m is None, n_params_r is None, ndim is None]):
           n_params_r, n_params_m, ndim = param_info(function_combi)
        
        MCMC_scales =  MCMC_scales[:, np.newaxis]
        noise = np.random.normal(0, MCMC_scales, size = (MCMC_scales.size, nwalkers))

        initial_params = initial_params[:, np.newaxis] + noise

        sampler = MCMC(nwalkers = nwalkers,
                       likelihood_func = self.get_MADD_likelihood, 
                       args = (n_params_m, n_params_r, function_combi),
                       step_sizes = MCMC_scales)
        sampler.run_mcmc(initial_params, nsteps = nsteps, verbose = kwargs['verbose'])

        burnin = nsteps // 4
        samples = sampler.get_chain(discard = burnin)
        likelihoods = sampler.get_likelihoods(discard = burnin)

        if kwargs['verbose']:
            tqdm.write(f'NUMBER OF ACCEPTED STEPS IN THE MCMC PROCESS: {sampler.accepted_steps}')
            infmask = np.logical_or(likelihoods == -np.inf, likelihoods == np.inf)
            tqdm.write(f'THERE ARE {infmask.sum()} / {likelihoods.size} likelihoods that are inf or -inf')

        best_arg = np.unravel_index(np.nanargmin(likelihoods), likelihoods.shape) 
        best_params = np.array([samples[i, *best_arg] for i in range(ndim)])
       
        # We refine the found optimum with a BFGS minimization to get a marginally better result for little time investment
        ll_func_for_minimize = lambda x: self.get_MADD_likelihood(x, n_params_m, n_params_r, function_combi)
        best_params = minimize(ll_func_for_minimize, x0 = best_params).x
        best_likelihood = ll_func_for_minimize(best_params)

        with open('/disks/cosmodm/vdvuurst/logs/log.txt', 'a') as f:
            f.write(f'MCMC likelihood: {likelihoods[best_arg]} and after minimize: {best_likelihood}\n')

        BIC_score = BIC(-1*best_likelihood, self.Ndata, ndim) # -1 because BIC takes into account logL not minlogL

        param_dict = {'parameters':list(best_params), 'likelihood':float(best_likelihood),
                       'BIC': float(BIC_score), 'functional_form':[list(f) for f in function_combi_names], # need to cast to list for json serialization
                       'nwalkers': nwalkers, 'nsteps': nsteps, 'initial_condition_method':init_condition_method}

        with open(os.path.join(filepath, f'function_combi_{combi_number}.json'), 'w') as f:
            dump(param_dict, f, indent = 1)


        if kwargs['plot']:
            fpath_for_plot = '/disks/cosmodm/vdvuurst/figures/onehalo_MADD/subsampled' if 'subsampled' in filepath else '/disks/cosmodm/vdvuurst/figures/onehalo_MADD'
            try:
                self.plot_in_bin(best_params, function_combi, combi_number, n_params_r, n_params_m,
                            BIC_score, kwargs['mbin'], kwargs['rbin'], show = kwargs['show_plot'],
                            filepath = fpath_for_plot)
            except KeyError: # bins not specified in kwargs so we pass this plot
                pass
        
            cornerpath = os.path.join(fpath_for_plot, 'corner_plots')
            mkdir_if_non_existent(cornerpath)
            
            flat_samples = sampler.get_chain(discard=250, thin=5)
            flat_samples = flat_samples.reshape(np.array((flat_samples.shape)[::-1]))
            flat_samples = flat_samples.reshape((flat_samples.shape[0]*flat_samples.shape[1], flat_samples.shape[2]))

            alpha = [r'$\theta$'+ f'$_{i}$' for i in range(ndim)]
            fig = corner(flat_samples, labels = alpha, quiet = True,
                            quantiles=[0.16, 0.5, 0.84])
            fig.tight_layout()
            fig.savefig(os.path.join(cornerpath, f'function_combi_{combi_number}_corner.png'), dpi = 200)
            plt.close(fig)

            
    def plot_in_bin(self, best_params: np.ndarray, function_combi: list, combi_number: int,
                    n_params_r: int, n_params_m: list,
                    BIC_score: np.float32, mbin: list | tuple, rbin: list | tuple, show: bool = False,
                    filepath: str = '/disks/cosmodm/vdvuurst/figures/onehalo_MADD/subsampled') -> None:
        
        # Define the bin
        mbin_mask = _make_mass_mask(self.halo_masses, *mbin, logmass = True)
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

        norm = 1 / ((one_min_lambda * DG_params[0]) + (DG_params[2] * DG_params[1])) * SQRT2PI_FAC

        # ax.plot(vel_data_in_bin, hist_area*double_gaussian_vec_for_plot(min_half_v_sq_in_bin, sigma_1_sq, sigma_2_sq, DG_params[2], one_min_lambda),
        #         '-', label = f"Double Gaussian (MADD)\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}",
        #         color='red')
        DAT = np.linspace(np.min(vel_data_in_bin),np.max(vel_data_in_bin), min_half_v_sq_in_bin.size).reshape(min_half_v_sq_in_bin.shape)

        plot_data =  hist_area * double_gaussian_vec_for_plot(-0.5*np.square(DAT), sigma_1_sq, sigma_2_sq, DG_params[2], one_min_lambda, norm)

        ax.plot(DAT.flatten(), plot_data, '-', label = f"Double Gaussian (MADD)\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}",
                color='red')
        
        # ax.scatter(DAT, hist_area* double_gaussian_vec_for_plot(min_half_v_sq_in_bin, sigma_1_sq, sigma_2_sq, DG_params[2], one_min_lambda, norm),
        #             label = f"Double Gaussian (MADD)\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}",
        #             color='red')

        ax.legend(fontsize=12.5, loc="upper right")
        
        if not show:
            fig.savefig(filename, dpi=200)
            plt.close()
        else:
            plt.show()