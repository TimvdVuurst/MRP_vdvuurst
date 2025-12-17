import numpy as np
import h5py
from typing import Tuple
from tqdm import tqdm
from plotting import format_plot
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from functions import Romberg, modified_logspace
from json import load, dump
import emcee
from corner import corner
from scipy.special import gammainc
import warnings

#TODO: expand accordingly
latex_formatter = {'sigma_1':r'$\sigma_1$', 'sigma_2': r'$\sigma_2$', 'lambda':r'$\lambda$', 'log_lambda':r'$\log_{10}\left(\lambda\right)$'}


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

        relative_COMs = SatCOMs - np.repeat(HostCOMs, subhalos_per_host, axis = 0)
        relative_vels = SatVels - np.repeat(HostVels, subhalos_per_host, axis = 0)

        with h5py.File(self.filename, 'w') as file:
            file.create_dataset('rel_pos', data  = relative_COMs, dtype = np.float32)
            file.create_dataset('rel_vels', data  = relative_vels, dtype = np.float32)
    
    def create_radial_catalogue(self, radial_bin: Tuple[np.float32, np.float32], filename:str):
        mass_filename = filename.replace(f'/r_{radial_bin[0]:.2f}-{radial_bin[1]:.2f}','')

        with h5py.File(mass_filename, 'r') as handle:
            rel_vels = handle['rel_vels'][:] 
            rel_pos = handle['rel_pos'][:] 

        rel_sq_dist = np.square(rel_pos).sum(axis = 1)

        radial_mask = (radial_bin[0]**2 <= rel_sq_dist) & (rel_sq_dist <= radial_bin[1]**2)
        masked_data = rel_vels[radial_mask]

        if masked_data.size < 100: #TODO: check minimum, maybe even more - like 1000? make modifiable? also check in fit_to_radial_bins if changed
            return True

        with h5py.File(filename, 'w') as file:
            file.create_dataset('rel_vels', data = masked_data, dtype = np.float32)



## GENERAL FUNCTIONS 

def mod_gaussian(v, sigma1, sigma2, lambda_): #for regular (i.e. untransformed velocity) input
    # normalization done to break degeneracy between lambda_ and sigma_i as much as possible

    norm = 1 / (((1- lambda_) * sigma1 + lambda_ * sigma2)* np.sqrt(2 * np.pi))
    vsq = -1 * np.square(v) * 0.5
    return norm * ((1-lambda_) * np.exp(vsq / sigma1**2) + lambda_ * np.exp(vsq / sigma2**2))

def mod_gaussian_loglambda(v, sigma1, sigma2, lambda_): #for lambda in log scale
    lambda_10 = 10**lambda_
    norm = 1 / (((1- lambda_10) * sigma1 + lambda_10 * sigma2)* np.sqrt(2 * np.pi))
    vsq = -1 * np.square(v) * 0.5
    return norm * ((1-lambda_10) * np.exp(vsq / sigma1**2) + lambda_10 * np.exp(vsq / sigma2**2))

def mod_gaussian_integral(sigma1,sigma2,lambda_,x_i,x_f):
    integral, _ = Romberg(x_i, x_f, lambda x: mod_gaussian(x,sigma1,sigma2,lambda_)) 
    return integral
    
def log_prior(theta):
    sigma_1, sigma_2, lambda_ = theta
    if sigma_2 < sigma_1 <= 1500. and 1. <= sigma_2 < sigma_1 and 0 <= lambda_ <= 1.:
        return 0.0
    return -np.inf

def log_prior_loglambda(theta):
    sigma_1, sigma_2, lambda_ = theta
    if sigma_2 < sigma_1 <= 1500. and 1. <= sigma_2 < sigma_1 and -np.inf < lambda_ <= 0.:
        return 0.0
    return -np.inf

def mod_gaussian_log_likelihood_binned(params, bin_edges, bin_heights): #full with prior
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    sigma1, sigma2, lambda_ = params
    hist_area = np.sum(bin_heights) 
    fit_integral = 1.
    A = hist_area / fit_integral
    
    log_L = 0
    for i in range(1,len(bin_edges)):
        f_b = A * mod_gaussian_integral(sigma1,sigma2,lambda_,bin_edges[i-1],bin_edges[i])
        
        #penalize negative values and zero
        if f_b <= 0:
            return 10**11
        
        n_b = bin_heights[i-1] #* bin_width
        log_L += (n_b * np.log(f_b)) - f_b  # herein lies the only difference with neg_log_likelihood
    
    return log_L

def mod_gaussian_log_likelihood(params, data, log_lambda): #full with prior
    prior = log_prior_loglambda if log_lambda else log_prior
    lp = prior(params)
    if not np.isfinite(lp):
        return -np.inf

    mu_func = mod_gaussian_loglambda if log_lambda else mod_gaussian
    mu_i = mu_func(data, *params)
    mu_i[mu_i < 0] = 0 # cast negative values to 0, raises errors otherwise
    return np.sum(np.log(mu_i)) + 1. #+1. for integral

def mod_gaussian_neg_log_likelihood_binned(params, bin_edges, bin_heights):
    sigma1, sigma2, lambda_ = params
    hist_area = np.sum(bin_heights) 
    # I do not know where this comes from, integral of mod_gaussian dv from -inf to inf is 1.  
    # fit_integral = (sigma1 * np.sqrt(2 * np.pi)) *(3*sigma2 + 1 + 105*lambda_) 
    fit_integral = 1.
    A = hist_area / fit_integral
    
    neg_log_L = 0
    for i in range(1,len(bin_edges)):
        f_b = A * mod_gaussian_integral(sigma1,sigma2,lambda_,bin_edges[i-1],bin_edges[i])
        
        #penalize negative values and zero
        if f_b <= 0:
            return 10**11
        
        n_b = bin_heights[i-1] #* bin_width
        neg_log_L += f_b - (n_b * np.log(f_b))
    
    return neg_log_L

def mod_gaussian_neg_log_likelihood(params, data):
    # "binning" such that each bin contains either 1 or 0 points. Poisson likelihood reduces to this.
    #Data must be x_i values
    sigma1, sigma2, lambda_ = params
    fit_integral = 1. #verified

    return -1 * np.sum(np.log(mod_gaussian(data, sigma1, sigma2, lambda_))) + fit_integral
     

class ONEHALO_fitter:
    def __init__(self, PATH: str, initial_param_file: str = None, joint: bool = False, log_lambda: bool = False):
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

        if initial_param_file:
            if joint:
                with open(initial_param_file, 'r') as f:
                    self.initial_param_dict = load(f)[f'M_{self.lower_mass}-{self.upper_mass}']
                    #TODO: same as below presumably, with removing errors
            else:
                with open(initial_param_file, 'r') as f:
                    self.initial_param_dict = load(f)
                #NOTE: some dicts are bigger, we want only the param data, we dont care about error values
                self.initial_param_dict = {"sigma_1": self.initial_param_dict['sigma_1'],
                                            "sigma_2": self.initial_param_dict['sigma_2'], 
                                            #   "lambda":np.random.normal(self.initial_param_dict['lambda'], 0.25)}
                                            "lambda": 1.} # start lambda in the middle of its prior space

        else:
            # some random stuff in the ballpark of where we want them to kickstart
            if joint:
                self.initial_param_dict = {"p": -227.7,"n": 0.22,"q": 37.6,"b": 396.4,"m": 7.8,
                                           "c": 70.9,"A": 0.69,"B": 20,"C": -0.005}
            # These values are emprically in the right region. 
            else:
                self.initial_param_dict = {'sigma_1':500.0, 'sigma_2': 150.0, 'lambda':float(not log_lambda)} # init lambda at 1 if not log, 0 if log

        with h5py.File(self.PATH, 'r') as handle:
            self.rel_pos = handle['rel_pos'][:]
            self.rel_vels = handle['rel_vels'][:]#.flatten()
        
        self.rel_sq_dist = np.square(self.rel_pos).sum(axis = 1)
        self.rel_vel_sq = np.square(self.rel_vels) * -0.5


    @staticmethod
    def _fit_modified_gaussian_emcee(data, bins, initial_guess: dict, log_likelihood_func, use_binned = False,
                                     nwalkers = 32, nsteps = 5000, param_labels = [r'$\sigma_1$',r'$\sigma_2$',r'$\lambda$'],
                                     plot = True, verbose = False, filename = 'Mfit', save_params = False, log_lambda = False, **kwargs):
        
        #param_names not to be confused with param_labels; latter is latex formatted
        init_guess, param_names = np.array(list(initial_guess.values())), list(initial_guess.keys()) 
        ndim = init_guess.shape[0]
        #TODO fix noise so that it is redrawn when it lies outside the bounds, not just cast to absolute value or smth
        noise = np.random.randn(nwalkers, ndim)
        noise[:,-1] = np.abs(noise[:,-1]) * 1e-4 #lambda must take positive values and be smaller
        if log_lambda:
            noise[:, -1] *= -1
        pos = init_guess + noise

        log_lambda_str = '_log_lambda' if log_lambda else ''

        if not use_binned:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_func, args = (data, log_lambda)) #, moves = emcee.moves.GaussianMove(0.05))
                                            #  moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
            sampler.run_mcmc(pos, nsteps, progress = verbose)

        else:
            # Compute histogram
            bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood_func, args = (bin_edges, bin_heights))#, moves = emcee.moves.GaussianMove(0.05))
                                            # moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
            
            sampler.run_mcmc(pos, nsteps, progress = verbose)

        # if verbose:
        #     print(
        #         "Autocorrelation time: {0:.2f} steps".format(
        #             sampler.get_autocorr_time()[0]
        #         ))

        if plot:
            # chain plots
            fig, axes = plt.subplots(ndim + 1, figsize=(12, 10), sharex=True)
            samples = sampler.get_chain()

            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(param_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            likelihoods = sampler.get_log_prob()

            axes[-1].plot(likelihoods, alpha = 0.3)
            axes[-1].set(ylabel = r'Log($\mathcal{L}$)')
            # axes[-1].yaxis.set_label_coords(-0.15, 0.5)
            axes[-1].ticklabel_format(useOffset = True)

            
            axes[-1].set_xlabel("Step number")

            plt.savefig(filename + f'_walkers{log_lambda_str}.png', dpi = 200)
            plt.close()

            #corner plot
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True) # this is probably fine

            fig = corner(flat_samples, labels = param_labels, quiet = True,
                            quantiles=[0.16, 0.5, 0.84])

            fig.savefig(filename + f"_corner{log_lambda_str}.png", dpi = 200)
            plt.close(fig)
        
        result = np.zeros(ndim)
        errs = np.zeros((3,2))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            err = np.diff(mcmc)
            
            result[i] = mcmc[1]
            errs[i,0] = err[0]
            errs[i,1] = err[1]
        
        param_dict = dict(zip(param_names, result.tolist()))
        param_dict['errors'] = errs.tolist()
        param_dict['nwalkers'] = nwalkers
        param_dict['nsteps'] = nsteps

        if save_params:
            head, rtail = os.path.split(filename)
            masstail = os.path.split(head)[1]

            param_path = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee'
            if not os.path.isdir(os.path.join(param_path, masstail)):
                os.mkdir(os.path.join(param_path, masstail))
            
            param_path = os.path.join(param_path, masstail, rtail + f'{log_lambda_str}.json')
            
            with open(param_path, 'w') as f:
                dump(param_dict, f, indent = 1)
        
        # Plot if function provided
        if plot:
            if log_lambda:
                plot_distribution_gaussian_mod(mod_gaussian_loglambda, param_dict, data, bins=bins, distname="Modified Gaussian", filename = filename + f'_fit{log_lambda_str}.png', log_lambda = True)
            else:
                plot_distribution_gaussian_mod(mod_gaussian, param_dict, data, bins=bins, distname="Modified Gaussian", filename = filename + f'_fit.png', log_lambda = False)

        return param_dict
        # return samples

    @staticmethod
    def _fit_modified_gaussian_minimize(data, bins, initial_guess,bounds, neg_log_likelihood_func,
                              plot: bool = False, distname = 'Modified Gaussian', use_binned = True,
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


        # if verbose:
        #   print("Optimized parameters:", result.x)
        #   print(result)
        
        if save_params:
            param_dict = dict(zip(param_names, result.x)) 
            head, rtail = os.path.split(filename)
            masstail = os.path.split(head)[1]

            param_path = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/minimize'
            if not os.path.isdir(os.path.join(param_path, masstail)):
                os.mkdir(os.path.join(param_path, masstail))
            
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
                     distname: str = 'Modified Gaussian', verbose: bool = False, save_params: bool = False, **kwargs):
        
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
            rpath = os.path.join(datapath, f'r_{rbin[0]:.2f}-{rbin[1]:.2f}.hdf5')

            try:
                with h5py.File(rpath, 'r') as file:
                    masked_data = file['rel_vels'][:]
            except FileNotFoundError:
                if kwargs['verbose']:
                    tqdm.write(f'{rpath} does not exist, skipping...')
                continue

            filename = f'/disks/cosmodm/vdvuurst/figures/{method}_results_radial_bins/M_{self.lower_mass}-{self.upper_mass}/r_{rbin[0]:.2f}-{rbin[1]:.2f}'
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
                            bins: int = 200, bounds: list = [(50, 1000), (50, 1000), (0, 1)], plot: bool = True,
                            nwalkers: int = 16, nsteps: int = 1000, non_bin_threshold: int = -1,
                            distname: str = 'Modified Gaussian', verbose: bool = False, save_params: bool = False, overwrite: bool = True,
                            return_values = False, log_lambda = False, **kwargs):
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
            distname (str, optional): _description_. Defaults to 'Modified Gaussian'.
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
                                                            verbose = verbose, save_params = save_params, plot = plot, filename = filename)
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
                                                     verbose = verbose, save_params = save_params, plot = plot, filename = filename, log_lambda = log_lambda)
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
                                                     verbose = verbose, save_params = save_params, plot = plot, filename = filename, log_lambda = log_lambda)
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

def plot_all_fits(massbin):
    #scraper of data and params. preferably data that already exists as a catalogue
    pass

def _Gstat(O,E):
    """G-test statistic.

    Args:
        O (array): Observed counts. MUST be integers.
        E (array): Expected counts under the null hypothesis e.g. the model predictions.

    Returns:
        G (float): The G-statistic value
    """
    G = 0
    for i, Oi in enumerate(O):
        if Oi == 0:
            continue
        G += Oi * np.log(Oi/E[i])
    return 2*G

def _Qval(x,k):
    """Find the Q-value aka p-value from a given chi^2 distributed statistic.

    Args:
        x (float): Statistic value
        k (int): The degrees of freedom of the problem.

    Returns:
        Q (float): The Q-value of the statistic given the degrees of freedom.
    """

    #gammainc in scipy is the *regularized* incomplete lower gamma function so /gamma(k/2) is implicit
    return 1 - gammainc(k/2,x/2)

def get_Qvalue(param_dict, data, bins = 200, sanitycheck = False):
    warnings.resetwarnings()
    bin_heights, bin_edges = np.histogram(data, bins = bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist_area=np.sum(bin_heights)

    model = hist_area * np.array([mod_gaussian_integral(param_dict['sigma_1'], param_dict['sigma_2'], param_dict['lambda'],
                                                               bin_edges[i], bin_edges[i+1]) for i in range(len(bin_centers))])
    
    Gval = _Gstat(bin_heights, model) 
    if sanitycheck:
        print(f"{sum(model) = }, {sum(bin_heights) = }")
        print(f'{Gval = }')
    #dof = 4 since three parameters and 1 extra less since if N -1 bins are filled we know exactly the Nth bin count
    warnings.filterwarnings('ignore')
    return _Qval(Gval, len(bin_heights) - 4)



#TODO: make this prettier
def plot_distribution_gaussian_mod(f,param_dict,data,bins,distname, filename = 'Mfit', show = False, log_lambda = False, sanitycheck = False):
    
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

    sigma1, sigma2, lambda_ = param_dict['sigma_1'], param_dict['sigma_2'], param_dict['lambda']
    paramstr = ''
    sig_digits = 4
    for i,param in enumerate(['sigma_1', 'sigma_2', 'lambda']):
        if i == 2 and log_lambda:
            param_latex = 'log_' + param
            sig_digits = 3
        else:
            param_latex = param

        paramstr += latex_formatter[param_latex] + f' = ${param_dict[param]:.{sig_digits}}^{{+{param_dict['errors'][i][0]:.{sig_digits -1}}}}_{{-{param_dict['errors'][i][1]:.{sig_digits -1}}}}$\n'
    
    Qval = get_Qvalue(param_dict, data, bins, sanitycheck = False)
    paramstr += f'Q = {Qval:.3}'

    frame.text(0.155, 0.71, paramstr, transform=plt.gcf().transFigure, backgroundcolor='white',zorder=-1, bbox = {'boxstyle':'round','facecolor':'white'}, fontsize = 12.5)

    hist_area=np.sum(bin_heights)
    frame.plot(DAT,hist_area*f(DAT,sigma1,sigma2,lambda_),'-', label=f"{distname},\nN={hist_area:.0f}",color='red')

    frame.legend(fontsize=12.5, loc="upper right")
    frame.tick_params(axis='both', which='major',length=6, width=2,labelsize=14)

    if log_lambda:
        weighted_sigma = np.average([sigma1, sigma2], weights = [1-10**(lambda_), 10**(lambda_)])
        if sanitycheck:
            print(f'{4*weighted_sigma = }')
    else:
        weighted_sigma = np.average([sigma1, sigma2], weights = [1- lambda_, lambda_])

    frame.set_xlim(-4 * weighted_sigma, 4 * weighted_sigma)
    
    if not show:
        fig.savefig(filename, dpi=200)
        plt.close()
    else:
        plt.show()


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