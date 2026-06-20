
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from json import load
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import os
from tqdm import tqdm
from ONEHALO import ONEHALO_MADD_fitter, param_info, _make_mass_mask, _make_radial_mask, str_from_mbin, str_from_rbin
from functions import mkdir_if_non_existent, rice_bins, modified_logspace, double_gaussian
from functional_forms import get_function_combinations

from warnings import filterwarnings
# this might be a liiiitle dangerous but cleans up the output by a lot. that's because we often see invalid values in log10, but that's ok
filterwarnings('ignore',category = RuntimeWarning)


def format_plot():
    # Define some properties for the figures so that they look good
    SMALL_SIZE = 8 * 2 
    MEDIUM_SIZE = 10 * 2
    BIGGER_SIZE = 12 * 2

    plt.rc('axes', titlesize=BIGGER_SIZE)                     # fontsize of the axes title\n",
    plt.rc('axes', labelsize=MEDIUM_SIZE)                    # fontsize of the x and y labels\n",
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
    plt.rc('legend', fontsize=SMALL_SIZE)                    # legend fontsize\n",
    mpl.rcParams['axes.titlesize'] = BIGGER_SIZE
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'STIXgeneral'

    mpl.rcParams['figure.dpi'] = 300

    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True

    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['ytick.minor.size'] = 4

    mpl.rcParams['xtick.major.width'] = 1.25
    mpl.rcParams['ytick.major.width'] = 1.25
    mpl.rcParams['xtick.minor.width'] = 1
    mpl.rcParams['ytick.minor.width'] = 1



def add_pretty_colorbar(im: mpl.image.AxesImage , ax: plt.Axes, fig: plt.Figure, label = '', fontsize = 40, ticksize= 30):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label(label, fontsize = fontsize)

class MADD_plotter:
    def __init__(self, init_method = 'Nelder-Mead', subsampled_functions = False, subsample_data = True):
        self.MADD_fitter = ONEHALO_MADD_fitter(PATH = f'/disks/cosmodm/vdvuurst/data/Onehalo_M_12-15.5{'_subsampled' if subsample_data else ''}.hdf5')
        self.param_path = f'/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/MADD{'_subsample' if subsample_data else ''}'
        if init_method == 'Nelder-Mead':
            self.param_path = os.path.join(self.param_path, init_method)

        if subsampled_functions:
            self.all_combis, self.all_names, self.combi_numbers, self.combi_subsample, self.combi_subsample_names, self.combi_subsamples_numbers = get_function_combinations(create_subsample=subsampled_functions)
        else:
            self.all_combis, self.all_names, self.combi_numbers = get_function_combinations(create_subsample=subsampled_functions)
        
        listed_param_path = os.listdir(self.param_path)
        bics = np.zeros(shape = len(listed_param_path))
        Ls = np.zeros(shape = len(listed_param_path))
        combi_nrs = bics.copy().astype(int)
        sigma_bounded = combi_nrs.copy().astype(bool)

        print('FINDING ALL BICs AND SORTING')
        for i,file in tqdm(enumerate(listed_param_path), total = len(listed_param_path)):
            filepath = os.path.join(self.param_path, file)
            if not os.path.isfile(filepath):
                continue
            combi_nr = int(file.split('_')[-1].split('.')[0])            
            with open(filepath) as f:
                param_dict = load(f)
            
            combi_nrs[i] = combi_nr
            bics[i] = param_dict['BIC']
            Ls[i] = param_dict['likelihood']
            sigma1, sigma2, _ = self._get_DG_from_combi_nr(combi_nr, param_dict)
            sigma_bounded[i] = np.all(np.logical_or(sigma1 == 2500., sigma1==1.)) or np.all(np.logical_or(sigma2 == 2500., sigma2==1.))

        argsort_bics = np.argsort(bics)
        self.best_combi_nrs = combi_nrs[argsort_bics]
        print(f'TOP 10 CNRs: {self.best_combi_nrs[:10]}')
        self.sorted_bics = bics[argsort_bics]
        self.subsample_data = subsample_data

    def _load_param_dict(self, combi_nr):
        with open(os.path.join(self.param_path,f'function_combi_{combi_nr}.json'), 'r') as f:
            param_dict = load(f)    
        return param_dict
    
    def _get_DG_from_combi_nr(self, combi_nr, param_dict):
        func_combi = self.all_combis[combi_nr - 1]
        n_params_r, n_params_m, _ = param_info(func_combi)
        params = np.array(param_dict['parameters'])
        split_params = self.MADD_fitter.split_parameters(params, n_params_m)
        DG = self.MADD_fitter.get_double_gauss_parameters(split_params, func_combi, n_params_r)

        return DG#, param_dict['BIC']

    def _rainbow_plots(self, DG_params: np.ndarray, combi_nr: int):
        yaxes = [r'$\sigma_1$ [km/s]', r'$\sigma_2$ [km/s]', r'$\lambda$']
        params = ['sigma_1', 'sigma_2', 'lambda']

        for param_idx in range(3): # loop over DG params

            fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (14, 7), sharey=False)

            dist_norm = plt.Normalize(np.min(self.MADD_fitter.rel_dist), np.max(self.MADD_fitter.rel_dist))
            mass_norm = plt.Normalize(np.min(self.MADD_fitter.halo_masses + 10), np.max(self.MADD_fitter.halo_masses +10))

            mass_scat = ax1.scatter(self.MADD_fitter.halo_masses+10, DG_params[param_idx], c = self.MADD_fitter.rel_dist, cmap = 'nipy_spectral', norm = dist_norm, s = 5)

            add_pretty_colorbar(mass_scat, ax1, fig, r'$r_{\mathrm{1h}}$ [R$_{\mathrm{200m}}$]', fontsize = 40)

            rad_scat = ax2.scatter(self.MADD_fitter.rel_dist, DG_params[param_idx], c = self.MADD_fitter.halo_masses+10, norm = mass_norm, cmap = 'nipy_spectral', s = 5)

            add_pretty_colorbar(rad_scat, ax2, fig, r'$M_{\mathrm{h}}$ [dex]', fontsize = 40)

            ax1.set_ylabel( yaxes[param_idx], fontsize = 40)
            ax1.set_xlabel(r'$M_{\mathrm{h}}$ [dex]', fontsize = 40)
            ax2.set_ylabel(yaxes[param_idx], fontsize = 40)
            ax2.set_xlabel( r'$r_{\mathrm{1h}}$ [R$_{\mathrm{200m}}$]', fontsize = 40)

            ax1.tick_params(axis='both', labelsize=30)
            ax2.tick_params(axis='both', labelsize=30)


            fig.tight_layout()

            subdir = './figures/onehalo_MADD/rainbow_plots'
            mkdir_if_non_existent(subdir)
            if self.subsample_data:
                subdir = os.path.join(subdir, 'subsampled')
                mkdir_if_non_existent(subdir)      

            subdir = os.path.join(subdir, f'function_combi_{combi_nr}')
            mkdir_if_non_existent(subdir)

            plt.savefig(os.path.join(subdir, f'{params[param_idx]}.png'), dpi = 300, bbox_inches = 'tight')
            plt.close()

    def plot_in_bin(self, best_params: np.ndarray, function_combi: list, combi_number: int,
                n_params_r: list, n_params_m: list,
                BIC_score: np.float32, mbin: list | tuple, rbin: list | tuple,
                filepath: str) -> None:
    
        halo_masses_prior = self.MADD_fitter.halo_masses
        rel_dist_prior = self.MADD_fitter.rel_dist

        # Define the bin
        mbin_mask = _make_mass_mask(halo_masses_prior, *mbin, logmass = True)
        rbin_mask = _make_radial_mask(rel_dist_prior, *rbin)
        bin_mask = np.logical_and(mbin_mask, rbin_mask)

        # Get binned data
        vel_data_in_bin = self.MADD_fitter.rel_vels[bin_mask].flatten() # first apply mask to all 3-vectors, then flatten
        min_half_v_sq_in_bin = self.MADD_fitter.min_half_v_sq_arr[bin_mask]
        masses_in_bin = halo_masses_prior[bin_mask]
        rel_dist_in_bin = rel_dist_prior[bin_mask]

        bins = rice_bins(vel_data_in_bin.size)

        # Calculate the DG params in the bin
        DG_params = self.MADD_fitter.get_double_gauss_parameters(self.MADD_fitter.split_parameters(best_params, n_params_m), function_combi,
                                                        n_params_r, masses_in_bin, rel_dist_in_bin)
        
        # Plotting
        fig, ax = plt.subplots(figsize = (7,7))
        ax.set_xlabel('One-halo velocity $v_j$ [km/s]')
        ax.set_ylabel('Density')
        ax.tick_params(axis='both', which='major',length=6, width=2, labelsize=14)

        # Bin velocity histogram and plot it
        bin_heights, bin_edges = np.histogram(vel_data_in_bin, bins=bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width= bin_edges[1] - bin_edges[0] 
        bin_widths = np.diff(bin_edges)  # The width of each bin
        number_density = bin_heights / bin_widths  # Normalize by bin width
        hist_area=np.sum(bin_heights)
        ax.bar(bin_centers, number_density, width=bin_width, align='center', edgecolor = 'black', label = r"$M$ = " + f"{hist_area:.0f}"+ "\n"
                                                                                                        + r'$N_\mathrm{b}$' + f" = {bins}")

        # # Add BIC score in textbox
        # ax.text(0.155, 0.83, f'BIC = {BIC_score:.2e}', transform=plt.gcf().transFigure,
        #         backgroundcolor='white',zorder=-1,
        #         bbox = {'boxstyle':'round','facecolor':'white'}, fontsize = 12)

        # Get the lowest and highest point of any model at all velocities
        DAT = np.linspace(np.min(vel_data_in_bin),np.max(vel_data_in_bin), min_half_v_sq_in_bin.size)
        ax_func = lambda x: hist_area * double_gaussian(DAT, *x).flatten() # This is kinda slow
        all_models = np.apply_along_axis(ax_func, 0, DG_params)
        min_model_at_point = np.min(all_models, axis = 1)
        max_model_at_point = np.max(all_models, axis = 1)
        ax.fill_between(DAT.flatten(), min_model_at_point, max_model_at_point, alpha = 0.5, color = 'red', zorder = 2)

        # Plot the model corresponding to the center of the bin
        center_bin_mass, center_bin_rad = np.mean(mbin), np.mean(rbin)
        DG_center = self.MADD_fitter.get_double_gauss_parameters(self.MADD_fitter.split_parameters(best_params, n_params_m), function_combi,
                                                        n_params_r, center_bin_mass, center_bin_rad)
        ax.plot(DAT.flatten(), ax_func(DG_center), label = 'Center bin model', color = 'blue', zorder = 1, lw = 1.5)

        # Finishing touches
        ax.legend(loc="upper left")
        axtitle1 = f'{str_from_mbin(mbin).replace('_',' = ').replace('M', '$M_{\mathrm{h}}$')} dex,'
        axtitle2 = f' {str_from_rbin(rbin).replace('_',' = ').replace('r','$r_{\mathrm{1h}}$')} ' + r'R$_{\mathrm{200m}}$'
        ax.set_title(axtitle1 + axtitle2, fontsize = 20)
        
        filename = os.path.join(filepath, f'function_combi_{combi_number}')
        mkdir_if_non_existent(filename)
        # filename = os.path.join(filename, f'{str_from_mbin(mbin)}_{str_from_rbin(rbin)}_fit.png')
        filename = os.path.join(filename, f'{str_from_mbin(mbin)}_{str_from_rbin(rbin)}_fit.pdf')

        # Setting xlims for better viewing
        data_std = np.std(vel_data_in_bin)
        ax.set_xlim(-5*data_std, 5*data_std)

        fig.tight_layout()
        fig.savefig(filename, bbox_inches= 'tight', dpi = 200)
        plt.close()



    def plot_rainbows_and_histograms_for_combi_nr(self, combi_nr):
        # Get relevant information about function combi from number
        func_combi = self.all_combis[combi_nr - 1]
        param_dict = self._load_param_dict(combi_nr)
        params = np.array(param_dict['parameters'])
        BIC_score = param_dict['BIC']

        # Get the DG parameters
        n_params_r, n_params_m, ntot = param_info(func_combi)
        split_params = self.MADD_fitter.split_parameters(params, n_params_m)
        DG = self.MADD_fitter.get_double_gauss_parameters(split_params, func_combi, n_params_r)

        # Rainbow plots
        # self._rainbow_plots(DG, combi_nr)
        print(f'CNR {combi_nr} finished rainbow plots.')

        # Histogram with shaded areas, need to manually adjust code here to fit to more bins
        fp_for_hists = '/disks/cosmodm/vdvuurst/figures/onehalo_MADD/histograms'
        mkdir_if_non_existent(fp_for_hists)
        if self.subsample_data:
            fp_for_hists = os.path.join(fp_for_hists, 'subsampled')
            mkdir_if_non_existent(fp_for_hists)
        
        # defining the mass and rbins from the simple onehalo model to iterate over
        mass_range = np.arange(2.0, 5.5 + 0.5, 0.5).astype(np.float32)
        mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])

        r_range = modified_logspace(0, 2.5, 20)
        rbins = np.array([[r_range[i],r_range[i+1]] for i in range(len(r_range)-1)]) 

        # Some example bins to show the model over larger ranges of mass and distance
        self.plot_in_bin(param_dict['parameters'], func_combi, combi_nr, n_params_r, n_params_m, BIC_score, [2.0, 2.5], [1, 1.5], filepath = fp_for_hists)
        self.plot_in_bin(param_dict['parameters'], func_combi, combi_nr, n_params_r, n_params_m, BIC_score, [4.0, 5.5], [0.00, 0.10], filepath = fp_for_hists)
        self.plot_in_bin(param_dict['parameters'], func_combi, combi_nr, n_params_r, n_params_m, BIC_score, [2.5, 3.5], [1.5, 2.50], filepath = fp_for_hists)
        self.plot_in_bin(param_dict['parameters'], func_combi, combi_nr, n_params_r, n_params_m, BIC_score, [3.0, 3.5], [0.00, 0.07], filepath = fp_for_hists)

        # if self.subsample_data: # All simple onehalo bins for comparison, takes too long for full data
        #     fp_for_hists = os.path.join(fp_for_hists, 'simple_bins')
        #     mkdir_if_non_existent(fp_for_hists)
        #     for mbin in mass_bins:
        #         for rbin in rbins:
        #             self.plot_in_bin(param_dict['parameters'], func_combi, combi_nr, n_params_r, n_params_m, BIC_score, mbin, rbin, filepath = fp_for_hists)


if __name__ == '__main__':
    format_plot()

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-IM', '--initial_method', default = 'Nelder-Mead', help = 'Initial condition method. Nelder-Mead for subsampled data, subsample for non-subsampled data. Defaults to Nelder-Mead.')
    args = parser.parse_args()

    init_method = args.initial_method

    match init_method:
        case 'Nelder-Mead':
            subsampled_data = True
        case 'subsample':
            subsampled_data = False
        case _:
            raise ValueError('Invalid initial method given. Choose Nelder-Mead or subsample.')

    MADD_plotter_instance = MADD_plotter(init_method=init_method, subsample_data=subsampled_data)
    print('For the best combis, creating plots...')
    # print(MADD_plotter_instance.sorted_bics[:10] / MADD_plotter_instance.sorted_bics[0])
    # print((MADD_plotter_instance.sorted_bics[:10] / MADD_plotter_instance.sorted_bics[0] -1 ) * 100)

    # print(MADD_plotter_instance.best_combi_nrs[-1])
    # print(MADD_plotter_instance.sorted_bics[-1] / MADD_plotter_instance.sorted_bics[0])
    # print((MADD_plotter_instance.sorted_bics[-1] / MADD_plotter_instance.sorted_bics[0] -1 ) * 100)


    if subsampled_data:
        iterable = MADD_plotter_instance.best_combi_nrs[:10]
    else:
        iterable = [5447] 
        # iterable = MADD_plotter_instance.best_combi_nrs

    from multiprocessing import Pool
    NPROCS = len(iterable)
    with Pool(NPROCS) as p, tqdm(total=len(iterable)) as pbar:
        for _ in p.imap_unordered(MADD_plotter_instance.plot_rainbows_and_histograms_for_combi_nr, iterable):
            pbar.update()