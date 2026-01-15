import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
import argparse
from functions import *
from json import load

latex_formatter = {'sigma_1':r'$\sigma_1$', 'sigma_2': r'$\sigma_2$', 'lambda':r'$\lambda$', 'loglambda':r'$\log_{10}\left(\lambda\right)$'}


def format_plot():
    # Define some properties for the figures so that they look good
    SMALL_SIZE = 10 * 2 
    MEDIUM_SIZE = 12 * 2
    BIGGER_SIZE = 14 * 2

    plt.rc('axes', titlesize=SMALL_SIZE)                     # fontsize of the axes title\n",
    plt.rc('axes', labelsize=MEDIUM_SIZE)                    # fontsize of the x and y labels\n",
    plt.rc('xtick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
    plt.rc('ytick', labelsize=SMALL_SIZE, direction='out')   # fontsize of the tick labels\n",
    plt.rc('legend', fontsize=SMALL_SIZE)                    # legend fontsize\n",
    mpl.rcParams['axes.titlesize'] = BIGGER_SIZE
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['font.family'] = 'STIXgeneral'

    mpl.rcParams['figure.dpi'] = 100

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


def plot_distribution_gaussian_mod(func, param_dict, data, bins, distname = 'Modified Gaussian', filename = 'Mfit', show = False, loglambda = False, fix_lambda = False, sanitycheck = False):
    sigma1, sigma2, lambda_ = param_dict['sigma_1'], param_dict['sigma_2'], param_dict['lambda']
    # print(f'LOOK: {sigma1 = }, {sigma2 = }, {lambda_ =}')
    # Set plot appeareance
    fig = plt.figure(figsize=(7,7))
    frame=fig.add_subplot(1,1,1)
    frame.set_xlabel('Velocity difference v', fontsize=16)
    frame.set_ylabel('Number of galaxies per v', fontsize=16)
    frame.tick_params(axis='both', which='major',length=6, width=2,labelsize=14)

    if loglambda:
        weighted_sigma = np.average([sigma1, sigma2], weights = [1-10**(lambda_), 10**(lambda_)])
        if sanitycheck:
            print(f'{4*weighted_sigma = }')
    else:
        weighted_sigma = np.average([sigma1, sigma2], weights = [1- lambda_, lambda_])

    # range_mask = (data >= -4 * weighted_sigma) & (data <= 4 * weighted_sigma)
    # data = data[range_mask]

    bin_heights, bin_edges = np.histogram(data, bins=bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width= bin_edges[1] - bin_edges[0] 
    bin_widths = np.diff(bin_edges)  # The width of each bin
    number_density = bin_heights / bin_widths  # Normalize by bin width


    frame.bar(bin_centers, number_density, width=bin_width, align='center', edgecolor = 'black')
    DAT=np.linspace(np.min(data),np.max(data),1000)

    # Creating textbox for parameter values
    paramstr = ''
    sig_digits = 4
    if not fix_lambda:
        for i,param in enumerate(['sigma_1', 'sigma_2', 'lambda']):
            if i == 2 and loglambda:
                param_latex = 'log' + param
                sig_digits = 3
            else:
                param_latex = param

            paramstr += latex_formatter[param_latex] + f' = ${param_dict[param]:.{sig_digits}}^{{+{param_dict['errors'][i][1]:.{sig_digits}}}}_{{-{param_dict['errors'][i][0]:.{sig_digits}}}}$\n'
    else:
        distname = 'Single Gaussian'
        i, param = 0, 'sigma_1'
        param_latex = param
        paramstr += latex_formatter[param_latex] + f' = ${param_dict[param]:.{sig_digits}}^{{+{param_dict['errors'][i][1]:.{sig_digits}}}}_{{-{param_dict['errors'][i][0]:.{sig_digits}}}}$\n'

    # Calculating G-statistic for goodness of fit estimate
    integral_func = mod_gaussian_integral if not loglambda else mod_gaussian_integral_loglambda
    # Qval = get_Qvalue(param_dict, bin_heights, bin_edges, integral_func = integral_func, sanitycheck = False)
    Gval = get_Gstat(param_dict, bin_heights, bin_edges, integral_func = integral_func)

    #Update textbox w/ G statistic and plot
    # paramstr += f'Q = {Qval:.3}'
    paramstr += f'G = {Gval:.1f}'
    if not fix_lambda:
        frame.text(0.155, 0.71, paramstr, transform=plt.gcf().transFigure, backgroundcolor='white',zorder=-1, bbox = {'boxstyle':'round','facecolor':'white'}, fontsize = 12)
    else:
        frame.text(0.155, 0.78, paramstr, transform=plt.gcf().transFigure, backgroundcolor='white',zorder=-1, bbox = {'boxstyle':'round','facecolor':'white'}, fontsize = 12)

    hist_area=np.sum(bin_heights)
    frame.plot(DAT,hist_area*func(DAT,sigma1,sigma2,lambda_),'-', label = f"{distname}\nN={hist_area:.0f}, N" + r'$_\mathrm{b}$' + f" = {bins}", color='red')

    frame.legend(fontsize=12.5, loc="upper right")

    frame.set_xlim(-4 * weighted_sigma, 4 * weighted_sigma)
    
    if not show:
        fig.savefig(filename, dpi=200)
        plt.close()
    else:
        plt.show()

def _get_data_and_plot(func, mass_bin, r_bin, loglambda, fix_lambda, show = False,
                       param_path_base = '/disks/cosmodm/vdvuurst/data/OneHalo_param_fits/emcee',
                       data_path_base = '/disks/cosmodm/vdvuurst/data/OneHalo_0.5dex',
                       plot_path_base = '/disks/cosmodm/vdvuurst/figures/emcee_results_radial_bins'):
    
    loglambda_str = '_log_lambda' if loglambda else ''
    fixlambda_str = '_fix_lambda' if fix_lambda else ''

    param_path = param_path_base + f'/M_{mass_bin[0]}-{mass_bin[1]}/r_{r_bin[0]:.2f}-{r_bin[1]:.2f}{loglambda_str}{fixlambda_str}.json'
    data_path = data_path_base + f'/M_{mass_bin[0]}-{mass_bin[1]}/r_{r_bin[0]:.2f}-{r_bin[1]:.2f}.hdf5'

    if not fix_lambda:
        plot_path = plot_path_base + f'/M_{mass_bin[0]}-{mass_bin[1]}/r_{r_bin[0]:.2f}-{r_bin[1]:.2f}_fit{loglambda_str}.png'
    else:
        plot_path = plot_path_base + f'/M_{mass_bin[0]}-{mass_bin[1]}/single_gaussian/r_{r_bin[0]:.2f}-{r_bin[1]:.2f}_fit{loglambda_str}.png'

    try: 
        with h5py.File(data_path, 'r') as handle:
            data = handle['rel_vels'][:].flatten()
        
        with open(param_path, 'r') as file:
            param_dict = load(file)
    except FileNotFoundError as e:
        return True, e
  
    plot_distribution_gaussian_mod(func, param_dict, data, rice_bins(data.size), show = show, loglambda = loglambda, fix_lambda = fix_lambda, filename = plot_path)

    return False, 'Nothing went wrong'

def plot_onehalo_fit(mass_bins = None, r_bins = None, show = False, loglambda = False, fix_lambda = False, verbose = False):
    if not mass_bins:
        mass_range = np.arange(12.0, 15.5 + 0.5, 0.5).astype(np.float32)
        mass_bins = np.array([[mass_range[i],mass_range[i+1]] for i in range(len(mass_range)-1)])

    if not r_bins:
        r_range = modified_logspace(0, 5., 18)
        r_bins = np.array([[r_range[i], r_range[i+1]] for i in range(len(r_range) - 1)])

    func = mod_gaussian_loglambda if loglambda else mod_gaussian
    
    if np.size(mass_bins) == 2: # just one massbin
        if np.size(r_bins) == 2: # just one rbin
            failed, msg = _get_data_and_plot(func, mass_bins, r_bins, loglambda, fix_lambda, show)
            if failed:
                print(f'Code failed: {msg}')
                return

        else: #all rbins
            iterable = tqdm(r_bins) if verbose else r_bins
            for r_bin in iterable:
                failed, msg = _get_data_and_plot(func, mass_bins, r_bin, loglambda, fix_lambda, show)
                if failed:
                    print(f'Code failed: {msg}')
                    return # higher r_bins will also not exist
    
    else: #all mbins
        if np.size(r_bins) == 2:
            iterable = tqdm(mass_bins) if verbose else mass_bins
            for mass_bin in iterable:
                failed, msg = _get_data_and_plot(func, mass_bin, r_bins, loglambda, fix_lambda, show)
                if failed:
                    print(f'Code failed: {msg}')
                    continue
        else:
            iterable = tqdm(mass_bins) if verbose else mass_bins
            for mass_bin in iterable:
                for r_bin in r_bins:
                    failed, msg = _get_data_and_plot(func, mass_bin, r_bin, loglambda, fix_lambda, show)
                    if failed:
                        break # higher r-bins will not exist so continue to next mass_bin



if __name__ == '__main__':
    format_plot()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-M1','--lower_mass', required = False, help = 'Lower bound of the mass range in dex above 10^10 Msun. This is inclusive!')
    parser.add_argument('-M2','--upper_mass', required = False, help = 'Upper bound of the mass range in dex above 10^10 Msun. This is inclusive!') 
    parser.add_argument('-r1','--lower_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!')
    parser.add_argument('-r2','--upper_radius', required = False, help = 'Lower bound of the radial range in Mpc. This is inclusive!') 
    parser.add_argument('-LL', '--loglambda', type = int, default = 0, help = 'Have the lambda parameter scale logarithmically instead of linearly. Has alternate filename structure. Default is 0.')
    parser.add_argument('-FL', '--fix_lambda', type = int, default = 0, help = 'Fixes lambda to 0 to emulate a single Gaussian. Has alternate filename structure. Default is 0.')
    parser.add_argument('-V', '--verbose', type = int, default = 1, help = 'Controls whether to show progress bars. Default is True')

    args = parser.parse_args()
    #TODO: implement something that allows you to plot all radial bins from a certain ondergrens (or until some upper limit) and vice versa for mass
    if args.lower_mass and args.upper_mass:
        lm, um = np.float32(args.lower_mass), np.float32(args.upper_mass)
        if lm < 10 or  um < 10:
            offset = 10
        else:
            offset = 0
        mass_bin = [lm + offset, um + offset]
    
    else:
        mass_bin = None
    
    if args.lower_radius and args.upper_radius:
        r_bin = [np.float32(args.lower_radius), np.float32(args.upper_radius)]
    else:
        r_bin = None
    
    plot_onehalo_fit(mass_bin, r_bin, loglambda = bool(args.loglambda), fix_lambda = bool(args.fix_lambda),
                     show = False, # if calling from terminal we want to save the plots, not show them
                     verbose = bool(args.verbose)
                     )
    

    

