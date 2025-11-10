## Won't work in isolation, only meant as a reference for Poisson fitting procedures


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from downhillsimplex import DownhillSimplex
from romberg import Romberg
from scipy.special import gammainc
from sampling import rejection_sampling
from quicksort import quicksort

#Defining quantities for 1a
Nsat = 100
a=2.4
b=0.25
c=1.6
xmax = 5
A = 256 / (5*np.power(np.pi,5/3))

#Defining the theoretical functions.
def n(x,A,Nsat,a,b,c):
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def integrand_for_A(x,a,b,c):
    #This is simply n(x) * x^2 (note: not integrated so there is no 4pi). This is used for ease.
    #x(a-3) absorbed the x^2 - now it won't diverge (for reasonable values of a).
    return x**(a-1) * b**(a-3) * np.exp(-(x/b)**c)

def N(x,A=A,Nsat=Nsat,a=a,b=b,c=c):
    return 4*np.pi *A * Nsat * integrand_for_A(x,a,b,c)
    
# ### 1a

#define the golden ratio.
golden = (1 + 5 ** 0.5) / 2
def golden_section_search(a,b,c,f,w=(1+golden)**-1,target_acc=1e-9,max_iter=100):
    """The golden ratio minimization algorithm. Searches in a given interval of width |c - a|. If one wants to find the maximum they should
        input -1 * their desired function.

    Args:
        a (float): The left-most point of the interval. 
        b (float): Center point of the interval.
        c (float): Right-most point of the interval.
        f (callable): function of which the minimum will be found.
        w (float, optional): Parameter that describes the setting of any new point. Defaults to (1+golden)**-1, where golden is the golden ratio.
        target_acc (float, optional): Target accuracy to be achieved, here defined as the interval width |c - a|. Defaults to 1e-9.
        max_iter (int, optional): Maximum amount of iterations before manual break in the algorithm. Defaults to 100.

    Returns:
        minimum,accuracy,niter: Returns the estimate of the minimum position (e.g. the x value of f(x)), the interval width (as accuracy measure) and the number of iterations this took.
    """
    cb = abs(c-b)
    ab = abs(a-b)

    n = 0
    while abs(c-a) > target_acc and n < max_iter:
        x = [a,c][np.argmax((ab,cb))] #take either a or c depending on which interval is biggest
        d = b + (x-b)*w #propose a new point d

        #first possibilty
        if f(d) < f(b):
            #redefine points based on their values
            if b <= d and d <= c:
                a,b = b,d
            elif a <= d and d <= b:
                c,b = b,d
        
        #otherwise:
        else:
            if b <= d and d <= c:
                c = d
            elif a <= d and d <= b:
                a = d

        #redefine the interval widths
        cb = abs(c-b)
        ab = abs(a-b)
        n += 1
    
    #We are done searching, but it still depends if we return d or b; we return the one with the smallest function value.
    if f(d) < f(b):
       return d,abs(c-a),n
    else:
       return b,abs(c-a),n

#put -N(x) as the function since the algorithm can find only a minimum.
xmax,acc,niter = golden_section_search(0,2.5,5,f=lambda x: -N(x),target_acc=1e-7)
#analytical x value of the extremum.
xmax_analytical = b* np.power((a-1)/c,1/c)

print(f'We find a maximum of N(x) at x= {xmax} with an accuracy of {acc:.5e} in {niter} iterations.')
print(f'This has a function value N(xmax) = {N(xmax)}')
print(f'The analytical value of xmax = {xmax_analytical}. There is a difference of {abs(xmax-xmax_analytical)} in these values.')
print(f'This is indicative of the fact that we have not underestimated our accuracy.')
print()

# ### 1b,c,d

#taken directly from given code
def readfile(filename):
    f = open(filename, 'r')
    data = f.readlines()[3:] #Skip first 3 lines 
    nhalo = int(data[0]) #number of halos
    radius = []
    
    for line in data[1:]:
        if line[:-1]!='#':
            radius.append(float(line.split()[0]))
    
    radius = np.array(radius, dtype=float)    
    f.close()
    return radius, nhalo #Return the relative radius for all the satellites in the file, and the number of halos

def optbins(x):
    return np.int32((np.max(x) - np.min(x)) * np.cbrt(len(x)) * 0.5)

def chi2(y,model,var):
    #equation for chi^2 
    return np.sum(np.square(y - model) / var)

def gauss_likelihood(y,edges,Nsat,*p):
    """Function to calculate the chi^2 value for given data and parameters a,b,c.

    Args:
        y (array): Counts of the bins.
        edges (array): Edges of the bins. It must hold that shape(edges) = shape(y) + 1
        Nsat (float): The value of <Nsat> for a given dataset. It is enforced that Ntilde sums to Nsat.
        *p: at the end of the function call the parameters a,b,c must be inputted (in order).

    Returns:
        chi2 (float): The chi^2 value of the model, using the Poisson variance.
    """

    a,b,c = p #unpack the parameters

    #Calculate A in the same way as done in assignment 2 by integrating the function (with A = 1) which should yield 1/A
    #this is recalculated for every step as every new combination of a,b,c yields a new normalization.
    integrand = lambda x: integrand_for_A(x,a=a,b=b,c=c)
    res = Romberg(0,xmax,integrand,m=9)[0]
    A = 1/(4*np.pi*res)

    #Now given the value of A, now calculate the model mean (and thus variance) for each bin.
    integrand_2 = lambda x: N(x,A=A,Nsat=Nsat,a=a,b=b,c=c)
    Ntilde = np.array([Romberg(edges[i],edges[i+1],integrand_2,m=6)[0] for i in range(len(y))])
    #renormalize the Ntilde values so that the model integrates (the sum of aligning integrals may be seen as a single integral)
    #to 1 and multiply by Nsat so that it sums to Nsat like the data does - this must be enforced.
    normalization = np.sum(Ntilde)
    Ntilde = 1/normalization * Ntilde * Nsat
    
    return chi2(y,Ntilde,Ntilde)

def neglnL(y,model):
    #equation for -ln(L) following a Poisson distribution.
    return -np.sum(y * np.log(model) - model)

def neg_log_poisson_likelihood(y,edges,Nsat,*p):
    """Function to calculate the negative log-likelihood value for given data and parameters a,b,c using a Poisson distribution.

    Args:
        y (array): Counts of the bins.
        edges (array): Edges of the bins. It must hold that shape(edges) = shape(y) + 1
        Nsat (float): The value of <Nsat> for a given dataset. It is enforced that Ntilde sums to Nsat.
        *p: at the end of the function call the parameters a,b,c must be inputted (in order).
    Returns:
        likelihood (float): The negative log-likelihood value of the model, using the Poisson distribution.
    """
    a,b,c = p 
    
    #Calculate A in the same way as done in assignment 2 by integrating the function (with A = 1) which should yield 1/A
    #this is recalculated for every step as every new combination of a,b,c yields a new normalization.
    integrand = lambda x: integrand_for_A(x,a=a,b=b,c=c)
    res, _ = Romberg(0,xmax,integrand,m=9)
    A = 1/(4*np.pi*res)

    #Now given the value of A, now calculate the model mean (and thus variance) for each bin.
    integrand_2 = lambda x: N(x,A=A,Nsat=Nsat,a=a,b=b,c=c)
    Ntilde = np.array([Romberg(edges[i],edges[i+1],integrand_2,m=6)[0] for i in range(len(y))])
    #renormalize the Ntilde values so that the model integrates (the sum of aligning integrals may be seen as a single integral)
    #to 1 and multiply by Nsat so that it sums to Nsat like the data does - this must be enforced.
    normalization = np.sum(Ntilde)
    Ntilde = 1/normalization * Ntilde * Nsat

    likelihood = neglnL(y,Ntilde)
    return likelihood

max_x_value = 0
min_x_value = 5 #the minimum will very surely be smaller than this
for i in range(5):
    fname = f'satgals_m1{i+1}.txt'
    xradii,_ = readfile(fname)
    maxx,minx = np.max(xradii),np.min(xradii)
    if maxx > max_x_value:
        max_x_value = maxx
    if minx < min_x_value:
        min_x_value = minx

print(f"Highest found x (radius) value is {max_x_value:.5f}. Therefore we set xmax at 2.5 and we are sure not to exclude any data.")
print(f"Lowest found x (radius) value is {min_x_value:5e}. Therefore we set xmin at 1e-4 and we are sure not to exclude any data.")
print()
xmin = 1e-4
xmax = 2.5 #from the data we see no larger than ~2.4 over all files

def likelihood_minimization(counts,edges,likelihood,Nsat,*p0):
    """Find the minimum likelihood value (e.g. least chi^2 or least -ln(L) for a Poisson distribution). Uses the 
        Downhill Simplex method for multidimensional minimization. 

    Args:
        counts (array): Array of counts
        edges (array): Edges of the bins. shape(edges) = shape(counts) + 1 must hold.
        likelihood (callable): Either neg_log_poisson_likelihood or gauss_likelihood depending on the method we want to use.
        Nsat (float): The value of <Nsat> for a given dataset. It is enforced that Ntilde sums to Nsat. 

    Returns:
        Optimum parameters in the order a, b, c, A and Ntilde at the end, which is an array of the (normalized) model predictions.
    """
    a,b,c = p0

    func_to_minimize = lambda *p: likelihood(counts,edges,Nsat,*p)
    #perform Downhill Simplex
    dwn = DownhillSimplex(func_to_minimize,p0,init_step=0.01)
    (aopt,bopt,copt),acc,niter = dwn.perform(target_accuracy=1e-9,max_iter=250)
    print(f'Downhill Simplex completed in {niter} iterations.')

    #recalculate A and Ntilde for the optimum parameters
    integrand = lambda n: integrand_for_A(n,a=a,b=b,c=c)
    res = Romberg(0,xmax,integrand,m=9)[0]
    A = 1/(4*np.pi*res)

    integrand_2 = lambda x: A*Nsat*integrand_for_A(x,a=aopt,b=bopt,c=copt)
    Ntilde = 4*np.pi*np.array([Romberg(edges[i],edges[i+1],integrand_2,m=9)[0] for i in range(len(counts))])
    #renormalize the Ntilde values so that the model integrates (the sum of aligning integrals may be seen as a single integral)
    #to 1 and multiply by Nsat so that it sums to Nsat like the data does - this must be enforced.
    normalization = np.sum(Ntilde)
    Ntilde = 1/normalization * Ntilde * Nsat #so that it sums to Nsat exactly

    return aopt,bopt,copt,A,Ntilde
      
 
def eval_files(method='chi2',save=False):
    """Evaluate the 5 sattelite files for a given method - either chi2 or Poisson log-likelihood. We find the best fitting parameters 
        and plot our results.

    Args:
        method (str,optional): The method used to find the optimum parameter values. Allowed are 'Poisson', 'Gauss' and 'chi2' (not case sensitive).
        save (bool, optional): _description_. Defaults to False.

    Returns:
        optvals,totalcounts: Returns the optimum parameters (including A, Nsat and nhalo) and the bin_counts for each file so we need only run this 
        function once for each method. 
    """

    #check the given method and assign relevant variables: the function to be minimzed, the savename for the plot and whether to report
    #the minimum chi2 or the minimum log-likelihood. 
    if method.lower() == 'poisson':
        likelihood_function = neg_log_poisson_likelihood
        savename = './plots/my_solution_1c.png'
        optchi2 = False
    elif method.lower() == 'gauss' or method.lower() == 'chi2':
        likelihood_function = gauss_likelihood
        savename = './plots/my_solution_1b.png'
        optchi2 = True
    else:
        print("Specified method cannot be interpreted. Try again. Allowed are \'Poisson\', \'Gauss\' and \'chi2\' (not case sensitive).")
        return

    #define two lists to store results so that we only have to run this function once
    optvals = []
    totalcounts = []

    #create figure and set parameters
    fig1b, ax = plt.subplots(3,2,figsize=(6.4,8.0))
    xmin,xmax = 1e-4,2.5 #after checking there is no value of radii larger than or equal to 2.5, so this maximum does not exclude any data. 


    #hardcoded since there are 5 files, but could be changed in the future.
    for i in range(5):
        filename = f'satgals_m1{i+1}.txt'
        print(f'NOW WORKING ON {filename}')
        x_radii,nhalo = readfile(filename)
        nbins = optbins(x_radii) #find the optimum number of bins for this data.
        print(f'Using {nbins} bins.')
        edges = np.exp(np.linspace(np.log(xmin),np.log(xmax),nbins+1)) #define our bins logarithmically spaced

        Nsat = x_radii.shape[0] / nhalo
        print(f"The mean number of satellites in each halo is {Nsat}")
        binned_data = np.histogram(x_radii,bins=edges,density=False)[0]  /  nhalo #(nhalo * Nsat)
        #binned_data now is the mean number of sattelites per halo in each mass bin i divided by the mean number of sattelites in a halo
        #This way the binned data sums to 1 - it is a probability distribution: p(x)dx = N(x)dx / <Nsat>

        p0 = (a,b,c) #take the initial guess to be what we set in 1a
        aopt,bopt,copt,A,Ntilde = likelihood_minimization(binned_data,edges,likelihood_function,Nsat,*p0)
        
        print(f'Optimal values for parameter:\na = {aopt}\nb = {bopt}\nc = {copt}')
        dof = len(binned_data) - 3 #since there's 3 params a,b,c

        #since data and model are normalized in the same way we can just take it out of the chi^2 equation
        if optchi2: print(f'Optimal chi2/dof = {chi2(binned_data,Ntilde,Ntilde) * (nhalo) / dof}')
        #this is however not as easy to do for the Poisson log-likelihood because of the logarithm. Therefore
        #we input the unnormalized data to calculate the mimimal log-likeihood.
        else: print(f"Minimal -ln(L(a,b,c)) = {neglnL(nhalo*binned_data,nhalo*Ntilde)}")

        optvals.append([[A,Nsat,aopt,bopt,copt],Nsat,nhalo,Ntilde])
        totalcounts.append(binned_data)

        row=i//2
        col=i%2
        print()
        if '1b' in savename:
            labelstring = r'Best-fit profile ($\chi^2$)'
            modelcolor = 'maroon'
        elif '1c' in savename:
            labelstring = r'Best-fit profile (Poisson)'
            modelcolor = 'darkgreen'
        ax[row,col].step(edges[:-1], binned_data, where='post', label='Binned data')
        ax[row,col].step(edges[:-1], Ntilde, where='post', label=labelstring,color=modelcolor,linestyle='dashed')
        ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel=r'N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
        # print()
    ax[2,1].set_visible(False)
    
    plt.tight_layout()
    handles,labels=ax[2,0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc=(0.65,0.15))
    if save: 
        plt.savefig(savename, dpi=600)
    #plt.show()
    plt.close()
    return optvals,totalcounts

#run for chi^2
optvals_gauss,totalcounts = eval_files(method='chi2',save=True)
#run for Poisson
optvals_poisson,_ = eval_files(method='poisson',save=True)

 
def Gstat(O,E):
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

#get the Ntilde values from the output
model_values_poisson = [optval[-1] for optval in optvals_poisson]
model_values_gauss = [optval[-1] for optval in optvals_gauss]

#get the parameters from the output
Nsatnhalo = [optval[-3:-1] for optval in optvals_gauss] #might as well be poisson, nhalo and Nsat are the same.
popt_gauss = [optval[0] for optval in optvals_gauss]
popt_poisson = [optval[0] for optval in optvals_poisson]

def Qval(x,k):
    """Find the Q-value aka p-value from a given chi^2 distributed statistic.

    Args:
        x (float): Statistic value
        k (int): The degrees of freedom of the problem.

    Returns:
        Q (float): The Q-value of the statistic given the degrees of freedom.
    """

    #gammainc in scipy is the *regularized* incomplete lower gamma function so /gamma(k/2) is implicit
    return 1 - gammainc(k/2,x/2)

#Calculate the Q value for the found models of all files.
for i in range(5):
    filename = f'satgals_m1{i+1}.txt'
    print(f'NOW WORKING ON {filename}')

    #Get the necessary parameters and values.
    #Undo any normalization so that the values in Oi are always integers.
    Nsat,nhalo = Nsatnhalo[i]
    Ntilde_poisson = np.array(model_values_poisson[i]) *nhalo
    Ntilde_gauss = np.array(model_values_gauss[i]) *nhalo 
    Oi = (totalcounts[i] * nhalo) #since we use the same binning for gauss and poisson this will be the same for both Poisson and Gauss

    dof = len(Oi) - 3 #3 parameters namely a,b,c
    print(f'k (degrees of freedom) = {dof}')

    Gstati_poisson = Gstat(Oi,Ntilde_poisson)
    Qval_poisson = Qval(Gstati_poisson,dof)

    Gstati_gauss = Gstat(Oi,Ntilde_gauss)
    Qval_gauss = Qval(Gstati_gauss,dof)
    
    print(f'Poisson G-statistic: {Gstati_poisson}, Gauss G-statistic: {Gstati_gauss}')
    print(f'Poisson Q-value: {Qval_poisson}, Gauss Q-value: {Qval_gauss}')
    if Qval_poisson > Qval_gauss: print('Poisson model is (slightly) more consistent with data.')
    elif Qval_poisson == Qval_gauss: print('Both models are equally consistent with data.')
    else: print('Gaussian chi^2 model is (slightly) more consistent with data.')
    print()
        

# ## Sampling (1e)
 
file = 'satgals_m13.txt' #has a decent amount of halos but doesn't take as long to load in as m11, also doesn't have (many) empty halos
xmin,xmax = 1e-4,2.5
def sample_and_fit(num_samples=50,N_generate=10000,file=file,xmin=xmin,xmax=xmax,method='poisson',seed=42,save=False):
    """Sample from a distribution (N_generate times) and re-fit to this synthetic data.

    Args:
        num_samples (int, optional): Amount of times we perform the sampling. Defaults to 10.
        N_generate (int, optional): Amount of samples taken each time. Defaults to 10000.
        file (string, optional): Datafile we want to base our sampling on. Defaults to file = 'satgals_m13.txt'.
        xmin (float, optional): Minimum value of x. Defaults to xmin = 1e-4.
        xmax (float, optional): Maximum value of x. Defaults to xmax = 2.5.
        method (str, optional): Method to use, either Poisson or chi2/gauss. Defaults to 'poisson'.
        seed (int, optional): Random seed used for sampling. Defaults to 42.
        save (bool, optional): Whether we want to save our resulting plot or not. Defaults to False.

    """
    if method.lower() == 'poisson':
        likelihood = neg_log_poisson_likelihood
    elif method.lower() == 'gauss' or method.lower() == 'chi2':
        likelihood = gauss_likelihood
    else:
        print("Specified method cannot be interpreted. Try again. Allowed are \'Poisson\', \'Gauss\' and \'chi2\' (not case sensitive).")
        return
    
    #reload the data and find the relevant information e.g. the binned data
    xradii,m13nhalo = readfile(file)
    m13Nsat = np.shape(xradii)[0] / m13nhalo
    nbins = optbins(xradii)
    edges = np.exp(np.linspace(np.log(xmin),np.log(xmax),nbins+1))
    binned_data = np.histogram(xradii,bins=edges)[0] / (m13nhalo) #average out over all haloes

    p0 = (2.4,0.25,1.6) #take the initial guess to be what we set in 1a
    #Refit the file so we can get the optimum parameters. As there is no randomization involved in this process this will yield the same
    #results as eariler in the code.
    aopt,bopt,copt,A,Ntilde = likelihood_minimization(binned_data,edges,likelihood,m13Nsat,*p0)
    p0 = (aopt,bopt,copt) #take the initial guess for the sample fits to be what we found before as optimal parameters to speed up the DHS

    #optN will be the function we sample from.
    optN = lambda x: N(x,A,m13Nsat,aopt,bopt,copt) / (m13Nsat) #this is p(x)dx but it takes on values larger than 1

    #find the x-value of the maximum op optN to greatly improve the speed of rejection sampling.
    x_at_maximum = bopt* np.power((aopt-1)/copt,1/copt) #analytical result of the maximum position
    #this could also be done via the golden ratio search method, but since there is an analyitcal value we take that as it is more accurate.

    #normalization offset from sampling which this constant fixes.
    normoffset = N_generate / m13Nsat

    print(f'Generating {N_generate} samples each time.')

    fig1e, ax = plt.subplots()
    ax.step(edges[:-1], binned_data, where='post', label='Binned data',zorder=9)
    ax.step(edges[:-1], Ntilde, where='post', label='Best-fit profile (Poisson)', color="C1",zorder=10)
    all_models = []
    for i in range(num_samples):
        seed = seed + i #will yield a different seed and thus different sample each iteration
        sample = rejection_sampling(optN,low=xmin,high=2.5,xpeak=x_at_maximum,target_size=N_generate,seed=seed)
        binned_sample = np.histogram(sample,bins=edges)[0] / (normoffset) #bin the sample and re-normalize with the found offset
        Ntilda = likelihood_minimization(binned_sample,edges,likelihood,m13Nsat,*p0)[-1] #get a new model from the sample
        all_models.append(Ntilda) #save it for finding the mean later
        sample_handle, = ax.step(edges[:-1], Ntilda, where='post', color=f"black",linestyle='dashdot',alpha=0.25)
    mean_model = np.mean(all_models,axis=0)
    mean_handle, = ax.step(edges[:-1], mean_model, where='post', color=f"black",linestyle='dashed',alpha=1,zorder=8)

    ax.set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{13}} M_{{\\odot}}/h$")
    handles,labels=ax.get_legend_handles_labels()
    handles.extend([sample_handle,mean_handle])
    labels.extend(['Best fits on sampled data','Mean profile of samplings'])
    plt.legend(handles,labels)
    if save:
        if method.lower() == 'poisson':
            savestring = './plots/my_solution_1e_poisson.png'
        elif method.lower() == 'gauss' or method.lower() == 'chi2':
            savestring = './plots/my_solution_1e_chi2.png'
        plt.savefig(savestring, dpi=600)
    plt.close()

sample_and_fit(method='Poisson',save=True) 
sample_and_fit(method='chi2',save=True)
