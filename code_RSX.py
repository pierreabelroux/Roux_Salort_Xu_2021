#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 27-07-2021
# Author: Pierre Roux
# Collaboration with Delphine Salort and Zhou Xu

""" Code associated with the article "Adaptation to DNA damage as a bet-hedging mechanism in a fluctuating environment"
    This code contains the parameters and functions used to produce the figures of the article. The test section contains some of the instructions used to produce the figures.
"""

__author__ = "Pierre Roux"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Pierre Roux", "Delphine Salort", "Zhou Xu"]
__license__ = "GNU General public license v3.0"
__version__ = "2.9.4"
__maintainer__ = "Pierre Roux"
__email__ = "pierre.rouxmp@laposte.net"
__status__ = "Final"



## Imports

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import statistics as stat
import scipy.stats
import time
from scipy.integrate import odeint, quad, simps

# Mesh ploting :
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

start_time = time.time()



## Definition of constants

T=np.linspace(0,1,500)     # Temporal grid of lenght one
GAMMA_DAM = 0.1        # probability of death for damaged cells  (between 0.05 and 0.2)
GAMMA_AD = 0.35       # Probabilite of death for adapted cells  (between 0.2 and 0.5)
GAMMA_REP=0       # probabily of death for repaired cells

DELTA = 0.02     # Probability of repairing for adapted cells
POP_MAX = 100000
INITIAL_CELLS = 20000

ALPHA_MAX = 0.5     # Maximum of repair probability
SIGMA = 0.5        # Variance of repair curve
MEAN_A = 1        # Mean of repair curve

BETA_MAX = 0.5      # Maximum of adaptation probability

NUMBER_TEST = 10000 # Number of runs for each average time computed in the stochastic model

#Display of constants when file is the main file
if __name__ == "__main__" :
    print("___## Version 9.3 ##___\n\n")
    print("___General Package___\n")

    print(" GAMMA_DAM = ",GAMMA_DAM,"\n","GAMMA_AD = ",GAMMA_AD,"\n","GAMMA_REP = ", GAMMA_REP,"\n\n",
               "DELTA = ",DELTA,"\n","POP_MAX = ",POP_MAX,"\n","INITIAL_CELLS = ",INITIAL_CELLS,"\n","\n")
    print("Parameters for alpha : \n","ALPHA_MAX = ",ALPHA_MAX," ; SIGMA = ",SIGMA," ; MEAN_A = ",MEAN_A)
    print("Parameters for beta : \n","BETA_MAX = ",BETA_MAX,"\n\n")


## Fading functions

def unity(t) :
    """
    Function that is always equal to one.
    """
    return 1


def exponential_decay(t, start=MEAN_A, l=0.25) :
    """
    Exponentially decaying function after "start" and 1 before start.
    """
    if t < start :
        return 1
    else :
        return np.exp(-l*(t-start))



## Functions defining the ODE system

def alpha(t, mean_a, fade=unity) :
    """
    Compute the probability for a cell to repair DNA damage.
    
    Args:
        t (float): time since damage occured.
        mean_a (float) : center of the gaussian repair curve.
        fade (function, optional): fading function, defaults to unity.
    
    Returns:
        float : probability of successful repair.
    """
    return ALPHA_MAX*np.exp(  - (t - mean_a )**2/SIGMA   ) * fade(t)


def beta(t,mean_b,slope_b) :
    """
    Compute the probability for a cell to adapt to DNA damage.
    
    Args:
        t (float): time since damage occured.
        mean_b (float): center of the logistic adaptation curve.
        slope_b (float) : slope parameter of the logistic adaptation curve.
    
    Returns:
        float : probability of adaptation.
    """
    return BETA_MAX/(1+np.exp(-slope_b*( t - mean_b ) ))


def f(y, t, k, mean_b, slope_b, mean_a=MEAN_A, environment=0, fade=unity) :
    """
    Function f defining the ODE y'=f(y,t).
    
    Args:
        t (float): time since damage occured.
        mean_b (float): center of the logistic adaptation curve.
        slope_b (float): slope parameter of the logistic adaptation curve.
        mean_a (float, optional): mean time of the repair curve, defaults to MEAN_A. 
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
        fade (function, optional): fading function for the repair curve, defaults to unity.
        
    
    Returns:
        float : probability of adaptation
    """
    
    if GAMMA_DAM + beta(t+k,mean_b,slope_b) >= 1 :
        beta_tmp = 1 - GAMMA_DAM
        alpha_tmp = 0
    else :
        beta_tmp = beta(t+k,mean_b,slope_b)
        if GAMMA_DAM + beta(t+k, mean_b, slope_b) +alpha(t+k, mean_a, fade=fade) >=1 :
            alpha_tmp = (1 - GAMMA_DAM - beta_tmp) * indic(t+k,environment)
        else :
            alpha_tmp = alpha(t+k, mean_a, fade=fade) * indic(t+k,environment)
    return [-(beta_tmp+alpha_tmp+GAMMA_DAM)*y[0], y[1]*(1-(y[0]+y[1]+y[2])/POP_MAX) + beta_tmp*y[0] - GAMMA_AD*y[1] - DELTA*y[1]  , y[2]*(1-(y[0]+y[1]+y[2])/POP_MAX) + alpha_tmp*y[0] + DELTA*y[1] - GAMMA_REP*y[2] ]



def f_2(y, t, k, mean_b, slope_b, mean_a=MEAN_A, environment=0, fade=unity) :
    """
    Function f_2 defining the ODE y'=f_2(y,t).
    
    Args:
        t (float): time since damage occured
        mean_b (float): center of the logistic adaptation curve
        slope_b (float): slope parameter of the logistic adaptation curve.
        mean_a (float, optional): mean time of the repair curve, defaults to MEAN_A. 
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
        fade (function, optional): fading function for the repair curve, defaults to unity.
    
    Returns:
        float: probability of adaptation
    """
    
    if GAMMA_DAM + beta(t+k,mean_b,slope_b) >= 1 :
        beta_tmp = 1 - GAMMA_DAM
        alpha_tmp = 0
    else :
        beta_tmp = beta(t+k,mean_b,slope_b)
        if GAMMA_DAM + beta(t+k, mean_b, slope_b) +alpha(t+k, mean_a, fade=fade) >=1 :
            alpha_tmp = (1 - GAMMA_DAM - beta_tmp) * indic(t+k,environment)
        else :
            alpha_tmp = alpha(t+k, mean_a, fade=fade) * indic(t+k,environment)
    return [-(beta_tmp+alpha_tmp+GAMMA_DAM)*y[0], y[1]*(1-(y[0]+y[1]+y[2]+y[3])/POP_MAX) + beta_tmp*y[0] - GAMMA_AD*y[1] - DELTA*y[1]  , y[2]*(1-(y[0]+y[1]+y[2]+y[3])/POP_MAX) + alpha_tmp*y[0] - GAMMA_REP*y[2], y[3]*(1-(y[0]+y[1]+y[2]+y[3])/POP_MAX) + DELTA*y[1] - GAMMA_REP*y[2] ]


def indic(t,environment) :
    """
    Return 0 if t < environment, 1 otherwise.
    """
    if t < environment :
        return 0
    else :
        return 1


def adaptation_pace(t, mean_b, slope_b):
    """
    Compute the adaptation pace by integrating the adaptation probability curve.
    """
    coef = quad( lambda s : beta(s,mean_b,slope_b),0,t)[0]
    return (1-np.exp( -coef ))*INITIAL_CELLS



## Environment probability densities

def gaussian_density(t,mean_g=2, sigma_g=0.5) :
    """
    Gaussian law of mean mean_g and variance sigma_g density fuction.
    """
    return ( 1/ (sigma_g*np.sqrt(2*np.pi)) )*np.exp( -(t-mean_g)**2/(2*sigma_g**2) )

def beta_density(t, alpha_d=2, beta_d=3) :
    """
    Beta law of parameters alpha_d, beta_d density function
    """
    return scipy.stats.beta.pdf(x, alpha_d, beta_d)

def exponential_density(t,l=0.5) :
    """
    Exponential law of mean 1/l density function.
    """
    return l*np.exp(-l*t)

def uniform_density_10(t):
    return 1/10



## Display functions

def display_1d(x, y, option, xlab=None, ylab=None, title=None) :
    """
    Plot y versus x as a line with appropriate legend depending on the option.
    Option should be 1 for mean_b and 2 for slope_b.
    """
    fig, ax = plt.subplots()
        
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.setp(ax.spines.values(), linewidth=3)
        
    ax.plot(x,y, linewidth=3)
    
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    if option == 1 :
        if xlab == None :
            ax.set_xlabel("Value of $\mu_b$", fontsize=35)
        else :
            ax.set_xlabel(xlab, fontsize=35)
        if title != None :
            ax.set_title(title)
        
    elif option == 2 :
        if xlab == None :
            ax.set_xlabel("Value of p", fontsize=35)
        else :
            ax.set_xlabel(xlab, fontsize=35)
        if title != None :
            ax.set_title(title)
    
    if ylab == None :
        ax.set_ylabel("Saturation time $T_S$", fontsize=35)
    else :
        ax.set_ylabel(ylab, fontsize=35)
    
    fig.show()


def display_2d(x, y, vz, option, zaxis_range=None, xlab=None, ylab=None, title=None) :
    """
    Two-dimensional data plot of vz versus an x,y grid.
    Option should be 1 (colormap plot), 2 (surface plot in 3D) or 3 (both previous plots).
    
    Args:
        x (numpy.ndarray): abscissa discretisation.
        y (numpy.ndarray): ordinate discretisation.
        vz (numpy.ndarray): matrix of values to plot.
        option (int): choice of graph style.
            1 : 2d colormap graph.
            2 : 3d surface graph.
        zaxis_range (tuple, optional): bounds for the z axis, defaults to None.
    """
    vx, vy = np.meshgrid(x, y)
    if option == 1 :
        fig, ax = plt.subplots()
        c = ax.pcolor(vx, vy, vz,  cmap=cm.coolwarm)
        
        # Labels
        if xlab == None :
            ax.set_xlabel('Value of $\mu_b$', fontsize=20)
        else :
            ax.set_xlabel(xlab, fontsize=20) 
        if ylab == None :
            ax.set_ylabel('Value of p', fontsize=20)
        else :
            ax.set_ylabel(ylab, fontsize=20)
        
        fig.colorbar(c, ax=ax)
    
    elif option == 2 :
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(vx, vy, vz, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        
        # Customize the z axis.
        if zaxis_range != None :
            ax.set_zlim(zaxis_range[0], zaxis_range[1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Labels
        if xlab == None :
            ax.set_xlabel('Value of $\mu_b$', fontsize=20)
        else :
            ax.set_xlabel(xlab, fontsize=20) 
        if ylab == None :
            ax.set_ylabel('Value of p', fontsize=20)
        else :
            ax.set_ylabel(ylab, fontsize=20)
        
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
    #title
    if title != None :
        ax.set_title(title) 
    
    fig.show()



## Stochastic model


def alpha_sto(n):
    """
    Computation of the repair probability in experiment().
    """
    return ALPHA_MAX*np.exp(  - (n - MEAN_A )**2/SIGMA   )


def beta_sto(n, mean_b, pente=3):
    """
    Computation of the adaptation probability in experiment().
    """
    return BETA_MAX/(1+np.exp(-pente*( n - mean_b ) ))


def reproduce_sto(totalDam,totalAd,totalRep):
    """
    Reproduction function used in experiment().
    """
    if totalDam + totalAd + totalRep ==0 :
        print("la population s'est eteinte")
        return POP_MAX, POP_MAX
    else :
        nbrMax = totalDam + totalAd + totalRep
        if nbrMax + totalAd + totalRep <= POP_MAX :
            return totalAd,totalRep
        else :
            return int(totalAd*(POP_MAX - nbrMax)/(totalAd+totalRep) )+1, int(totalRep*(POP_MAX - nbrMax)/(totalAd+totalRep) )


def experiment(mean_b, n_max=POP_MAX, n0=INITIAL_CELLS):
    """
    Simulate the fate of the cell population. Random variables are drawn from a multinomial law to determine the fate of the cells at each step, which is equivalent to drawing a random variable for each cell independently.
    
    Args:
        mean_b (float): value of the adaptation timing. 
        n_max (int, optional): maximum population allowed, defaults to POP_MAX.
        n_0 (int, optional):  initial population, defaults to INITIAL_CELLS.
    
    Returns:
        int : number of steps needed to fill the medium with healthy (repaired) cells
    """
    damaged_cells = n0
    repaired_cells = 0
    adapted_cells = 0
    dead_cells = 0
    n = 0
    
    while repaired_cells < n_max:
        
        a = alpha_sto(n)
        b = beta_sto(n, mean_b)
        
        # Fate of damaged cells
        draw = rd.multinomial(damaged_cells, [max(0,1-a-b-GAMMA_DAM), a, b, min(1-a-b,GAMMA_DAM)]) # Draw a multinomial law
        damaged_cells = draw[0]
        repaired_cells += draw[1]
        adapted_cells += draw[2]
        dead_cells += draw[3]
        
        # Fate of adapted cells
        draw = rd.multinomial(adapted_cells, [DELTA, GAMMA_AD, 1-DELTA-GAMMA_AD])   # Draw a multinomial law
        repaired_cells = min(n_max, repaired_cells + draw[0])
        dead_cells += draw[1]
        adapted_cells = draw[2]
        
        # Reproduce cells is space is available
        if damaged_cells + repaired_cells + adapted_cells < n_max :      
                r = r = reproduce_sto(damaged_cells, adapted_cells, repaired_cells)
                adapted_cells += r[0]
                repaired_cells += r[1]
        n+=1

    return n



## Minima search functions : stable environment

def best_mean(start, end, nbr_param, slope_b=3, mean_a=MEAN_A, display=True, display_text=True, environment=0) :
    """
    Find the optimal value for mean_b among nbr_param evenly spaced parameters from start to end.
    Can display graphs or print information depending on optional parameters.
    
    Args:
        start (float): smaller parameter tested.
        end (float): end of the parameters set.
        nbr_param (int) : number of parameters tested.
        slope_b (int, optional): value of the slope_b parameter in experiments, defaults to 3.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
        display (bool, optional): Wether to plot data or not, defaults to True.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
    
    Returns:
        list : list of optimal parameters.
        numpy.ndarray : list of tested parameters.
        numpy.ndarray : results of the experiments for the set of parameters.
    """
    parameters = np.linspace(start, end, nbr_param)       # List of tested parameters
    final_times = []
    for value in parameters :
        mean_b = value
        k = 0
        Tmax = 0
        cpt = 0
        x = [INITIAL_CELLS,0,0]

        while abs( x[-1] - POP_MAX ) > 1 :
            cpt += 1
            k += 1
            # Solve ODE on time interval of length 1 :
            y = odeint(lambda x,t : f(x, t, k, mean_b, slope_b, mean_a=mean_a, environment=environment), x, T)        
            x = y[-1]
            Tmax += 1
        i = 0
        while abs(y[i][-1] - POP_MAX) > 1 :
            i += 1
        final_times.append(Tmax-1 + T[i])
    
    best_time = min(final_times)        # find the optimal recovery time among the tests
    best_parameters = [parameters[i] for i, j in enumerate(final_times) if j == best_time]  # find optimal parameters
    
    if display :
        display_1d(parameters, final_times, 1)
    
    if display_text :
        print("List of mean_b values tested : \n", parameters, "\n")
        print("The best parameters are : ", best_parameters, "\n")
    
    return best_parameters, parameters, np.array(final_times)


def best_slope(start, end, nbr_param, mean_b=3.886, mean_a=MEAN_A, display=True, display_text=True, environment=0) :
    """
    Find the optimal value for slope_b among nbr_param evenly spaced parameters from start to end.
    Can display graphs or print information depending on optional parameters.
    
    Args:
        start (float): smaller parameter tested.
        end (float): end of the parameters set.
        nbr_param (int) : number of parameters tested.
        mean_b (int, optional): value of the slope_b parameter in experiments, defaults to 3.886.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
        display (bool, optional): Wether to plot data or not, defaults to True.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
    
    Returns:
        list: list of optimal parameters.
        numpy.ndarray: list of tested parameters.
        numpy.ndarray: results of the experiments for the set of parameters.
    """
    parameters = np.linspace(start, end, nbr_param)       # List of tested parameters
    final_times = []
    for value in parameters :
        slope_b = value
        k = 0
        Tmax = 0
        cpt = 0
        x = [INITIAL_CELLS,0,0]

        while abs(x[-1] - POP_MAX) > 1 :
            cpt += 1
            k += 1
            # Solve ODE on time interval of length 1 :
            y = odeint(lambda x,t : f(x, t, k, mean_b, slope_b, mean_a=mean_a, environment=environment), x, T)
            x = y[-1]
            Tmax += 1
        i = 0
        while abs(y[i][-1] - POP_MAX) > 1 :
            i += 1
        final_times.append(Tmax-1 + T[i])
    
    best_time = min(final_times)        # find the optimal recovery time among the tests
    best_parameters = [parameters[i] for i, j in enumerate(final_times) if j == best_time]  # find optimal parameters
    
    if display :
        display_1d(parameters, final_times, 2)
    
    if display_text :
        print("List of slope_b values tested : \n",np.linspace(start, end, nbr_param),"\n")
        print("The best parameters are : ",best_parameters,"\n")
    
    return best_parameters, parameters, np.array(final_times)


def best_mean_and_slope(start_mean, end_mean, nbr_mean, start_slope, end_slope, nbr_slope,
                                        display=3, display_text=True, mean_a=MEAN_A, environment=0) :
    """
    Find the optimal pair of values for mean_b and slope_b among a grid of parameters.
    Can display graphs or print information depending on optional parameters.
    
    Args :
        start_mean (float): smaller parameter tested.
        end_mean (float): end of the parameters set.
        nbr_mean (int) : number of parameters tested for mean_b.
        start_slope (float): smaller parameter tested.
        end_slope (float): end of the parameters set.
        nbr_slope (int) : number of parameters tested for slope_b.
        display (int, optional): instruction for ploting graphs, defaults to 3.
            0 : no graph.
            1 : 2d colormap graph.
            2 : 3d surface graph.
            3 : both 2d colormap graph and 3d surface graph.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
    
    Returns :
        list : list of optimal parameters.
        numpy.ndarray : list of tested parameters for mean_b.
        numpy.ndarray : list of tested parameters for slope_b.
        numpy.ndarray : (nbr_slope,nbr_mean)-matrix containing the results of the experiments.
    """
    means = np.linspace(start_mean, end_mean, nbr_mean)       # List of tested means
    slopes = np.linspace(start_slope, end_slope, nbr_slope)       # List of tested slopes
    
    final_times = np.zeros((nbr_slope, nbr_mean))
    for k1 in range(nbr_mean) :
        for k2 in range(nbr_slope) :
            mean_b, slope_b = means[k1], slopes[k2]
            k = 0
            Tmax = 0
            cpt = 0
            x = [INITIAL_CELLS,0,0]
    
            while abs( x[-1] - POP_MAX ) > 1 :
                cpt += 1
                k += 1
                # Solve ODE on time interval of length 1 :
                y = odeint(lambda x,t : f(x, t, k, mean_b, slope_b, mean_a=mean_a, environment=environment), x, T)
                x = y[-1]
                Tmax += 1
            i = 0
            while abs(y[i][-1] - POP_MAX) > 1 :
                i += 1
            final_times[k2,k1] = Tmax-1 + T[i]
    
    best_time = np.amin(final_times)        # find the optimal recovery time among the tests
    ind = np.unravel_index(np.argmin(final_times, axis=None), final_times.shape)  # find optimal parameters
    best_parameters = (means[ind[1]],slopes[ind[0]])

    if display == 1 :
        display_2d(means, slopes, final_times, 1)
    elif display == 2 :
        display_2d(means, slopes, final_times, 2)
    elif display == 3 :
        display_2d(means, slopes, final_times, 1)
        display_2d(means, slopes, final_times, 2)
    
    if display_text :
        print("List of tested values for mean_b: \n",means,"\n")
        print("List of tested values for slope_b: \n",slopes,"\n")
        print("The best (mean_b, slope_b) couple is:",best_parameters,"\n !!! warning : it shows the first pair of optimal parameters encountered, there may be more of them !!!\n")
        # print("Final times : \n",final_times)
    
    return best_parameters, means, slopes, final_times



## Minima search functions : random environment

def best_slope_heterogeneity(start, end, nbr_param, density_environment, mean_b=3.886, mean_a=MEAN_A,
                                                        shift=[False,], display=True, display_text=True, ) :
    """
    Find the optimal value for slope_b among nbr_param evenly spaced parameters from start to end with environmental effect with probability density density_environment.
    Can display graphs or print information depending on optional parameters.
    
    Args:
        start (float): smaller parameter tested.
        end (float): end of the parameters set.
        nbr_param (int) : number of parameters tested.
        density_environment (function) : probability density on [0,10] defining environmental effect random value.
        mean_b (int, optional): value of the slope_b parameter in experiments, defaults to 3.886.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
        display (bool, optional): Wether to plot data or not, defaults to True.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
    
    Returns:
        list : list of optimal parameters.
        numpy.ndarray : list of tested parameters.
        numpy.ndarray : results of the experiments for the set of parameters.
    """
    parameters = np.linspace(start, end, nbr_param)       # List of tested parameters
    
    if shift[0] :
        tt = np.linspace(1,11,100)
    else :
        tt = np.linspace(0,7,100)
    proba = np.zeros(tt.size)
    for i in range(proba.size) :
        proba[i]=density_environment(tt[i])
    mass = simps(proba,tt)
    integrals = []
    for value in parameters :
        slope_b = value
        final_times_weighted = []
        for t in tt :
            k = 0
            Tmax = 0
            cpt = 0
            x = [INITIAL_CELLS,0,0]
    
            while abs(x[-1] - POP_MAX) > 1 :
                cpt += 1
                k += 1
                # Solve ODE on time interval of length 1 :
                if shift[0] :
                    y = odeint(lambda x,s : f(x, s, k, mean_b, slope_b, mean_a=t, environment=0, fade=shift[1]), x, T)
                else :
                    y = odeint(lambda x,s : f(x, s, k, mean_b, slope_b, mean_a=mean_a, environment=t), x, T)
                x = y[-1]
                Tmax += 1
            i = 0
            while abs(y[i][-1] - POP_MAX) > 1 :
                i += 1
            final_times_weighted.append( (Tmax-1 + T[i])*density_environment(t) )
        
        mean_time = (1/mass)*simps( np.array(final_times_weighted), tt)
        
        integrals.append(mean_time)

    best_time = min(integrals)        # find the optimal recovery time among the tests
    best_parameters = [parameters[i] for i, j in enumerate(integrals) if j == best_time]  # find optimal parameters
    
    if display :
        if shift[0]:
            display_1d(parameters, integrals, 2)
        else :
            display_1d(parameters, integrals, 2)
    
    if display_text :
        print("mean_b = ", mean_b)
        print("The total numerical mass of the probability density is :", mass)
        print("List of slope_b values tested : \n", np.linspace(start, end, nbr_param),"\n")
        print("The best parameters are : ", best_parameters, "\n")
    
    return best_parameters, parameters, np.array(integrals)


def best_slope_heterogeneity_gaussian_2d(start_slope, end_slope, nbr_slope, start_gaussian, end_gaussian, nbr_gaussian, 
                                      sigma_g=1.5, mean_b=3.886, shift=[False,], display=3, display_text=True, mean_a=MEAN_A) :
    """
    Find the optimal value for slope_b among nbr_param evenly spaced parameters from start to end with environmental effect with probability density density_environment.
    Can display graphs or print information depending on optional parameters.
    
    Args:
        start (float): smaller parameter tested.
        end (float): end of the parameters set.
        nbr_param (int) : number of parameters tested.
        density_environment (function) : probability density on [0,10] defining environmental effect random value.
        mean_b (int, optional): value of the slope_b parameter in experiments, defaults to 3.886.
        display (bool, optional): Wether to plot data or not, defaults to True.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
    
    Returns:
        list : list of optimal parameters.
        numpy.ndarray : list of tested parameters.
        numpy.ndarray : results of the experiments for the set of parameters.
    """
    slopes = np.linspace(start_slope, end_slope, nbr_slope)       # List of tested parameters
    gaussian_means = np.linspace(start_gaussian, end_gaussian, nbr_gaussian)       # List of tested parameters
    tt = np.linspace(0,7,100)
    integrals = np.zeros((nbr_slope, nbr_gaussian))
    
    for k1 in range(nbr_gaussian) :
        print("k1 vaut : ",k1) #
        end_time = time.time()
        print("Execution time : ",end_time-start_time)
        mean_g = gaussian_means[k1]
        proba = np.zeros(tt.size)
        for i in range(proba.size) :
            proba[i]= gaussian_density(tt[i],mean_g=mean_g, sigma_g=sigma_g)
        mass = simps(proba,tt)
        for k2 in range(nbr_slope) :
            print("     k2 vaut : ",k2)  #
            slope_b = slopes[k2]
            final_times_weighted = []
            for t in tt :
                k = 0
                Tmax = 0
                cpt = 0
                x = [INITIAL_CELLS,0,0]
        
                while abs(x[-1] - POP_MAX) > 1 :
                    cpt += 1
                    k += 1
                    # Solve ODE on time interval of length 1 :
                    y = odeint(lambda x,s : f(x, s, k, mean_b, slope_b, mean_a=mean_a, environment=t), x, T)
                    x = y[-1]
                    Tmax += 1
                i = 0
                while abs(y[i][-1] - POP_MAX) > 1 :
                    i += 1
                final_times_weighted.append( (Tmax-1 + T[i])*gaussian_density(t,mean_g=mean_g, sigma_g=sigma_g) )
            
            mean_time = (1/mass)*simps( np.array(final_times_weighted), tt)
            
            integrals[k2, k1] = mean_time
    
    best_time = np.amin(integrals)        # find the optimal recovery time among the tests
    ind = np.unravel_index(np.argmin(integrals, axis=None), integrals.shape)  # find optimal parameters
    best_parameters = (gaussian_means[ind[1]],slopes[ind[0]])
    
    if display == 1 :
        display_2d(gaussian_means, slopes, integrals, 1, xlab="mean of random environment")
    elif display == 2 :
        display_2d(gaussian_means, slopes, integrals, 2, xlab="mean of random environment")
    elif display == 3 :
        display_2d(gaussian_means, slopes, integrals, 1, xlab="mean of random environment")
        display_2d(gaussian_means, slopes, integrals, 2, xlab="mean of random environment")
    
    if display_text :
        print("mean_b = ", mean_b)
        print("Variance of random gaussian environment :", sigma_g)
        print("The total numerical mass of the probability density is :", mass)
        print("List of slope_b values tested : \n", slopes,"\n")
        print("List of gaussian_mean values tested : \n", gaussian_means,"\n")
        print("The best parameters are : ", best_parameters, "\n")
    
    return best_parameters, gaussian_means, slopes, integrals



## Percentage of final cells which undergo adaptation


def final_cells_ratio(start, end, nbr_param, mean_b=3.886, mean_a=MEAN_A, display=True, display_text=True, environment=0):
    """
    Find the value for slope_b that maximises the percentage of adapted cells descendents in the final healthy population, slope_b being tested among nbr_param evenly spaced parameters from start to end.
    Can display graphs or print information depending on optional parameters.
    
    Args:
        start (float): smaller parameter tested.
        end (float): end of the parameters set.
        nbr_param (int) : number of parameters tested.
        mean_b (int, optional): value of the slope_b parameter in experiments, defaults to 3.886.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
        display (bool, optional): Wether to plot data or not, defaults to True.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
    
    Returns:
        list : list of ratios for the different values of p
        list : list of optimal parameters.
        numpy.ndarray : list of tested parameters.
        numpy.ndarray : results of the experiments for the set of parameters.
    """
    parameters = np.linspace(start, end, nbr_param)       # List of tested parameters
    ratios = []
    for value in parameters :
        slope_b = value
        k = 0
        Tmax = 0
        cpt = 0
        x = [INITIAL_CELLS,0,0,0]

        while abs(x[-1]+x[-2] - POP_MAX) > 1 :
            cpt += 1
            k += 1
            # Solve ODE on time interval of length 1 :
            y = odeint(lambda x,t : f_2(x, t, k, mean_b, slope_b, mean_a=mean_a, environment=environment), x, T)
            x = y[-1]
            Tmax += 1
        i = 0
        while abs(y[i][-1] + y[i][-2] - POP_MAX) > 1 :
            i += 1
        # final_times.append(Tmax-1 + T[i])
        ratios.append(y[i][-1]/(y[i][-1]+y[i][-2])*100)
        # print("percentage for p=", value,": ",  y[i][-1]/(y[i][-1]+y[i][-2])*100)
        
    best_ratio = max(ratios)        # find the optimal recovery time among the tests
    best_parameters = [parameters[i] for i, j in enumerate(ratios) if j == best_ratio]  # find optimal parameters
    
    if display :
        display_1d(parameters, ratios, 2, "Value of p", "Percentage of final healthy cells descending from adapted cells")
    
    if display_text :
        print("List of slope_b values tested : \n",np.linspace(start, end, nbr_param),"\n")
        print("The best parameters are : ",best_parameters,"\n")
    
    return ratios, best_parameters, parameters, np.array(ratios)
    
    

def final_cells_evolution(mean_b=3.886, slope_b=1, mean_a=MEAN_A, display=True, display_text=True, environment=0):
    """
    Finds the Saturation time and then compute a complete simulation of the ODE up to that time.
    Can display graphs or print information.
    
    Args:
        mean_b (float, optional): value of the mean_b parameter in experiments, defaults to 3.886.
        slope_b (float, optional): value of the slope_b parameter in experiments, defaults to 1.
        mean_a (float, optional): value of mean_a in experiments, defaults to module level constant MEAN_A.
        display (bool, optional): Wether to plot data or not, defaults to True.
        display_text (bool, optional): Wether to write information about tested parameters or not, defaults to True.
        environment (float, optional): allows to include environmental effect in experiments.
            Repair is impossible between t=0 and t=environment.
            Defaults to 0 (no environmental effect).
    
    Returns:
        numpy.ndarray : complete odeint matrix after a simulation up to Saturation time.
    """
    
    # Obtention of Saturation time T_final
    k = 0
    Tmax = 0
    cpt = 0
    x = [INITIAL_CELLS,0,0,0]
    while abs(x[-1]+x[-2] - POP_MAX) > 1 :
        cpt += 1
        k += 1
        # Solve ODE on time interval of length 1 :
        y = odeint(lambda x,t : f_2(x, t, k, mean_b, slope_b, mean_a=mean_a, environment=environment), x, T)
        x = y[-1]
        Tmax += 1
    i = 0
    while abs(y[i][-1] + y[i][-2] - POP_MAX) > 1 :
        i += 1
    T_final = Tmax-1 + T[i]
    
    # Simulation beetween 0 and T_final
    x_0 = [INITIAL_CELLS,0,0,0]
    time_grid_complete = np.linspace(0,T_final,500)
    y_final = odeint(lambda x,t : f_2(x, t, 0, mean_b, slope_b, mean_a=mean_a, environment=environment), x_0, time_grid_complete)
    
    # Display of the graphs along time if wanted
    if display :
        fig, ax = plt.subplots()
            
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.setp(ax.spines.values(), linewidth=3)
        
        ax.plot(time_grid_complete, y_final[:,0], 'k', label="damaged cells", linewidth=3)
        ax.plot(time_grid_complete, y_final[:,1], 'b', label="adapted cells", linewidth=3)
        ax.plot(time_grid_complete, y_final[:,2], 'orange', label="repaired cells from damaged cells", linewidth=3)
        ax.plot(time_grid_complete, y_final[:,3], 'c', label="repaired cells from adapted cells", linewidth=3)
        
        ax.xaxis.set_tick_params(width=3)
        ax.yaxis.set_tick_params(width=3)
        
        ax.set_xlabel("Time", fontsize=35)
        ax.set_ylabel("Cell number", fontsize=35)
        ax.legend(prop={'size': 17})
        
        fig.show()
    
    # Display of information about parameters in the run
    if display_text :
        print("mean_b = ", mean_b, "slope_b = ",slope_b, "mean_a = ", mean_a)
    
    return y_final



## Tests

if __name__ == "__main__" :
    
    # tt = np.linspace(0,20,1000)
    # aa = np.zeros(tt.size)
    # for i in range (aa.size):
    #     aa[i] =  adaptation_pace(tt[i], 4, 10)
    # 
    # fig, ax = plt.subplots()
    # ax.plot(tt, aa)
    # ax.set_xlabel("time")
    # ax.set_title("Pure adaptation process without death and repair")
    # ax.set_ylabel("number of adapted cells")
    # fig.show()
    # 
    # 
    # experiment(3.886, n_max=POP_MAX, n0=INITIAL_CELLS)
    
    start = 0.8
    end = 8
    nbr_param = 50
    best_mean(start, end, nbr_param, slope_b=3, display=1, mean_a=MEAN_A, environment=0)
    
    
    # 
    # start = 0.03
    # end = 7
    # nbr_param = 100
    # best_slope(start, end, nbr_param, mean_b=1, display=1, mean_a=MEAN_A, environment=0)
    # 
    # start_mean = 1
    # end_mean = 8
    # nbr_mean = 50
    # start_slope = 0.03
    # end_slope = 5
    # nbr_slope = 50
    # 
    # best_mean_and_slope(start_mean, end_mean, nbr_mean, start_slope, end_slope, nbr_slope,
    #                                       display=3, display_text=1, mean_a=MEAN_A, environment=0)
    # 
    # start = 0.1
    # end = 4.7
    # nbr_slope = 50
    # 
    # print("\ntest with a Gaussian density for environment variation \n Parameters of the gaussian law : mean_g=1.7, sigma_g=1.5\n")
    #     
    # best_slope_heterogeneity(start, end, nbr_slope, lambda t : gaussian_density(t,mean_g=1.7,sigma_g=1.5), mean_b=3.886, display=True, display_text=True, mean_a=MEAN_A)
    
    
    
    # start_slope, end_slope, nbr_slope, start_gaussian, end_gaussian, nbr_gaussian = 1.3, 5, 30, 1.4, 1.9, 30
    # best_slope_heterogeneity_gaussian_2d(start_slope, end_slope, nbr_slope, start_gaussian, end_gaussian, nbr_gaussian, 
    #                                   sigma_g=1.5, mean_b=4, shift=[False,], display=3, display_text=True, mean_a=MEAN_A)

    #   start = 0.4
    # end = 5
    # nbr_slope = 50
    # 
    # print("\nTest with an exponential density for environment variation \n Parameter of the exponential law : l=0.42\n")
    # 
    # best_slope_heterogeneity(start, end, nbr_slope, lambda t : exponential_density(t,l=0.42), mean_b=3.886, display=True, display_text=True, mean_a=MEAN_A)
    # 
    # 
    # 
    # start = 0.3
    # end = 5
    # nbr_slope = 50
    # 
    # print("\nTest with a uniform density on [1,11] for environment variation")
    # print("\nExponential decay : \n         lambda t : exponential_decay(t, start=MEAN_A, l=0.3)")
    # 
    # best_slope_heterogeneity(start, end, nbr_slope, uniform_density_10, mean_b=3.886, shift=[True, lambda t : exponential_decay(t, start=MEAN_A, l=0.2)], display=True, display_text=True, mean_a=MEAN_A)
    # 
    
    # means_a = np.linspace(0.9, 10, 10)
    # means_b = np.zeros(len(means_a))
    # for i in range(len(means_a)):
    #     print(means_a[i])
    #     means_b[i] = best_mean(0, 14, 80, slope_b=7, mean_a=means_a[i], display=False, display_text=False,  environment=0)[0][0]
    # display_1d(means_a, means_b, 1, "Value of $\mu_a$", "Optimal value of $\mu_b$")
    
    
    # 
    # start = 0.03
    # end = 7
    # nbr_param = 50
    # 
    # absis = np.linspace(start, end, nbr_param)
    # courbe_2 = final_cells_ratio(start, end, nbr_param, mean_b=3.886, mean_a=MEAN_A, display=True, display_text=False, environment=1.7)[0]
    # courbe_1 = final_cells_ratio(start, end, nbr_param, mean_b=3.886, mean_a=MEAN_A, display=False, display_text=False, environment=0)[0]
    # 
    # fig, ax1 = plt.subplots()
    # 
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.setp(ax1.spines.values(), linewidth=3)
    # 
    # color = 'tab:red'
    # ax1.plot(absis, courbe_1, color=color, linewidth=3)
    # ax1.set_xlabel("Time", fontsize=35)
    # ax1.xaxis.set_tick_params(width=3)
    # ax1.yaxis.set_tick_params(width=3)
    # ax1.tick_params(axis='y', labelcolor=color, labelsize=30)
    # ax1.set_ylabel("Percentage of final healthy cells descending\n from adapted cells: stable environment", color=color, fontsize=29)
    # 
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # 
    # color = 'tab:blue'
    # ax2.plot(absis, courbe_2, color=color, linewidth=3)
    # ax2.yaxis.set_tick_params(width=3)
    # ax2.set_ylabel("Percentage of final healthy cells descending\n from adapted cells: random environment", color=color, fontsize=29)
    # ax2.tick_params(axis='y', labelcolor=color, labelsize=30)
    # plt.setp(ax2.spines.values(), linewidth=3)
    # 
    # # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # 
    # fig.show()
    # 
    # 
    # 
    # final_cells_evolution()
    # 
    # final_cells_evolution(mean_b=3.886, slope_b=1, mean_a=-50, display=True, display_text=True, environment=0)
    # 
    # final_cells_evolution(mean_b=3.886, slope_b=1, mean_a=1, display=True, display_text=True, environment=1.7)
    # 
    

## # Execution time

end_time = time.time()

if __name__ == "__main__" :
    print("Execution time : ",end_time-start_time)

