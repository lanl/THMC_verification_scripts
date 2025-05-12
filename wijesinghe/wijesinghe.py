"""
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

"""

__author__ = "Jeffrey Hyman"
__version__ = "0.1"
__maintainer__ = "Jeffrey Hyman"
__email__ = "jhyman@lanl.gov"

"""
Analytic solution for Anada Wijesinghe (1986) https://doi.org/10.2172/59961

a constant load is applied at one end of a fracture and drainage is allowed out of the same end of said fracture.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import logging
import os
import scipy.special as sp

def aperture(t, x_array, P0, b_i,k_n,P_i,mu):
    """ Analytic solution for aperture

    Parameters
    -------------------
        t : float 
            Time in seconds
        x_array : numpy array
            x values of the domain
        P0 : float
            Inlet pressure
        b_i : float 
            initial aperture (m) 
        k_n : float 
            fracture stiffness
        P_i : float
            initial pressure (Pa)
        mu : int
            viscosity (Pa*s)

    Returns 
    -----------------
        b : numpy array
            fracture aperture at x_array, t 

    Notes
    -----------------
        This is an approximate solution for small fracture deformation
    
    """
    print("Computing aperture Wijesinghe")
    D = b_i**3*k_n/(24*mu)
    L = np.sqrt(D*t)
    eta = x_array/L
    
    w_0 = 1-(P_i-P0)/(b_i*k_n)
    
    w_n = 1-(sp.erfc(eta/np.sqrt(8))/2*(1-w_0))
    b = b_i*w_n
    print("Computing aperture - done")
    return b

def make_plots(x_array_analytic, time, aperture):

    """ Makes plot of solutions and save to file 

    Parameters
    ------------------
        x_array : numpy array
            x values of the domain
        time : float
            time in seconds
        aperture : numpy array
            aperture solution 

 
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """

    print("\nMaking plots")
    fig,ax = plt.subplots()
    ax.plot(x_array_analytic, aperture)

    filename = f"wijesinghe_solution_{int(time)}_s.png"
    print(f"Saving plots into file {filename}")
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")

def main():


    t = 50
    L = 25 
    b_i = 1.0e-5 #initial aperture
    k_n = 1.0e11 #fracture stiffness
    P_i = 11.0e6 #initial pressure
    P0 = 11.9e6 #applied pressure
    mu = 0.001 #fluid viscosity
    
    x_array = np.linspace(0, L, 100)
    aperture_analytic = aperture(t, x_array, P0, b_i,k_n,P_i,mu)

    make_plots(x_array, t, aperture_analytic)     
  
    
if __name__ == "__main__":
    main()
