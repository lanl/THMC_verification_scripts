__author__ = "Annie Nonimus"
__version__ = "0.1"
__maintainer__ = "Jeffrey Hyman"
__email__ = "jhyman@lanl.gov"

"""
Analytic solution for H.A. Lauweier "The transport of heat in an oil layer caused by the injection of hot fluid" (1955). Thermo-Hydro coupled solution

Injection of water at a constant rate into a single fracture in thermally active impermeable matrix. Fracture aperature is fixed.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from scipy.special import erfc
import logging
import os



def fracture_temperature(xi, theta, tau, T_inject, T0):
    """ Analytic solution for pressure

    Parameters
    -------------------
        xi : numpy array 
            dimensionless length
        theta : float
            ratio of specific heat of water and surrounding rock
        tau : float
            dimensionless time
        T_inject : float
            injection temperature
        T0 : float
            initial rock temperature

    Returns 
    -----------------
        temperature : numpy array
            temperature solution 

    Notes
    -----------------
        None
    
    """
    print("Computing Temperature (Lauwerier)")

    T_relative = (T_inject - T0)
    temperature = T_relative*erfc(xi/(2*np.sqrt(theta*(tau-xi)))) + T0

    return temperature

def matrix_temperature(xi, theta, tau, eta, T_inject, T0):
    """ Analytic solution for pressure

    Parameters
    -------------------
        xi : numpy array 
            dimensionless length
        theta : float
            ratio of specific heat of water and surrounding rock
        tau : float
            dimensionless time
        T_inject : float
            injection temperature
        T0 : float
            initial rock temperature

    Returns 
    -----------------
        temperature : numpy array
            temperature solution 

    Notes
    -----------------
        None
    
    """
    print("Computing Temperature (Lauwerier)")
    T_relative = (T_inject - T0)
    eta[(abs(eta)<1)] = 1 #constant temperature in fracture
    temperature = T_relative*erfc((xi+abs(eta)-1)/(2*np.sqrt(theta*(tau-xi)))) + T0
    return temperature

def write_files(t, temperature):
    """ Write solution data to file

    Parameters
    ------------------
        t : float
            time in seconds
        temperature : numpy array
            temperature solution 
 
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """
    temperature_filename = f'temperature_{t}.dat'
    print(f"\nWriting pressure into file: {temperature_filename}")
    np.savetxt(temperature_filename, temperature)
    print(f"Writing pressure into file: {temperature_filename} - done")


def make_plots(x_array, z_array, time, fracture_temperature_analytic, matrix_temperature_analytic):

    """ Makes plot of solutions and save to file 

    Parameters
    ------------------
        x_array : numpy array
            x values of the domain
        z_array : numpy array
            z values of the domain
        time : float
            time in years
        fracture_temperature : numpy array
            temperature solution along fracture flowline 
        matrix_temperature : numpy array
            temperature solution in matrix
    
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """

    print("\nMaking plots")    
    fig,ax = plt.subplots(nrows = 2)
    ax[0].plot(x_array, fracture_temperature_analytic)
    ax[1].plot(z_array, matrix_temperature_analytic)

    filename = f"lauwerier_solution_{time}_years.png"
    print(f"Saving plots into file {filename}")
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")

def main():

    time = 250000 # s 
    
    # Model Parameters 
    rho_r = 2757 # density of rock (kg/m^3)
    rho_w = 1000 # density of water (kg/m^3)
    
    c_r = 1180 # heat capacity of rock (j/kg/K)
    c_w = 4184 # heat capacity of water (j/kg/K)
    
    b = 1e-3 # fracture aperature (m)
    
    k_r = 0.5 # thermal conductivity rock (W/mK)
    k_w = 0.598 # thermal conductivity water (W/mK)
    
    v_w = 0.4025/100. # velocity of water in the fracture (m/s) 
    
    Lx = 20 # domain length (m)
    dx = 0.1  # spacing of grid points (m)
    Nx = np.ceil(Lx / dx).astype(int)  # number of grid points
    x_array = np.linspace(dx, Lx, Nx) # grid of points to compute solution

    Lz = 20
    dx = 0.1  # spacing of grid points (m)
    Nx = np.ceil(Lz / dx).astype(int)  # number of grid points
    z_array = np.linspace(-Lz/2, Lz/2, Nx) # grid of points to compute solution

    xi = k_r/((b/2)**2*rho_w*c_w*v_w)*x_array # dimensionless distance along fracture
    
    theta = rho_w*c_w/(rho_r*c_r) # ratio of specific heat of water and rock
    eta = z_array/(b/2) #dimensionless distance perpendicular to fracture

    T_inject = 90 # injection water temperature (degC)
    T0 = 20 # initial rock temperature (degC)
    
    tau = k_r/((b/2)**2*rho_w*c_w)*time # dimensionless time
    
    fracture_temperature_analytic = fracture_temperature(xi,theta,tau,T_inject,T0)
    matrix_temperature_analytic = matrix_temperature(xi, theta, tau, eta, T_inject, T0)

    make_plots(x_array, z_array, time, fracture_temperature_analytic, matrix_temperature_analytic)

if __name__ == "__main__":
    main()
