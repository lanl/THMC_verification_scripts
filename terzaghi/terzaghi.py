"""
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

"""

__author__ = "Yao Huang"
__version__ = "0.1"
__maintainer__ = "Jeffrey Hyman"
__email__ = "jhyman@lanl.gov"

"""
Analytic solution for Karl Terazaghi. Theoretical soil mechanics. John Wiley and Sons, 1965.

A constant compressive load is applied on the left side of a porous sample while zero displacement is specified on the right side. The porous matrix is fully saturated, and drainage is only allowed across the left side where the compressive load is applied. Fluid pressure on the drainage boundary is fixed at the initial pressure. The variables of interest here are the excess pore pressure and vertical strain as a function of time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os


def excess_pore_pressure(t, x_array, P0, Cv, L, sum_size = 100):
    """ Analytic solution for pressure

    Parameters
    -------------------
        t : float 
            Time in seconds
        x_array : numpy array
            x values of the domain
        P0 : float
            Inlet pressure
        Cv : float 
            coefficient of consolidation 
        L : float 
            Domain Length (m)
        sum_size : int
            truncation length of infinite sum. Default = 100

    Returns 
    -----------------
        pressure : numpy array
            pressure solution 

    Notes
    -----------------
        Infinite sum is approximated using the first sum_size values
    
    """
    print("Computing pressure ")
    pressure = np.zeros_like(x_array)
    for ix, x in enumerate(x_array):
        pressure[ix] = 0
        for n in range(sum_size):
            sum_value = 1 / (2 * n + 1) \
                   * np.sin( ((2 * n + 1) * np.pi * x) / (2*L)) \
                   * np.exp(-((2 * n + 1)**2 * np.pi**2 * Cv * t) / (4*L**2))
            pressure[ix] += sum_value 

    pressure = 4 * P0 / np.pi * pressure
    print("Computing pressure - done")
    return pressure


def vertical_strain(pressure, P0, E, v):
    """ Analytic solution for vertical strain 

    Parameters
    -------------------
        pressure : numpy array 
            Pressure solution obtained from excess_pore_pressure function 
        P0 : float
            Inlet pressure
        E : float 
            Young's Modulus 
        v : float 
            Poisson's Ratio

    Returns 
    -----------------
        strain : numpy array
            vertical strain solution 

    Notes
    -----------------
        None 

    """
    print("\nComputing Strain")
    strain = np.zeros_like(pressure)
    strain = (pressure - P0) * ((1 - 2 * v) * (1 + v)) / (E * (1 - v))
    print("Computing Strain - done")
    return strain


def write_files(x_array, t, pressure, epsilon):
    """ Write x, pressure, and strain data to a CSV file

    Parameters
    -------------------
        x_array : numpy array
            x values of the domain
        t : float
            time in seconds
        pressure : numpy array
            pressure solution (same shape as x_array)
        epsilon : numpy array
            vertical strain solution (same shape as x_array)

    Returns
    -----------------
        None

    Notes
    -----------------
        Saves a file named terzaghi_data_<t>.csv with columns: x, pressure, strain
    """
    print(f"\nWriting data files for t = {t} s")
    data = np.column_stack((x_array, pressure, epsilon))
    filename = f"terzaghi_data_{t}.csv"
    header = "x,pressure,strain"
    np.savetxt(filename, data, delimiter=",", header=header, comments="")
    print(f"Data saved to {filename} - done")


def make_plots(x_array, time, pressure, epsilon):

    """ Makes plot of solutions and save to file 

    Parameters
    ------------------
        x_array : numpy array
            x values of the domain
        time : float
            time in seconds
        pressure : numpy array
            pressure solution 
        epsilon : numpy array
            vertical strain solution 

    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """

    print("\nMaking plots")
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(x_array, pressure)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("Excess Pore Pressure [MPa]")
    ax[0].grid(True)  # <-- turns on grid lines

    ax[1].plot(x_array, epsilon)
    ax[1].set_ylabel("Vertical Strain")
    ax[1].set_xlabel("x [m]")
    ax[1].grid(True)  # <-- turns on grid lines

    filename = f"terzaghi_solution_{time}.png"
    print(f"Saving plots into file {filename}")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")


def main():

    P0 = 100.0e6  # Inlet Pressure (Pascals)
    k = 3e-14     # Matrix Permeability (m^2)
    E = 6.0e10    # Young's modulus (Pa)
    v = 0.2       # Poisson Ratio (dimensionless)
    K_l = 2.1e9   # Water Bulk Modulus (Pa)
    phi = 0.1     # porosity
    mu = 0.001    # Water Viscosity (Pa*s)
    
    # Cv calculation
    Cv = k / (mu * (phi / K_l))  # coefficient of consolidation 

    # Model Parameters 
    L = 24  # Domain length (m)
    x_array = np.linspace(0, L, 100)
    t = 10  # time in seconds

    print("\nParameters")
    print("--------------------------------------")
    print(f"Inlet pressure\t\t{P0} [Pa]")
    print(f"Permeability\t\t{k} [m^2]")
    print(f"Coefficient of consolidation Cv = {Cv:.3e} [m^2/s]\n")

    pressure = excess_pore_pressure(t, x_array, P0, Cv, L) / 10e6  # convert to MPa
    epsilon = vertical_strain(pressure, P0 / 10e6, E, v)  # Note: P0 scaled similarly

    # Write data to file (x, pressure, strain)
    write_files(x_array, t, pressure, epsilon)

    # Generate and save plots
    make_plots(x_array, t, pressure, epsilon)


if __name__ == "__main__":
    main()