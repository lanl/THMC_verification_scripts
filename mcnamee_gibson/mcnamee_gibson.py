"""
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

"""

__author__ = "Xiang Huang"
__version__ = "0.1"
__maintainer__ = "Jeffrey Hyman"
__email__ = "jhyman@lanl.gov"

"""
Analytic solution for 2D coupled hydromechanics Mcnamee-Gibson Problem (McNamee&Gibson, 1960a, 1960b).
McNamee J, Gibson RE (1960a) Displacement functions and linear transforms applied to diffusion through porous elastic media. Q J Mech Appl Math 13(1):98–111
McNamee J, Gibson RE (1960b) Plain strain and axially symmetric problems of the consolidation of a semi-infinite clay stratum. Q J Mech Appl Math 13(2):210–227

Two-dimensional consolidation problem and soil is fully saturated.
A constant face load is applied on the partial side of the top surface (with length a) of a sample while the roller/no-flux boundary conditions are applied to the left/right/bottom three surfaces.
Drainage is only allowed across the top surface.
Single-phase flow with incompressible water.
Plane strain (x-z system), no gravity, no source/sinks.
"""

import math
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


def excess_pore_pressure_left(t, z_array, P0, L0, Cv, v):
    """ Analytic solution for pressure along the left boundary

    Parameters
    -------------------
        t : float 
            Time in seconds
        z_array : numpy array
            x values of the left boundary of the domain
        P0 : float
            Strip loading pressure (Pa)
        L0 : float
            Strip loading length 
        Cv : float 
            coefficient of consolidation 
        v  : float 
            Poisson's ratio

    Returns 
    -----------------
        pressure : numpy array
            pressure solution (along the left boundary) given the elapsed time

    Notes
    -----------------
        none
    
    """
    print("Computing pressure ")
    
    t=t/(L0**2/Cv)       # Dimensionless time
    zz=z_array/L0        # Dimensionless z-coordinate
    x=0                  # Left boundary in x-z system x equals 0
    eta = (1-v)/(1-2*v)  # Auxiliary elastic constant
    var1 = [
        eta/(2*eta -1)
        * quad(lambda ksi: 2/np.pi/ksi*np.cos(x*ksi)*np.sin(ksi)*np.exp(-ksi*z)* \
                           (1+math.erf(ksi*np.sqrt(t))+ \
                              (eta-1)/eta*np.exp((1-2*eta)*ksi**2*t/eta/eta)* \
                                 math.erfc((eta-1)/eta*ksi*np.sqrt(t)) ),
            0,
            np.inf,
            epsabs=1e-15,
            limit=1000,
        )[0]
        for z in zz
    ]

    var2 = [
        eta/(2*eta -1)
        * quad(lambda ksi: 2/np.pi/ksi*np.cos(x*ksi)*np.sin(ksi)*  \
                          (np.exp(-ksi*z)*math.erfc(z/2/np.sqrt(t)-ksi*np.sqrt(t))+ \
                            (eta-1)/eta*np.exp((1-2*eta)*ksi**2*t/eta/eta+(eta-1/eta)*ksi*z)* \
                               math.erfc((eta-1)/eta*ksi*np.sqrt(t)+z/2/np.sqrt(t)) ),
            0,
            np.inf,
            epsabs=1e-15,
            limit=1000,
        )[0]
        for z in zz
    ]    

    pressure1 = (np.array (var1) - np.array (var2))  # dimensionless pressure
    
    # converting the dimensionless pressure to dimensional pore pressure
    pressure  = pressure1*P0                         
   
    print("Computing pressure - done")
    return pressure


def displacement_top(t, x_array, P0, L0, Cv, v, G):
    """ Analytic solution for pressure along the top boundary

    -------------------
        t : float 
            Time in seconds
        x_array : numpy array
            x values of the left boundary of the domain
        P0 : float
            Strip loading pressure (Pa)
        L0 : float
            Strip loading length 
        Cv : float 
            coefficient of consolidation 
        v  : float 
            Poisson's ratio
        G  : float 
            Shear modulus
               
    Returns 
    -----------------
        displacement : numpy array
           displacement solution (along the top boundary) given the elapsed time

    Notes
    -----------------
        None 

    """
    print("\nComputing displacement (with increased limlst)...")
    
    t_dimless = t / (L0**2 / Cv)
    xx        = x_array / L0
    eta       = (1 - v) / (1 - 2 * v)
    
    eps = 1e-8
    
    def f_kernel(ksi, t_rel):
        bracket = (
            math.erf(ksi * math.sqrt(t_rel))
            - (eta - 1) / eta
              * (1 - math.exp((1 - 2 * eta) * ksi**2 * t_rel / eta**2)
                     * math.erfc((eta - 1) / eta * ksi * math.sqrt(t_rel)))
        )
        # (sin(κ)/κ) is stable near κ=0, and bracket/κ is O(1) near κ=0.
        return (
            (eta / (2 * eta - 1))
            * (2 / np.pi)
            * (math.sin(ksi) / ksi)   # = sinc(κ)
            * (bracket / ksi)
        )
    
    var1 = []
    for x in xx:
        # By adding limlst=500, we allow up to 500 "oscillatory cycles" to be handled.
        val, err = quad(
            f_kernel,
            eps,
            np.inf,
            args=(t_dimless,),
            weight='cos',
            wvar=x,
            epsabs=1e-6,
            epsrel=1e-6,
            limlst=500
        )
        var1.append(val)
        
    """
    # calc initial instant displacement, but this integrand requires high precision
        var2 = [
            1/2/G
            * quad(lambda E: 1/E*2/np.pi/E*np.cos(x*E)*np.sin(E),
                0,
    #            np.inf,
                5000,
                epsabs=1e-5,
                epsrel=1e-5,
                limit=100000
            )[0]
            for x in xx
         ]    
    
    # converting the dimensionless displacement to the actual displacement 
    displacement  = ( np.array(var1)/2/G + np.array(var2) ) *p0*L0  
    """     
    
    displacement1 = np.array(var1)            # dimensionless 2·G·δu
    displacement  = displacement1 / (2 * G) * P0 * L0  # [m]
    
    print("Computing displacement (with increased limlst) - done")
    return displacement


def excess_pore_pressure_left_point(t_array, z_array, P0, L0, Cv, v):
    """ Analytic solution for pressure at specific obs points on the left boundary

    Parameters
    -------------------
        t_array : numpy array 
            Time in seconds
        z_array : float
            z values of the left boundary of the domain
        P0 : float
            Strip loading pressure (Pa)
        L0 : float
            Strip loading length 
        Cv : float 
            coefficient of consolidation 
        v  : float 
            Poisson's ratio

    Returns 
    -----------------
        pressure : numpy array
            pressure solution at specific obs points against the elapsed time

    Notes
    -----------------
        none
    
    """
    print("Computing pressure at specific points ")
    
    tt=t_array/(L0**2/Cv) # Dimensionless time
    z=z_array/L0          # Dimensionless z-coordinate, typically choosing 0.5, 1.0, or 1.5
    x=0                   # Left boundary in x-z system x equals 0
    eta = (1-v)/(1-2*v)   # Auxiliary elastic constant
    var1 = [
        eta/(2*eta -1)
        * quad(lambda ksi: 2/np.pi/ksi*np.cos(x*ksi)*np.sin(ksi)*np.exp(-ksi*z)* \
                           (1+math.erf(ksi*np.sqrt(t))+ \
                              (eta-1)/eta*np.exp((1-2*eta)*ksi**2*t/eta/eta)* \
                                 math.erfc((eta-1)/eta*ksi*np.sqrt(t)) ),
            0,
            np.inf,
            epsabs=1e-15,
            limit=1000,
        )[0]
        for t in tt
    ]

    var2 = [
        eta/(2*eta -1)
        * quad(lambda ksi: 2/np.pi/ksi*np.cos(x*ksi)*np.sin(ksi)*  \
                          (np.exp(-ksi*z)*math.erfc(z/2/np.sqrt(t)-ksi*np.sqrt(t))+ \
                            (eta-1)/eta*np.exp((1-2*eta)*ksi**2*t/eta/eta+(eta-1/eta)*ksi*z)* \
                               math.erfc((eta-1)/eta*ksi*np.sqrt(t)+z/2/np.sqrt(t)) ),
            0,
            np.inf,
            epsabs=1e-15,
            limit=1000,
        )[0]
        for t in tt
    ]    

    pressure1 = (np.array (var1) - np.array (var2))  # dimensionless pressure
    
    # converting the dimensionless pressure to dimensional pore pressure
    pressure  = pressure1*P0                         
   
    print("Computing pressure at specific points - done")
    return pressure


def displacement_top_point(t_array, x_array, P0, L0, Cv, v, G):
    """ Analytic solution for pressure at specific obs points on the top boundary

    -------------------
        t_array : numpy array
            Time in seconds
        x_array : float 
            x values of the top boundary of the domain
        p0 : float
            Strip loading pressure (Pa)
        L0 : float
            Strip loading length 
        Cv : float 
            coefficient of consolidation 
        v  : float 
            Poisson's ratio
        G  : float 
            Shear modulus
               
    Returns 
    -----------------
        displacement : numpy array
           displacement solution at specific obs points against the elapsed time

    Notes
    -----------------
        None 

    """
    print("\nComputing displacement at specific points")
    
    tt=t_array/(L0**2/Cv) # Dimensionless time
    x=x_array/L0          # Dimensionless x-coordinate, typically choosing 0.5, 1.0, or 1.5
    z=0                   # Top boundary in x-z system z equals 0
    eta = (1-v)/(1-2*v)   # Auxiliary elastic constant    
#    E=60.0E9             # Young's modulus 
#    p0=1E6               # Strip Loading Pressure (Pascals)
#    G=E0/2/(1+v)         # Shear modulus  

    var1 = [
        eta/(2*eta -1)
        * quad(lambda ksi: 1/ksi*2/np.pi/ksi*np.cos(x*ksi)*np.sin(ksi)* \
                           ( math.erf(ksi*np.sqrt(t))- \
                              (eta-1)/eta*
                                 ( 1-np.exp((1-2*eta)*ksi**2*t/eta/eta)* \
                                     math.erfc((eta-1)/eta*ksi*np.sqrt(t)) )   ),
            0,
            np.inf,
#           5000,
            epsabs=1e-5,
            epsrel=1e-5,
            limit=100000
        )[0]
        for t in tt
       ]
        
    """
    # calc initial instant displacement, but this integrand requires high precision
        var2 = [
            1/2/G
            * quad(lambda E: 1/E*2/np.pi/E*np.cos(x*E)*np.sin(E),
                0,
    #            np.inf,
                5000,
                epsabs=1e-5,
                epsrel=1e-5,
                limit=100000
            )[0]
            for t in tt
         ]    
    
    # converting the dimensionless displacement to actual displacement 
    displacement  = ( np.array(var1)/2/G + np.array(var2) ) *p0*L0  
    """      
    
    displacement1  = np.array(var1)          # this is dimensionless 2*G*delta_u 
    displacement  = np.array(var1)/2/G*P0*L0 # this is dimensional delta_u [m]  

    print("Computing displacement at specific points - done")
    return displacement


def write_files(time, x_array, pressure, displacement):
    """ Write solution data to file

    Parameters
    ------------------
        time : float
            time in seconds
        x_array: numpy array
             x (or z) values of the top (or left) boundary of the domain
        pressure : numpy array
            pressure solution 
        displacement : numpy array
            vertical strain solution 
 
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """
    pressure_filename = f'pressure_left_boundary_{time}s.dat'
    print(f"\nWritting pressure into file: {pressure_filename}")
    np.savetxt(pressure_filename, np.column_stack([x_array, pressure]), 
           header="z_array,pressure", delimiter=",", comments='')
    print(f"Writting pressure into file: {pressure_filename} - done")

    displacement_filename = f'displacement_top_boundary_{time}s.dat'
    print(f"\nWritting vertical displacement into file: {displacement_filename}")
    np.savetxt(displacement_filename, np.column_stack([x_array, displacement]), 
           header="x_array,displacement", delimiter=",", comments='')
    print(f"Writting vertical displacement into file: {displacement_filename} - done")
    
    
def write_files_point(time, x_array, pressure, displacement):
    """ Write solution data to file

    Parameters
    ------------------
        time : float
            time in seconds
        x_array: numpy array
            x (or z) values of the top (or left) boundary of the domain
        pressure : numpy array
            pressure solution 
        displacement : numpy array
            vertical strain solution 
 
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """
    pressure_filename = f'pressure_left_boundary_point_{x_array}m.dat'
    print(f"\nWritting pressure into file: {pressure_filename}")
    np.savetxt(pressure_filename, np.column_stack([time, pressure]), 
                header="time,pressure", delimiter=",", comments='')
    print(f"Writting pressure into file: {pressure_filename} - done")

    displacement_filename = f'displacement_top_boundary_point_{x_array}m.dat'
    print(f"\nWritting vertical displacement into file: {displacement_filename}")
    np.savetxt(displacement_filename, np.column_stack([time, displacement]), 
                header="time,displacement", delimiter=",", comments='')
    print(f"Writting vertical displacement into file: {displacement_filename} - done")


def make_plots(time, x_array, pressure, displacement):

    """ Makes plot of solutions and save to file 

    Parameters
    ------------------
        time : float
            time in seconds
        x_array : numpy array
            x values of the domain
        pressure : numpy array
            pressure solution 
        displacement : numpy array
            displacement solution 
 
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """
    print("\nMaking plots")

    fig1,ax1 = plt.subplots(nrows=1)
    ax1.plot(x_array, pressure)
    ax1.set_xlabel(u'z-distance [m]')
    ax1.set_ylabel(u'Pressure [Pa]') 
    filename = f"McName-Gibson_pressure_solution_left_boundary_{time}s.png"
    plt.title(filename)
    print(f"Saving plots into file {filename}")
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")
    
    fig2,ax2 = plt.subplots(nrows=1)
    ax2.plot(x_array, displacement)
    ax2.set_xlabel(u'x-distance [m]')
    ax2.set_ylabel(u'Displacement \u0394u [m]') 
    filename = f"McName-Gibson_displacement_solution_top_boundary_{time}s.png"
    plt.title(filename)
    print(f"Saving plots into file {filename}")
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")
    
    
def make_plots_point(time, obs_point_loc, pressure, displacement):

    """ Makes plot of solutions and save to file 

    Parameters
    ------------------
        time : float
            time in seconds
        obs_point_loc : observation point
        pressure : numpy array
            pressure solution 
        displacement : numpy array
            displacement solution 
 
    Returns
    --------------
        None

    Notes
    ---------------
        None 
    
    """
    print("\nMaking plots")

    fig1,ax1 = plt.subplots(nrows=1)
    ax1.plot(time, pressure)
    ax1.set_xlabel(u'Time [s]')
    ax1.set_ylabel(u'Pressure [Pa]') 
    ax1.set_xscale('linear')
    filename = f"McName-Gibson_pressure_solution_left_boundary_point_{obs_point_loc}m.png"
    plt.title(filename)
    print(f"Saving plots into file {filename}")
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")
    
    fig2,ax2 = plt.subplots(nrows=1)
    ax2.plot(time, displacement)
    ax2.set_xlabel(u'Time [s]')
    ax2.set_ylabel(u'Displacement \u0394u [m]') 
    ax2.set_xscale('linear')
    filename = f"McName-Gibson_displacement_solution_top_boundary_point_{obs_point_loc}m.png"
    plt.title(filename)
    print(f"Saving plots into file {filename}")
    plt.savefig(filename)
    print(f"Saving plots into file {filename} - done")


def main():
    print("Computing solution for McName-Gibson 1960")
    # Model Parameters 
    P0 = 7e3     # Strip loading pressure (Pascals)
    L0 = 1.0         # Strip loading length (meters)
    k = 1.0e-12    # Matrix permeability (m^2) =50[mD] 
    phi = 0.2      # Porosity 
    E  = 1.0E9    # Young's modulus
    v  = 0.0       # Poisson's ratio
    biot  = 1.0       # Biot coefficient
    Cf = 0         # Fluid compressbility
    Cm = 1e-10     # Porous medium compressbility
    miu = 0.001     # Dynamic viscosity
    G = E / (2 * (1+v)) # Shear modulus    
    K_c = E*(1 - v) / ((1 + v)*(1 - 2*v)) # Constrained modulus
    # M  = 1 /( (por*Cf + (b-por)*(1-b)*Cm) ) # Biot modulus
    ss = phi*Cf + (biot-phi)*(1-biot)*Cm # Specfic storage
    Cv = k*K_c / miu / (K_c*ss+biot**2) # Coefficient of consolidation 
    

    print("\nParameters")
    print("--------------------------------------")
    print(f"Strip loading pressure\t\t{P0} [Pa]")
    print(f"Strip loading length\t\t{L0} [m]")
    print("")
    
    #########################################################################
    # calculate pressure along the left boundary at the end of the given elapsed time
    # calculate displacement along the top boundary at the end of the given elapsed time
    z = np.concatenate( (np.linspace(0.01, 0.1, 100), np.linspace(0.1, 1, 19), np.linspace(1, 10, 100)) )
    
    t = 1.0
    pressure     = excess_pore_pressure_left(t, z, P0, L0, Cv, v)
    displacement = displacement_top(t, z, P0, L0, Cv, v, G)
    write_files(t, z, pressure, displacement)
    make_plots(t, z, pressure, displacement)
    
    t = 1e-2
    pressure     = excess_pore_pressure_left(t, z, P0, L0, Cv, v)
    displacement = displacement_top(t, z, P0, L0, Cv, v, G)
    write_files(t, z, pressure, displacement)
    make_plots(t, z, pressure, displacement)
    
    t = 1e-4
    pressure     = excess_pore_pressure_left(t, z, P0, L0, Cv, v)
    displacement = displacement_top(t, z, P0, L0, Cv, v, G)
    write_files(t, z, pressure, displacement)
    make_plots(t, z, pressure, displacement)

    #########################################################################
    # calculate pressure at the specific points on the left boundary
    # calculate displacement at the specific points on the left boundary
    t_array = np.linspace(1e-15, 2.0, 100)
    
    obs_point_loc = 0.5*L0
    pressure     = excess_pore_pressure_left_point(t_array, obs_point_loc, P0, L0, Cv, v)
    displacement = displacement_top_point(t_array, obs_point_loc, P0, L0, Cv, v, G)
    write_files_point(t_array, obs_point_loc, pressure, displacement)
    make_plots_point(t_array, obs_point_loc, pressure, displacement)
    
    obs_point_loc = 1.0*L0
    pressure     = excess_pore_pressure_left_point(t_array, obs_point_loc, P0, L0, Cv, v)
    displacement = displacement_top_point(t_array, obs_point_loc, P0, L0, Cv, v, G)
    write_files_point(t_array, obs_point_loc, pressure, displacement)
    make_plots_point(t_array, obs_point_loc, pressure, displacement)
    
    obs_point_loc = 1.5*L0
    pressure     = excess_pore_pressure_left_point(t_array, obs_point_loc, P0, L0, Cv, v)
    displacement = displacement_top_point(t_array, obs_point_loc, P0, L0, Cv, v, G)
    write_files_point(t_array, obs_point_loc, pressure, displacement)
    make_plots_point(t_array, obs_point_loc, pressure, displacement)
    
    print("\nComputing solution for McName-Gibson 1960\n")

if __name__ == "__main__":
    main()

