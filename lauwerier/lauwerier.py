"""
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

"""

__author__ = "Annie Nonimus"
__version__ = "0.1"
__maintainer__ = "Jeffrey Hyman"
__email__ = "jhyman@lanl.gov"

"""
Analytic solution for H.A. Lauwerier "The transport of heat in an oil layer caused by the injection of hot fluid" (1955).
Thermo-Hydro coupled solution for injecting hot water into a single fracture in an impermeable, thermally‐active matrix.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from scipy.special import erfc
import logging
import os


def fracture_temperature(xi, theta, tau, T_inject, T0):
    """
    Analytic solution for fracture‐temperature:
      T_f(ξ,τ) = T0 + (T_inject − T0) * erfc[ ξ / (2 sqrt( θ(τ − ξ) )) ] * U(τ − ξ)
    i.e. if (τ ≤ ξ) we just return T0 (unit step).
    Parameters
    ----------
      xi        : numpy.ndarray, shape (Nx,)
                  dimensionless distance along fracture
      theta     : float
                  ratio of (ρ_w c_w)/(ρ_r c_r)
      tau       : float
                  dimensionless time
      T_inject  : float
                  injection temperature (°C)
      T0        : float
                  background/initial rock temperature (°C)
    Returns
    -------
      temperature : numpy.ndarray, shape (Nx,)
                    fracture‐temperature at each ξ
    """
    T_rel = T_inject - T0
    # Build argument only where (τ − ξ) > 0.  Otherwise, T = T0.
    # arg = ξ / (2 * sqrt[ θ (τ − ξ) ])
    arg = xi / (2.0 * np.sqrt(theta * (tau - xi)))
    T_f = np.where(
        tau > xi,
        T_rel * erfc(arg) + T0,
        T0
    )
    return T_f


def matrix_temperature(xi, theta, tau, eta, T_inject, T0):
    """
    Analytic solution for matrix‐temperature:
      T_m(ξ,η,τ) = T0 + (T_inject − T0) * erfc[ (ξ + |η| − 1)/(2 sqrt( θ(τ − ξ) )) ] * U(τ − ξ),
      but with |η|<1 clamped to 1 to enforce T_m(ξ, ±1, τ) = T_f(ξ, τ).
    Parameters
    ----------
      xi        : numpy.ndarray, shape (Nx,)
                  dimensionless distance along fracture
      theta     : float
                  ratio of (ρ_w c_w)/(ρ_r c_r)
      tau       : float
                  dimensionless time
      eta       : numpy.ndarray, shape (Nz,)
                  dimensionless distance perpendicular to fracture
      T_inject  : float
                  injection temperature (°C)
      T0        : float
                  background/initial rock temperature (°C)
    Returns
    -------
      temperature : numpy.ndarray, shape (Nx, Nz)
                    matrix‐temperature at each (ξ, η)
    """
    T_rel = T_inject - T0

    # 1) Compute η_eff = max(|η|, 1.0) so that |η|<1 -> 1.0
    eta_eff = np.where(np.abs(eta) < 1.0, 1.0, np.abs(eta))

    # 2) Broadcast xi→(Nx×1), eta_eff→(1×Nz) so that we get an (Nx×Nz) grid
    ξ2d = xi[:, np.newaxis]       # shape (Nx, 1)
    η2d = eta_eff[np.newaxis, :]  # shape (1, Nz)

    # 3) Build the erfc argument: (ξ + η_eff − 1) / [2 sqrt( θ (τ − ξ)) ]
    denom = 2.0 * np.sqrt(theta * (tau - ξ2d))
    arg_matrix = (ξ2d + η2d - 1.0) / denom

    # 4) Mask where τ ≤ ξ: those entries become T0
    mask = (tau > ξ2d)

    T_m = np.where(
        mask,
        T_rel * erfc(arg_matrix) + T0,
        T0
    )
    return T_m


def write_files(t, fracture_T, matrix_T, x_a, z_a):
    """
    Write solution data to disk for debugging or post‐processing.
    This will create two files:
      - fracture_T_{t}_yrs.dat    (shape: Nx)
      - matrix_T_{t}_yrs.dat      (shape: Nx × Nz)
    Parameters
    ----------
      t         : int or float
                  time (in years) used for naming
      fracture_T: numpy.ndarray, shape (Nx,)
                  fracture‐temperature at this time
      matrix_T  : numpy.ndarray, shape (Nx, Nz)
                  matrix‐temperature at this time
      x_a       : numpy.ndarray, shape (Nx,)
                  physical X array (m)
      z_a       : numpy.ndarray, shape (Nz,)
                  physical Z array (m)
    Returns
    -------
      None
    """
    # Create an output directory if desired (optional)
    # os.makedirs("output_data", exist_ok=True)

    # 1) Fracture data
    fname1 = f"fracture_T_{t}_yrs.dat"
    print(f" → Writing fracture temperature to {fname1}")
    header1 = "X(m)\tT_f(°C)\n"
    data1 = np.column_stack((x_a, fracture_T))
    np.savetxt(fname1, data1, header=header1)

    # 2) Matrix data
    #   We flatten it or write as two-dimensional block: [x_i, z_j, T_m(i,j)]
    fname2 = f"matrix_T_{t}_yrs.dat"
    print(f" → Writing matrix temperature to {fname2}")
    Nx, Nz = matrix_T.shape
    # Create a (Nx*Nz) × 3 array: [x_i, z_j, T_m(i,j)]
    Xv = np.repeat(x_a, Nz)
    Zv = np.tile(z_a, Nx)
    Tv = matrix_T.flatten()
    data2 = np.column_stack((Xv, Zv, Tv))
    header2 = "X(m)\tZ(m)\tT_m(°C)\n"
    np.savetxt(fname2, data2, header=header2)
    print("   Done writing files.\n")


def make_plots(x_a, z_a, times_years, fracture_list, matrix_list):
    """
    Create a 2×2 figure—one panel is T_f vs X, three panels are T_m vs Z at fixed X.
    We loop over times_years = [1, 5, 10], and for each, pick the corresponding fracture_list[i], matrix_list[i].
    Parameters
    ----------
      x_a           : numpy.ndarray, shape (Nx,)
                      physical X array (m)
      z_a           : numpy.ndarray, shape (Nz,)
                      physical Z array (m)
      times_years   : list of floats, e.g. [1, 5, 10]
                      times (in years) to label each curve
      fracture_list : list of numpy.ndarray, each of shape (Nx,)
                      fracture temperature arrays, one per time
      matrix_list   : list of numpy.ndarray, each of shape (Nx, Nz)
                      matrix temperature arrays, one per time
    Returns
    -------
      None (saves out 3 PNG files: one per time)
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = [f"{int(t)} yr" if float(t).is_integer() else f"{t:.1f} yr"
              for t in times_years]

    # We will save each time separately, but produce the same 2×2 layout,
    # so we loop over times_years, fracture_list, matrix_list in parallel.
    for idx, t_yr in enumerate(times_years):
        Tf = fracture_list[idx]      # shape (Nx,)
        Tm = matrix_list[idx]        # shape (Nx, Nz)

        # Create a 2×2 figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Lauwerier Analytic Solution (t = {t_yr} yr)", fontsize=16)

        # — Top-Left panel: T_f vs X
        ax00 = axes[0, 0]
        ax00.plot(x_a, Tf, color=colors[idx], label=f"Solution @ {labels[idx]}")
        ax00.set_xlabel("X (m)")
        ax00.set_ylabel("Fracture Temp (°C)")
        ax00.set_title("Fracture Temperature vs X")
        ax00.grid(True)
        ax00.legend()

        # We need three “Z‐profiles” at X = 10 m, 50 m, and 90 m.
        # Find the nearest indices in x_a for those physical locations:
        def find_index(x_array, target):
            return np.argmin(np.abs(x_array - target))

        idx10 = find_index(x_a, 10.0)
        idx50 = find_index(x_a, 50.0)
        idx90 = find_index(x_a, 90.0)

        # — Top-Right panel: T_m vs Z at X=10 m
        ax01 = axes[0, 1]
        ax01.plot(z_a, Tm[idx10, :], color=colors[idx], label=labels[idx])
        ax01.set_xlabel("Z (m)")
        ax01.set_ylabel("Temp (°C)")
        ax01.set_title("Matrix Temp vs Z @ X = 10 m")
        ax01.grid(True)
        ax01.legend()

        # — Bottom-Left panel: T_m vs Z at X=50 m
        ax10 = axes[1, 0]
        ax10.plot(z_a, Tm[idx50, :], color=colors[idx], label=labels[idx])
        ax10.set_xlabel("Z (m)")
        ax10.set_ylabel("Temp (°C)")
        ax10.set_title("Matrix Temp vs Z @ X = 50 m")
        ax10.grid(True)
        ax10.legend()

        # — Bottom-Right panel: T_m vs Z at X=90 m
        ax11 = axes[1, 1]
        ax11.plot(z_a, Tm[idx90, :], color=colors[idx], label=labels[idx])
        ax11.set_xlabel("Z (m)")
        ax11.set_ylabel("Temp (°C)")
        ax11.set_title("Matrix Temp vs Z @ X = 90 m")
        ax11.grid(True)
        ax11.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        outname = f"lauwerier_solution_{int(t_yr)}_years.png"
        print(f" → Saving figure: {outname}")
        plt.savefig(outname)
        plt.close(fig)


def main():
    # Material + geometry parameters (unchanged)
    rho_r   = 2757        # kg/m^3 (rock)
    rho_w   = 1000        # kg/m^3 (water)
    c_r     = 1180        # J/kg/K (rock)
    c_w     = 4184        # J/kg/K (water)
    b       = 1e-3        # m (fracture aperture)
    k_r     = 0.5         # W/m/K (rock conductivity)
    v_w     = 0.4025/100  # m/s (water velocity)

    # Physical domain in meters
    Lx = 100.0   # fracture length [0..100 m]
    dx = 1.0
    Nx = int(np.ceil(Lx / dx))
    x_a = np.linspace(dx, Lx, Nx)

    Lz = 100.0   # matrix depth [−50..+50 m]
    dz = 1.0
    Nz = int(np.ceil(Lz / dz))
    z_a = np.linspace(-Lz/2, Lz/2, Nz)

    # Dimensionless variables
    xi    = (k_r / ((b/2)**2 * rho_w * c_w * v_w)) * x_a
    theta = (rho_w * c_w) / (rho_r * c_r)
    eta   = z_a / (b/2)

    # Convert “years” into seconds
    sec_per_year = 365.0 * 24.0 * 3600.0

    # We want to run exactly at t = 1 yr, 5 yr, and 10 yr
    times_years = [1, 5, 10]
    times_secs  = [t * sec_per_year for t in times_years]

    # Dimensionless times (τ = (k_r/((b/2)^2 ρ_w c_w)) * t_phys )
    taus = [(k_r / ((b/2)**2 * rho_w * c_w)) * t_phys for t_phys in times_secs]

    # Injection / initial temperatures
    T_inject = 90.0  # °C
    T0       = 20.0  # °C

    # Pre‐allocate lists to hold each “snapshot” of T_f and T_m
    fracture_list = []
    matrix_list   = []

    for tau in taus:
        # Compute fracture temperature (Nx,)
        Tf = fracture_temperature(xi, theta, tau, T_inject, T0)
        fracture_list.append(Tf)

        # Compute matrix temperature (Nx, Nz)
        Tm = matrix_temperature(xi, theta, tau, eta, T_inject, T0)
        matrix_list.append(Tm)

    # Optionally write raw .dat files for each time
    for idx, t_yr in enumerate(times_years):
        Tf = fracture_list[idx]
        Tm = matrix_list[idx]
        write_files(t_yr, Tf, Tm, x_a, z_a)

    # Make and save the 2×2 panels for each time
    make_plots(x_a, z_a, times_years, fracture_list, matrix_list)


if __name__ == "__main__":
    main()