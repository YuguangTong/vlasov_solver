from .util import zp, pade, zp_mp, VlasovException, real_imag
from .util import (pmass, emass, echarge, permittivity, permeability, cspeed, boltzmann)
from .dispersion_tensor import f_d
from .parallel_mode import r_wave_eqn
from scipy import linalg
import numpy as np

def input_gen(wrel, kpar, kperp, betap, t_list, \
              a_list, n_list, q_list, m_list, v_list, \
              n = 10, method = 'pade', aol=1/5000):
    """
    Takes in dimensionless parameters for a multiple-
    component plasma and return a input in SI units. 
    The latter is used in the 'kernel' (the main portion
    of the code).
    At this moment, THE FIRST COMPONENT IS ALWAYS PROTON.
    
    parameters
    ----------
    wrel: dimensionless wave frequency 
        \omega/\Omega_p
    kpar: dimensionless parallel wave number
        k * \rho_{p\parallel}
    kperp: dimensionless perpendicular wave number
        k * \rho_{p\parallel}
    betap: proton parallel beta
        \beta_{p\parallel}, 
    t_list: temperature ratio T_{s\parallel}/T_{p\parallel}.
        where s --> species. The first component by default represent proton.
    a_list: temperature anisotropy
        a_s \equiv T_{s\perp}/T_{s\parallel}
    n_list: density fraction
        n_s \equiv n_s/n_p, n_p --> proton density
    q_list: charge in unit of proton charge.
    m_list: mass ratio
        m_s \equiv m_s/m_p, m_p --> proton mass.
    v_list: dimensionless bulk drift.
        v_{ds} = v_{ds}/v_A, where v_A --> Alfven speed
    n: number of Bessel functions to sum over
    method: how to compute the plasma dispersion funciton.
    aol: v_a/c. 

    returns
    -------
    inp: for each plasma component provide the following list
        [n, w, kz, kp, wps, ts_par, ts_perp, vths_par, \
         vths_perp, omegas, vds, method] in SI units

    """
    b0 = 1e-8 # 10nT by default
    va = cspeed * aol # Alfven speed
    nproton = (b0/va)**2 / (permeability * pmass)
    tp_par = betap * b0**2 / (2 * permeability * nproton * boltzmann)
    omega_p = echarge * b0/pmass # proton gyrofrequency
    vthp_par = np.sqrt(2 * boltzmann * tp_par/pmass) # proton parallel thermal speed
    rhop_par = vthp_par/omega_p
    w = wrel * omega_p
    kz = kpar/rhop_par
    kp = kperp/rhop_par
    inp = []

    for i in range(len(t_list)):
        ns = nproton * n_list[i] # density
        ts_par = tp_par * t_list[i]
        ts_perp = ts_par * ( a_list[i])
        ms = pmass * m_list[i]
        vds = va * v_list[i]
        qs = echarge * q_list[i]
        wps = np.sqrt(ns * qs**2 / (ms * permittivity))
        omegas = qs * b0/ms
        vths_par = np.sqrt(2 * boltzmann * ts_par/ms) 
        vths_perp = np.sqrt(2 * boltzmann * ts_perp/ms)
        species = [n, w, kz, kp, wps, ts_par, ts_perp, vths_par, vths_perp, omegas, vds, method]
        inp += [species]
    return inp
    
    
def oblique_wrapper(wrel, kpar, kperp, betap, t_list, a_list, n_list, q_list, m_list, v_list, n = 10, method = 'pade', aol=1/5000):
    """
    Consider oblique wavenumber vectors, take in parameters 
    for a multiple-component plasma and return the determinant 
    of the dispersion matrix.
    Assume that THE FIRST COMPONENT IS ALWAYS PROTON.
    
    Kyeword arguments
    -----------------
    wrel: dimensionless wave frequency 
        \omega/\Omega_p
    kpar: dimensionless parallel wave number
        k * \rho_{p\parallel}
    kperp: dimensionless perpendicular wave number
        k * \rho_{p\parallel}
    betap: proton parallel beta
        \beta_{p\parallel}, 
    t_list: temperature ratio T_{s\parallel}/T_{p\parallel}.
        where s --> species. The first component by default represent proton.
    a_list: temperature anisotropy
        a_s \equiv T_{s\perp}/T_{s\parallel}
    n_list: density fraction
        n_s \equiv n_s/n_p, n_p --> proton density
    q_list: charge in unit of proton charge.
    m_list: mass ratio
        m_s \equiv m_s/m_p, m_p --> proton mass.
    v_list: dimensionless bulk drift.
        v_{ds} = v_{ds}/v_A, where v_A --> Alfven speed
    mode: one of 'r', 'l', 's', stands for right-handed (EM).
        left-handed, and electrostatic.

    returns
    -------
    determinant of the dispersion tensor.
    """
    inp = input_gen(wrel, kpar, kperp, betap, t_list, a_list,\
                    n_list, q_list, m_list, v_list, n, method, aol)
    omega_p = inp[0][9]
    # the 0th component is proton. the 9th term is gyrofrequency
    param = list(map(list, zip(*inp)))
    res =  linalg.det(f_d(param) * aol**2 /omega_p**2)
    return res

def parallel_wrapper(wrel, kpar, kperp, betap, t_list, a_list,
                     n_list, q_list, m_list, v_list, n = 0,
                     method = 'pade', aol=1/5000):
    """
    Consider parallel wavenumber vectors, take in parameters 
    for a multiple-component plasma and return the determinant 
    of the dispersion matrix.
    Assume that THE FIRST COMPONENT IS ALWAYS PROTON.

    parameters
    ----------
    wrel: dimensionless wave frequency 
        \omega/\Omega_p
    kpar: dimensionless parallel wave number
        k * \rho_{p\parallel}
    kperp: dimensionless perpendicular wave number
        k * \rho_{p\parallel}
    betap: proton parallel beta
        \beta_{p\parallel}, 
    t_list: temperature ratio T_{s\parallel}/T_{p\parallel}.
        where s --> species. The first component by default represent proton.
    a_list: temperature anisotropy
        a_s \equiv T_{s\perp}/T_{s\parallel}
    n_list: density fraction
        n_s \equiv n_s/n_p, n_p --> proton density
    q_list: charge in unit of proton charge.
    m_list: mass ratio
        m_s \equiv m_s/m_p, m_p --> proton mass.
    v_list: dimensionless bulk drift.
        v_{ds} = v_{ds}/v_A, where v_A --> Alfven speed
    n: number of Bessel functions to sum over
    method: how to compute the plasma dispersion funciton.
    aol: v_a/c. 

    returns
    -------
    returns the value of the dispersion equation.

    N.B.
    This wrapper does solve for electrostatic mode.
    We don't explicitly separate the RH-circularly
    polarized mode from LH-cicularly polarized mode. 
    positive real freq --> R mode
    negative real freq --> L mode
    """
    
    # propagation along B
    if kperp != 0:
        raise VlasovException("parallel mode should have kperp = 0.\n")
    inp = input_gen(wrel, kpar, kperp, betap, t_list, a_list,
                    n_list, q_list, m_list, v_list, n, method, aol)
    param = list(map(list, zip(*inp)))
    return r_wave_eqn(param)

def disp_det(wrel, kpar, kperp, betap, t_list, a_list,
                     n_list, q_list, m_list, v_list, n = 0,
                     method = 'pade', aol=1/5000):
    """
    Consider parallel or oblique waves propagating in
    a multiple-component plasma and return the determinant 
    of the dispersion matrix.
    Assume that THE FIRST COMPONENT IS ALWAYS PROTON.

    parameters
    ----------
    wrel: dimensionless wave frequency 
        \omega/\Omega_p
    kpar: dimensionless parallel wave number
        k * \rho_{p\parallel}
    kperp: dimensionless perpendicular wave number
        k * \rho_{p\parallel}
    betap: proton parallel beta
        \beta_{p\parallel}, 
    t_list: temperature ratio T_{s\parallel}/T_{p\parallel}.
        where s --> species. The first component by default represent proton.
    a_list: temperature anisotropy
        a_s \equiv T_{s\perp}/T_{s\parallel}
    n_list: density fraction
        n_s \equiv n_s/n_p, n_p --> proton density
    q_list: charge in unit of proton charge.
    m_list: mass ratio
        m_s \equiv m_s/m_p, m_p --> proton mass.
    v_list: dimensionless bulk drift.
        v_{ds} = v_{ds}/v_A, where v_A --> Alfven speed
    n: number of Bessel functions to sum over
    method: how to compute the plasma dispersion funciton.
    aol: v_a/c. 

    returns
    -------
    returns the value of the dispersion equation.

    """

    if kperp == 0:
        return  parallel_wrapper(wrel, kpar, kperp, betap, t_list, a_list,
                                 n_list, q_list, m_list, v_list, n = n,
                                 method = method, aol=aol)
    else:
        return oblique_wrapper(wrel, kpar, kperp, betap, t_list, a_list,
                               n_list, q_list, m_list, v_list, n = n,
                               method = method, aol=aol)
