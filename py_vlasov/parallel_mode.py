from .util import zp, pade, zp_mp, VlasovException, real_imag
from .util import (pmass, emass, echarge, permittivity, permeability, cspeed, boltzmann)
import numpy as np
import scipy.optimize

def f_zeta(w, kz, vz, Omega, vthz, n):
    """
    Calculate the argument of plasma dispersion function.
    
    Keyword arguments
    -----------------
    w: frequency (rad/s)
    kz: parallel wavenumber (rad/m)
    vz: parallel drift of the particle species (m/s)
    Omega: gyrofrequency of the species (rad/s)
    vthz: parallel thermal speed (m/s)
    n: resonance number
    
    Return
    ------
    \zeta_{ns}
    """
    return (w-kz*vz-n*Omega)/(kz*vthz)

def choose_zp_fn(method):
    """
    choose which function to calclate the plasma dispersion function.
    
    Keyword arguments
    -----------------
    method: a string in ['pade', 'numpy', 'mpmath']
    
    Return
    ------
    return the pointer to the function object.
      
    """
    if method == 'pade':
        f_zp = pade
    elif method == 'numpy':
        f_zp = zp
    elif method == 'mpmath':
        f_zp = zp_mp
    else:
        raise VlasovException("Unreconized method.\n" +
            "Please choose between 'pade', 'numpy' and 'mpmath'")
    return f_zp

def r_wave_rhs(n, w, kz, kp, wp, tz, tp, vthz, vthp, Omega, vz, method = 'pade'):
    """

    Keyword arguments
    -----------------
    n: number of terms to sum over. do not need for parallel propagation.
    w: frequency
    kz: parallel wavenumber
    kp: perpendicular wavenumber. kp = 0 for parallel propagation.
    wp: plasma frequency of the species
    tz: parallel temperature
    tp: perpendicular temperature
    vthz: parallel thermal speed
    vthp: perpendicular thermal speed
    Omega: gyrofrequency
    vz: parallel drift
    
    Return
    ------
    The value of the summed term on the RHS of Eq (2), P. 267, Stix (1992). 
    Eq (2) yields R wave.
    """
    term_1 = (tp-tz)/tz
    term_2 = ((w - kz*vz + Omega)*tp - Omega*tz)/(kz * vthz * tz)
    f_zp = choose_zp_fn(method)
    zeta = f_zeta(w, kz, vz, Omega, vthz, -1)
    term_3 = f_zp(zeta)
    rhs = wp**2 * (term_1 + term_2 * term_3)
    return rhs

def l_wave_rhs(n, w, kz, kp, wp, tz, tp, vthz, vthp, Omega, vz, method = 'pade'):
    """

    Keyword arguments
    -----------------
    n: number of terms to sum over. do not need for parallel propagation.
    w: frequency
    kz: parallel wavenumber
    kp: perpendicular wavenumber. kp = 0 for parallel propagation.
    wp: plasma frequency of the species
    tz: parallel temperature
    tp: perpendicular temperature
    vthz: parallel thermal speed
    vthp: perpendicular thermal speed
    Omega: gyrofrequency
    vz: parallel drift
    
    Return
    ------
    The value of an the summed term in Eq (3), P. 267, Stix (1992). 
    Eq (3) yields L wave.
    """
    term_1 = (tp-tz)/tz
    term_2 = ((w - kz*vz - Omega)*tp + Omega*tz)/(kz * vthz * tz)
    f_zp = choose_zp_fn(method)
    zeta = f_zeta(w, kz, vz, Omega, vthz, 1)
    term_3 = f_zp(zeta)
    rhs = wp**2 * (term_1 + term_2 * term_3)
    return rhs

def static_rhs(n, w, kz, kp, wp, tz, tp, vthz, vthp, Omega, vz, method = 'pade'):
    """

    Keyword arguments
    -----------------
    n: number of terms to sum over. do not need for parallel propagation.
    w: frequency
    kz: parallel wavenumber
    kp: perpendicular wavenumber. kp = 0 for parallel propagation.
    wp: plasma frequency of the species
    tz: parallel temperature
    tp: perpendicular temperature
    vthz: parallel thermal speed
    vthp: perpendicular thermal speed
    Omega: gyrofrequency
    vz: parallel drift
    
    Return
    ------
    The value of an the summed term in Eq (4), P. 267, Stix (1992). 
    Eq (4) yields electrostatic wave.
    """
    term_1 = 2 * (wp/ kz / vthz)**2
    term_2 = (w - kz * vz)/ (kz * vthz)
    f_zp = choose_zp_fn(method)
    zeta = f_zeta(w, kz, vz, Omega, vthz, 0)
    term_3 = f_zp(zeta)
    rhs = term_1 * (1 + term_2 * term_3)
    return rhs

def r_wave_eqn(param):
    """
    Keyword arguments
    -----------------
    param: a 2D list, where param[:, j] = [n_j, w, kz, kp, wp_j, tz_j, tp_j, vthz_j, vthp_j, Omega_j, vz_j, method = 'pade']
    
    Return
    ------
    Return the value of dispersion equation for R wave.    
    """
    w = param[1][0]
    kz = param[2][0]
    return w**2 + np.sum(np.array(list(map(r_wave_rhs, *param))), axis = 0) - (kz * cspeed)**2

def l_wave_eqn(param):
    """
    Keyword arguments
    -----------------
    param: a 2D list, where param[:, j] = [n_j, w, kz, kp, wp_j, tz_j, tp_j, vthz_j, vthp_j, Omega_j, vz_j, method = 'pade']
    
    Return
    ------
    Return the value of dispersion equation for L wave.    
    """
    w = param[1][0]
    kz = param[2][0]
    return w**2 + np.sum(np.array(list(map(l_wave_rhs, *param))), axis = 0) - (kz * cspeed)**2

def static_wave_eqn(param):
    """
    Keyword arguments
    -----------------
    param: a 2D list, where param[:, j] = [n_j, w, kz, kp, wp_j, tz_j, tp_j, vthz_j, vthp_j, Omega_j, vz_j, method = 'pade']
    
    Return
    ------
    Return the value of dispersion equation for electrostatic waves.    
    """
    return 1 + np.sum(np.array(list(map(static_rhs, *param))), axis = 0)

def parallel_em_wave_wrapper(wrel, k, betap, t_list, a_list, n_list, q_list,
                             m_list, v_list, method = 'pade', aol=1/5000, mode='r'):
    """
    A more systematic way to consider multiple component plasmas.
    Assume that THE FIRST COMPONENT IS ALWAYS PROTON.
    
    Kyeword arguments
    -----------------
    wrel: dimensionless wave frequency 
        \omega/\Omega_p
    k: dimensionless wave number
        k * \rho_{p\parallel}
    betap: proton parallel beta
        \beta_{p\parallel}, 
    t_list: temperature ratio T_{s\parallel}/T_{p\parallel}.
        where s --> species. The first component by default represent proton.
    a_list: temperature anisotropy
        a_s \equiv 1 - T_{s\perp}/T_{s\parallel}
    n_list: density fraction
        n_s \equiv n_s/n_p, n_p --> proton density
    q_list: charge in unit of proton charge.
    m_list: mass ratio
        m_s \equiv m_s/m_p, m_p --> proton mass.
    v_list: dimensionless bulk drift.
        v_{ds} = v_{ds}/v_A, where v_A --> Alfven speed
    mode: one of 'r', 'l', 's', stands for right-handed (EM).
        left-handed, and electrostatic.
    """
    b0 = 1e-8 # 10nT by default
    va = cspeed * aol # Alfven speed
    nproton = (b0/va)**2 / (permeability * pmass)
    tp_par = betap * b0**2 / (2 * permeability * nproton * boltzmann)
    omega_p = echarge * b0/pmass # proton gyrofrequency
    vthp_par = np.sqrt(2 * boltzmann * tp_par/pmass) # proton parallel thermal speed
    rhop_par = vthp_par/omega_p
    w = wrel * omega_p
    kz = k/rhop_par
    kp = 0 # parallel propogation.
    n = 0 # no summation for parallel modes. Unnecessary parameters
    inp = []
    
    for i in range(len(t_list)):
        ns = nproton * n_list[i]
        ts_par = tp_par * t_list[i]
        ts_perp = ts_par * (1 - a_list[i])
        ms = pmass * m_list[i]
        vds = va * v_list[i]
        qs = echarge * q_list[i]
        wps = np.sqrt(ns * qs**2 / (ms * permittivity))
        omegas = qs * b0/ms
        vths_par = np.sqrt(2 * boltzmann * ts_par/ms) 
        vths_perp = np.sqrt(2 * boltzmann * ts_perp/ms)
        species = [n, w, kz, kp, wps, ts_par, ts_perp, vths_par, vths_perp, omegas, vds, method]
        inp += [species]
        
    param = list(map(list, zip(*inp)))
    if mode == 'r':
        res = r_wave_eqn(param) 
    elif mode == 'l':
        res = l_wave_eqn(param) 
    elif mode == 's':
        res = static_eqn(param)
    return res

