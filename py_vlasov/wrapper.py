from .util import zp, pade, zp_mp, VlasovException, real_imag
from .util import (pmass, emass, echarge, permittivity, permeability, cspeed, boltzmann)
from .dispersion_tensor import f_d
from scipy import linalg
import numpy as np

def oblique_wrapper(wrel, kpar, kperp, betap, t_list, a_list, n_list, q_list, m_list, v_list, n = 10, method = 'pade', aol=1/500):
    """
    Takes in parameters for a multiple-component plasma 
    and return the determinant of the dispersion matrix.
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
    kz = kpar/rhop_par
    kp = kperp/rhop_par
    inp = []

    for i in range(len(t_list)):
        ns = nproton * n_list[i] # density
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
    return linalg.det(f_d(param))*1e-40 