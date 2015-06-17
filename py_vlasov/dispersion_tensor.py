from .util import zp, pade, zp_mp, VlasovException
import numpy as np
import scipy.special

cspeed = 299792458 #(m/s)

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

def f_abn(n, w, kz, tp, tz, vthz, Omega, vz, method='pade'):
    """
    Calculate the coefficient \A_n and \B_n, as defined in Stix(1992).
    
    Keyword arguments
    -----------------
    n: resonance number
    w: frequency (rad/s)
    kz: parallel wavenumber (rad/m)
    tp: perpendicular temperature (of the species) (Joule)
    tz: parallel temperature (Joule)
    vthz: parallel thermal speed (m/s)
    Omega: gyrofrequency (rad/s)
    vz: parallel drift (m/s)
    
    Return
    ------
    A_n and B_n
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
    zeta = f_zeta(w, kz, vz, Omega, vthz, n)
    zp_zeta = f_zp(zeta)
    an_1 = 1/w * (tp-tz)/tz
    an_2 = 1/(kz * vthz) * ((w - kz*vz - n*Omega)*tp + n*Omega*tz)/(w*tz)
    an = an_1 + an_2 * zp_zeta
    bn_1 = 1/kz * ((w-n*Omega)*tp - (kz*vz-n*Omega)*tz)/(w*tz)
    bn_2 = 1/kz * (w-n*Omega)/(kz*vthz)*((w-kz*vz-n*Omega)*tp + n*Omega*tz)/(w*tz)
    bn = bn_1 + bn_2 * zp_zeta
    return an, bn   

def f_lambda(kp, vthp, Omega):
    """
    Calculate the square of the dimensionless perpendicular wavenumber.
    
    Keyword arguments
    -----------------
    kp: perpendicular wavenumber
    vthp: perpendicular thermal speed
    Omega: gyrofrequency
    
    Return
    ------
    return \lambda = 0.5 * (k_{\perp} * \rho_p)^2
    """
    return (kp * vthp/Omega)**2 / 2

def f_yn(n, w, kz, kp, tp, tz, vthz, vthp, Omega, vz, method = 'pade'):
    """
    Calculate tensor Y_n as defined in 'Waves in plamas' (p 258, Stix 1992)
    
    Keyword arguments
    -----------------
    n: order of expansion
    w: frequency
    kz: parallel wavenumber
    kp: perpendicular wavenumber
    tp: perpendicular temperature
    tz: parallel temperature
    vthz: parallel thermal speed
    vthp: perpendicular thermal speed
    Omega: gyrofrequency
    vz: species drift
    
    Return
    ------
    return tensore Y_n
    """
    
    lamb = f_lambda(kp, vthz, Omega)
    i_n = scipy.special.iv(n, lamb)
    i_np = 0.5 * (scipy.special.iv(n-1, lamb) + scipy.special.iv(n+1, lamb))
    an, bn = f_abn(n, w, kz, tp, tz, vthz, Omega, vz, method)
    y = np.zeros((3, 3), dtype = np.cfloat)
    y[0, 0] = n**2 * i_n/lamb * an
    y[0, 1] = -1j * n * (i_n - i_np) * an
    y[0, 2] = kp/Omega * n * i_n/lamb * bn
    y[1, 0] = -y[0, 1]
    y[1, 1] = (n**2/lamb * i_n + 2*lamb * (i_n - i_np)) * an
    y[1, 2] = 1j * kp/Omega * (i_n - i_np) * bn
    y[2, 0] = y[0, 2]
    y[2, 1] = -y[1, 2]
    y[2, 2] = 2*(w - n*Omega)/(kz * vthp**2) * i_n * bn
    return y

def f_chi(n, w, kz, kp, wp, tz, tp, vthz, vthp, Omega, vz, method = 'pade'):
    """
    Calculate the susceptibility tensor \chi
    
    Keyword arguments
    -----------------
    n: number of terms to sum over
    w: frequency
    kz: parallel wavenumber
    kp: perpendicular wavenumber
    wp: plasma frequency of the species
    tz: parallel temperature
    tp: perpendicular temperature
    vthz: parallel thermal speed
    vthp: perpendicular thermal speed
    Omega: gyrofrequency
    vz: parallel drift
    
    Return
    ------
    Return the susceptibility tensor \chi for the species
    """
    chi_tensor = np.zeros((3, 3), dtype = np.cfloat)
    chi_tensor[2, 2] = 2 * wp**2/ (w * kz * vthp**2)
    lamb = f_lambda(kp, vthz, Omega)
    y_sum = np.sum(np.array([f_yn(i, w, kz, kp, tp, tz, vthz, vthp, Omega, vz, method)
                             +f_yn(-i, w, kz, kp, tp, tz, vthz, vthp, Omega, vz, method) for i in np.arange(1, n+1)]),
                            axis=0)
    y_sum += f_yn(0, w, kz, kp, tp, tz, vthz, vthp, Omega, vz, method)
    chi_tensor += wp**2 / w * np.exp(-lamb) * y_sum
    return chi_tensor

def f_epsilon(param):
    """
    Calculate the dielectric tensor \epsilon of the plasma.
    \epsilon = identity_matrix + \sum_s \chi_s
    
    Keyword arguments
    -----------------
    param: a 2D list, where param[:, j] = [n_j, w, kz, kp, wp_j, tz_j, tp_j, vthz_j, vthp_j, Omega_j, vz_j, method = 'pade']
    
    Return
    ------
    Return the dielectric tensor
    """
    
    return np.identity(3, dtype = np.cfloat) + np.sum(np.array(list(map(f_chi, *param))), axis = 0)

def f_d(param):
    """
    Calculate the dispersion tensor of the plasma.
    See Eq. 73 in 'Waves in plasmas' (Stix, 1992)
     
    Keyword arguments
    -----------------
    param: a 2D list, where param[:, j] = [n_j, w, kz, kp, wp_j, tz_j, tp_j, vthz_j, vthp_j, Omega_j, vz_j, method = 'pade']
    
    Return
    ------
    Return the dispersion tensor    
    """
    w = param[1][0]
    kz = param[2][0]
    kp = param[3][0]
    nz = kz * cspeed/ w
    nx = kp * cspeed/ w
    return f_epsilon(param) + np.array([[-nz**2, 0, nx*nz], [0, -nx**2-nz**2, 0], [nz*nx, 0, -nx**2]])

