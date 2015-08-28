import numpy as np
from scipy import linalg
from .dispersion_tensor import dt_wrapper, f_d, f_chi
from .util import (real_imag, list_to_complex, nullspace)
from .util import (pmass, emass, echarge, permittivity, permeability, cspeed, boltzmann)

def input_gen(wrel, kperp, kpar, betap, tetp = 1, method = 'pade', mratio=1836, n=10, aol=1/5000):
    """
    Generate input for f_d(), f_chi() functions from dimensionless plasma parameters
    Assume that there are only protons and electrons.
    No temperature anisotropy or bulk drifts. 
    
    Keyword arguments
    -----------------
    kperp: k * rho_perp
    kpar: k * rho_parallel
    betap: proton plasma beta
    tetp: electron proton temperature ratio
    method: algorithm to calculate plasma dispersion function. \
            by default, pade approximation is used for speed. \
            'numpy' refers to complex error function modules in numpy/scipy library \
            which implement a fast algorithm Pope (1990) in C++.
    mratio: mass ratio, m_p/m_e. If mratio != 1836, electron mass is adjusted.
    n: the number of Bessel functions in summation.
    aol: va/c, where c = lightspeed.
    
    Return
    ------
    determinant of dispersion tensor
    
    """
    if mratio == 1836:
        emass_local = emass
    else:
        emass_local = pmass/mratio
        
    # by default add over 10 terms
    b0 = 1e-8      # 10nT by default
    vz = 0         # no bulk drift
    va = cspeed * aol
    nproton = (b0/va)**2 / (permeability * pmass)
    tp = betap * b0**2 / (2 * permeability * nproton * boltzmann)
    te = tp * tetp
    wpp = np.sqrt(nproton * echarge**2 / (pmass * permittivity))
    wpe = np.sqrt(nproton * echarge**2 / (emass_local * permittivity))
    omega_p = echarge * b0/ pmass # proton gyro-freqeuncy
    omega_e = -echarge * b0/emass_local
    vthp = np.sqrt(2 * boltzmann * tp/pmass) # proton thermal speed
    vthe = np.sqrt(2 * boltzmann * te/emass_local) # proton thermal speed
    rhop = vthp/omega_p # proton gyroradius
    w = wrel * omega_p
    kz = kpar/rhop
    kp = kperp/rhop
    vthz = vthp # assume no temperature anisotropy
    
    proton = [n, w, kz, kp, wpp, tp, tp, vthp, vthp, omega_p, 0, method]
    electron = [n, w, kz, kp, wpe, te, te, vthe, vthe, omega_e, 0, method]
    
    inp = [proton, electron]
    return inp

def transport_ratios(inp, print_result = False):
    """
    Parameters
    ----------
    inp = parameters in SI unit
    
    Return
    ------
    [eigen_e, alpha, [p_e_b0, p_b_b0, p_b_k], sigma, epsilon_b, \
    c_bb, e_l_tot, c_bn, c_par, ra, sigma_c]
    
    See Krauss-Varban et al. (1994) for definitions
    
    eigen_e: eigen electric field
    alpha: polarization vector
    p_e_b0: polarization of E along B0
    p_b_b0: polarization of B along B0
    p_b_k: polarization of B along k
    sigma: helicity
    epsilon_b: generalized ellipticity
    c_bb: magnetic compression ratio
    e_l_tot: ratio of longitudinal to total electric field power
    c_bn: compressibility for each species
    c_par: magnetic field-density correlation (parallel compressibility
    ra: Alfven ratio
    sigma_c: cross helicity

    """
    w, kz, kp = inp[0][1:4]
    k = np.sqrt(kp**2 + kz**2)
    cos_theta = kz/k
    sin_theta = kp/k
    
    # reshape the parameters to calculate dispersion tensor
    param = list(map(list, zip(*inp)))
    # dispersion tensor
    dt = f_d(param)
    # eigen electric field
    eigen_e = nullspace(dt*1e-20).reshape((3,))
    # polarization
    e_y = eigen_e[1]
    alpha = -1j * eigen_e/ e_y
    
    # alternative coordinate system
    xhead = np.array([1, 0, 0])
    yhead = np.array([0, 1, 0])
    zhead = np.array([0, 0, 1])
    kvec = np.array([kp, 0, kz])
    khead = kvec/linalg.norm(k)
    xihead = np.cross(yhead, khead)

    # different polarizatins
    alpha_x = alpha[0]
    alpha_xi = np.dot(alpha, xihead)
    p_e_b0 = -alpha_x
    p_b_b0 = -alpha_xi/cos_theta
    p_b_k = -1/alpha_xi

    # helicity
    sigma = -2*np.real(alpha_xi)/(1+linalg.norm(alpha_xi)**2)
    
    # generalized ellipticity
    epsilon_b = (1-alpha_xi)/(1+alpha_xi)

    # magnetic compression ratio &
    # ratio of longitudinal to total electric power
    c_bb=1/(1+linalg.norm(alpha_xi)**2/cos_theta**2)
    alpha_k = np.dot(alpha, khead)
    e_l_tot = linalg.norm(alpha_k)**2/ linalg.norm(alpha)**2
    
    # compressibility, parallel compressibility, Alfven ratio \
        # & cross helicity
    # number of species (at least one)
    ns = np.array(inp).shape[0]
    n_vec = kvec * cspeed/w
    n_scalar = k * cspeed/w
    k_cross_alpha = np.cross(khead, np.conjugate(alpha))

    c_bn = []
    c_par = []
    ra = []
    sigma_c = []
    
    for j in range(ns):
        omega_pj = inp[j][4]
        Omega_cj = inp[j][9]
        chi_j = f_chi(*inp[j])
        chi_alpha_j = np.dot(chi_j, alpha)
        tensor_prod_j = np.dot(khead, chi_alpha_j)
        
        c_bn_j = np.abs(w/Omega_cj)**2 * (Omega_cj/omega_pj)**4
        c_bn_j *= np.abs(tensor_prod_j)**2 / (1 + linalg.norm(alpha_xi)**2)
        c_bn += [c_bn_j]

        c_par_j = w/Omega_cj * (Omega_cj/omega_pj)**2 * tensor_prod_j/sin_theta
        c_par += [c_par_j]

        ra_j = np.abs(w/omega_pj)**2 * linalg.norm(chi_alpha_j)**2
        ra_j /= linalg.norm(n_vec)**2 * (1 + linalg.norm(alpha_xi)**2)
        ra += [ra_j]
    
        numer = np.imag(np.dot(chi_alpha_j, k_cross_alpha)/n_scalar ** 2)
        denom = (1 + linalg.norm(alpha_xi)**2) * (1 + ra_j)    
        sigma_c_j = 2 * (cspeed * k/omega_pj) * numer/ denom
        sigma_c += [sigma_c_j]

    if print_result:
        print('polarization alpha = {0}'.format(alpha))
        print('p_e_b0 = {0}'.format(p_e_b0))
        print('p_b_b0 = {0}'.format(p_b_b0))
        print('p_b_k = {0}'.format(p_b_k))
        print('helicy = {0}'.format(sigma))
        print('E_L/E_tot = {0}'.format(e_l_tot))
        print('compressibility = {0}'.format(c_bn))
        print('parallel compressibility = {0}'.format(c_par))
        print('cross helicty = {0}'.format(sigma_c))
        
    return [[alpha, p_e_b0, p_b_b0, p_b_k], sigma, epsilon_b, c_bb, e_l_tot, [c_bn, c_par, ra, sigma_c]]
