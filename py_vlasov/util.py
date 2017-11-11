import numpy as np
import mpmath as mp
import scipy.special
from numpy.linalg import svd

class VlasovException(Exception):
    """
    Exception for the vlasov solver package.
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

emass = 9.10938291e-31
pmass = 1.67262178e-27
echarge = 1.60217657e-19
permittivity = 8.854187817e-12
permeability = 4 * np.pi * 1e-7
cspeed = 299792458
boltzmann = 1.3806488e-23

c_arr = np.array([
2.237687789201900 - 1.625940856173727j,
-2.237687789201900 - 1.625940856173727j,
1.465234126106004 - 1.789620129162444j,
-1.465234126106004 - 1.789620129162444j,
0.8392539817232638 - 1.891995045765206j,
-0.8392539817232638 - 1.891995045765206j,
0.2739362226285564 - 1.941786875844713j,
-0.2739362226285564 -1.941786875844713j])

b_arr = np.array([-0.01734012457471826 - 0.04630639291680322j,
-0.01734012457471826 + 0.04630639291680322j,
-0.7399169923225014 + 0.8395179978099844j,
-0.7399169923225014 - 0.8395179978099844j,
5.840628642184073 + 0.9536009057643667j,
5.840628642184073 - 0.9536009057643667j,
-5.583371525286853 - 11.20854319126599j,
-5.583371525286853 + 11.20854319126599j])

def pade(z):
    """
    Pade approximations to plasma dispersion function.
    
    Keyword arguments:
    z: dimensionless argument of the plasma dispersion function.
    
    Return the value of Zp(z) using Pade approximations.
    """
    return np.sum(b_arr/(z-c_arr))

def zp(z):
    """
    Plasma dispersion function
    Utilize the faddeeva function, wofz, in scipy.special module.
    Note 1.7724538509055159 = sqrt(pi)
    
    Keyword arguments:
    z: dimensionless argument of the plasma dispersion function.
    
    Return the value of Zp(z)
    """
    return 1.7724538509055159j * scipy.special.wofz(z)
    
def zpd(x):
    """
    Derivative of the plasma dispersion function
    
    """
    return -2 * (1 + x * zp(x))

def real_imag(val):
    """
    Return the list [real(val), imag(val)]
    
    """
    return [np.real(val), np.imag(val)]

def list_to_complex(x):
    """
    Convert a list of two numbers to a complex number:
    x -> x[0] + x[1] * j
    """
    return x[0] + x[1] * 1j

def kzkp(k, theta):
    """
    parameter
    ---------
    k: wave number
    theta: angle (degree) between k and B (background B field). 
    Return
    ------
    (kz, kp)
    kz: parallel wavenumber
    kp: perpendicular wavenumber
    
    """
    theta_rad = theta * np.pi/180.
    kz = k * np.cos(theta_rad)
    kp = k * np.sin(theta_rad)
    return kz, kp

def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    # tol = max(atol, rtol * s[0])
    # nnz = (s >= tol).sum()
    # ns = vh[nnz:].conj().T
    ns = vh[2].conj()
    return ns

def do_cprofile(func):
    """
    wrapper of func to do profile 
    """

    import cProfile, pstats
    
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            ps = pstats.Stats(profile)
            ps.strip_dirs().sort_stats('time').print_stats(20)
    return profiled_func
