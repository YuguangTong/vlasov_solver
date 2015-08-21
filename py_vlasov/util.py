import numpy as np
import mpmath as mp
import scipy.special

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
    Utilize the Dawnson function, dawsn, in scipy.special module.

    Keyword arguments:
    z: dimensionless argument of the plasma dispersion function.
    
    Return the value of Zp(z)
    """
    return -2. * scipy.special.dawsn(z) + 1.j * np.sqrt(np.pi) * np.exp(- z **2)
    
def zpd(x):
    """
    Derivative of the plasma dispersion function
    
    """
    return -2 * (1 + x * zp(x))

def zp_mp(z):
    """
    Plasma dispersion function to user-defined precison.                              
    Utilize the complementary error function in mpmath library.                       

    Keyword arguments:
    z: dimensionless argument of the plasma dispersion function.
    
    Return the value of Zp(z) to arbitrary precision.
    'import mpmath' and 'mpmath.mp.dps=n' set the precison to n digit.
    """
    return -mp.sqrt(mp.pi) * mp.exp(-z**2) * mp.erfi(z) + mp.mpc(0, 1) * mp.sqrt(mp.pi) * mp.exp(-z**2)

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
