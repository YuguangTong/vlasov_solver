import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.wrapper import disp_det, oblique_wrapper
from py_vlasov.util import real_imag, list_to_complex, VlasovException
import scipy.optimize

class Test_parallel(unittest.TestCase):

    def test_mirror_1(self):
        """
        benchmark disp_det() using dispersion relation of mirror mode
        by DSHARK. Krauss-Varban (1994) Fig 8(a) motivates these tests.
        """                     
        k = 1.2
        angle = np.pi/3.
        kz = k * np.cos(angle)
        kp = k * np.sin(angle)
        betap = 4.
        t_list=[1., 0.05]
        a_list=[1., 1.]
        n_list=[1., 1.] 
        q_list=[1., -1.,]
        m_list=[1., 1/1836,]
        v_list=[0, 0]

        f = lambda wrel: real_imag(disp_det(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'pade', aol=1e-4))
        guess = -0.2j
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_almost_equal(wr, 0., decimal=7)
        npt.assert_allclose(wi, -0.1904439678, rtol = 1e-5)

        a_list=[2., 1.]
        f = lambda wrel: real_imag(disp_det(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'pade', aol=1e-4))
        guess = 0.02j
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_almost_equal(wr, 0., decimal=7)
        npt.assert_allclose(wi, 2.4989329269e-2, rtol = 1e-5)

        a_list=[3., 1.]
        f = lambda wrel: real_imag(disp_det(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'pade', aol=1e-4))
        guess = 0.1j
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_almost_equal(wr, 0., decimal=7)
        npt.assert_allclose(wi, 0.10031199989, rtol = 1e-5)        
        
if __name__ == '__main__':
    unittest.main()
