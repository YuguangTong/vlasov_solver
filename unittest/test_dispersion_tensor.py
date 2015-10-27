import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.dispersion_tensor import f_abn, f_yn, f_chi


class Test_dispersion_tensor(unittest.TestCase):

    def test_an_bn_1(self):
        """
        Test coefficient An and Bn, in Stix (1992).

        """
        n = 0
        w = 2.1
        kz = 3.e-5
        tp = 2.e3 
        tz = 100.
        vthz = 6.1e4
        omega = 0.34
        vz = 3.4e3
        
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
        an, bn = an/w, bn/w
        npt.assert_allclose(an, -1.9218850625427564 + 5.595463531399411j,
                            rtol=1e-7)
        npt.assert_allclose(bn, -102817.6686637071 + 391682.4471979587j,
                            rtol=1e-7)

        w = 0.03
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
        an, bn = an/w, bn/w
        npt.assert_allclose(an, 631.2715006824648 - 46.41868239430783j,
                            rtol=1e-7)
        npt.assert_allclose(bn, 551271.5006824646 - 46418.68239430781j,
                            rtol=1e-7)
        
        n, w = 5, 0.03
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
        an, bn = an/w, bn/w
        npt.assert_allclose(bn, 1.6137363813074678e6 + 2.3743306625716157e7j,
                            rtol=1e-7)
        npt.assert_allclose(an, -30.426402059415523 - 426.52646633023045j,
                            rtol=1e-7)

        n, w, kz, vthz = 20, 3.7, 0.11, 0.35
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
        an, bn = an/w, bn/w
        npt.assert_allclose(an, -0.26539669257210097 -0.0j, \
                            rtol=1e-7)
        npt.assert_allclose(bn, -902.3486484918872 + 0.0j,\
                            rtol=1e-7)

    def test_yn_1(self):
        """
        test Yn function in Stix (1992).
        """
        
        n = 0
        w = 0.01
        kz = 1.0e-7
        kp = 1.0e-7
        tp = 1.0e-18
        tz = 1.0e-18
        vthz = 1e5
        vthp = 1e5
        omega = 0.1
        vz = 100.

        yn = f_yn(n, w, kz, kp, tp, tz, vthz, vthp, omega, vz, method = 'numpy')/w
        expected_yn = np.array(
            [[0,0,0],
             [0, -1.072552723 + 0.6510730508j, -6.510730508 - 0.7604398751j],
             [0, 6.510730508 + 0.7604398751j, -15.24691467 + 130.5409615j]])
        npt.assert_allclose(yn, expected_yn, rtol = 1e-7)

    def test_chi_1(self):
        """
        test chi tensor as defined in Stix (1992).
        """
        NN = 0
        w = 0.01
        kz = 1.0e-7
        kp = 1.0e-7
        wp = 2.0e3
        tz = 1.0e-18
        tp = 1.0e-18
        vthz = 1e5
        vthp = 1e5
        omega = 0.1
        vz = 100.

        chi = f_chi(NN, w, kz, kp, wp, tz, tp, vthz, vthp,
                    omega, vz, method = 'numpy')/w**2
        expected_chi = np.array([
            [0,0,0],
            [0, -4.268813377e8+2.591303241e8j, -2.591303241e9-3.026588661e8j],
            [0, 2.591303241e9+3.026588661e8j, -5.988348146e9+5.195595431e10j]])
        npt.assert_allclose(chi, expected_chi, rtol = 1e-7)

    def test_chi_2(self):
        """
        test chi tensor as defined in Stix (1992).
        """
        NN = 4
        w = 0.01
        kz = 1.0e-7
        kp = 3.0e-7
        wp = 2.0e3
        tz = 2.0e-18
        tp = 1.0e-18
        vthz = 1e5
        vthp = 1e5
        omega = 0.1
        vz = 100.

        chi = f_chi(NN, w, kz, kp, wp, tz, tp, vthz, vthp,
                    omega, vz, method = 'numpy')/w**2
        expected_chi = np.array([
            [4.972725326e8, 3.812283500e9j, -3.086420946e8],
            [-3.812283500e9j, -3.037753676e9 + 1.098455531e9j, -3.661518437e9-3.465700712e8j],
            [-3.086420946e8, 3.661518437e9+3.465700712e8j, -1.976136342e9 + 2.497184730e10j]])
        npt.assert_allclose(chi, expected_chi, rtol = 1e-7)

if __name__ == '__main__':
    unittest.main()
