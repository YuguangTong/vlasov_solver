import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.dispersion_tensor import f_abn, f_yn


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
        npt.assert_allclose(an, -1.9218850625427564 + 5.595463531399411j,
                            rtol=1e-7)
        npt.assert_allclose(bn, -102817.6686637071 + 391682.4471979587j,
                            rtol=1e-7)

        w = 0.03
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
        npt.assert_allclose(an, 631.2715006824648 - 46.41868239430783j,
                            rtol=1e-7)
        npt.assert_allclose(bn, 551271.5006824646 - 46418.68239430781j,
                            rtol=1e-7)
        
        n, w = 5, 0.03
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
        npt.assert_allclose(bn, 1.6137363813074678e6 + 2.3743306625716157e7j,
                            rtol=1e-7)
        npt.assert_allclose(an, -30.426402059415523 - 426.52646633023045j,
                            rtol=1e-7)

        n, w, kz, vthz = 20, 3.7, 0.11, 0.35
        an, bn = f_abn(n, w, kz, tp, tz, vthz, omega, vz, method='numpy')
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

        yn = f_yn(n, w, kz, kp, tp, tz, vthz, vthp, omega, vz, method = 'numpy')
        expected_yn = np.array(
            [[0,0,0],
             [0, -1.072552723 + 0.6510730508j, -6.510730508 - 0.7604398751j],
             [0, 6.510730508 + 0.7604398751j, -15.24691467 + 130.5409615j]])
        npt.assert_allclose(yn, expected_yn, rtol = 1e-7)
        
        
if __name__ == '__main__':
    unittest.main()
