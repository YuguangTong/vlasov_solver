import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.util import zp

class Test_util(unittest.TestCase):
    def test_zp(self):
        """
        test plasma dispersion function.
        """
        npt.assert_allclose(zp(5.), \
                            -0.2042681488485533 + 2.4615739584615114e-11j,\
                            rtol=1e-05)
        
        npt.assert_allclose(zp(-5.), \
                            0.2042681488485533 + 2.4615739584615114e-11j, \
                            rtol=1e-05)
        npt.assert_allclose(zp(0.01 + 0.01j), \
                            -0.019648175685503314 + 1.7524564823364268j, \
                            rtol=1e-5)
        npt.assert_allclose(zp(0.01 + 1.3j), \
                            -0.003518385129695739 + 0.633887475976703j, \
                            rtol=1e-5)
        npt.assert_allclose(zp(0.01 + 1.3j), \
                            -0.003518385129695739 + 0.633887475976703j, \
                            rtol=1e-5)
        npt.assert_allclose(zp(0.013 + 4.9j), \
                            -0.0005102157592773438 + 0.2000732421875j, \
                            rtol=1e-5)
        npt.assert_allclose(zp(1.0 + 0.17j), \
                            -0.8791163449240138 + 0.6622129836119347j, \
                            rtol=1e-5)
        
if __name__ == '__main__':
    unittest.main()
