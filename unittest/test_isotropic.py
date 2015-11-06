import numpy.testing as npt
import numpy as np
import unittest
import scipy.optimize
from py_vlasov.util import zp, kzkp, list_to_complex, real_imag
from py_vlasov.wrapper import oblique_wrapper

class Test_isotropic(unittest.TestCase):
    """
    we test the solution of plasma wave frequency against DSHARK
    in isotropic plasma.

    """
    def test_alfven(self):
        """
        test finding solution of alfven waves given a good guess.
        """

        betap = 1.
        t_list=[1.,1]
        a_list=[1., 1.]
        n_list=[1.,1.] 
        q_list=[1.,-1.]
        m_list=[1., 1./1836.]
        v_list=[0.,0.]
        
        theta = 80.

        #---------------------------#
        # intermediate k, unity beta
        #---------------------------#
        k = 0.8
        kz, kp = kzkp(k, theta)
        alfven = kz /np.sqrt(betap) # guess

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = alfven
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr, 0.1487200, rtol = 1e-5)
        npt.assert_allclose(wi, -0.3055460e-02, rtol = 1e-5)
        
        #--------------------#
        # small k, unity beta
        #--------------------#
        k = 0.1
        kz, kp = kzkp(k, theta)
        alfven = kz /np.sqrt(betap) # guess

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = alfven
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr, 1.7379506101250795E-002, rtol = 1e-5)
        npt.assert_allclose(wi, -7.5337102688622651E-006, rtol = 1e-5)

        #--------------------#
        # small k, small beta
        #--------------------#
        k = 0.1
        betap = 0.01
        kz, kp = kzkp(k, theta)
        alfven = kz /np.sqrt(betap) # guess

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = alfven
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr, 0.17167100801749985, rtol = 1e-5)
        npt.assert_allclose(wi, -1.6919089742913979E-004, rtol = 1e-5)

        #--------------------#
        # small k, large beta
        #--------------------#
        k = 0.1
        betap = 100.
        kz, kp = kzkp(k, theta)
        alfven = kz /np.sqrt(betap) # guess

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = alfven
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr, 1.7346964017576358E-003, rtol = 1e-5)
        npt.assert_allclose(wi, -2.4496530345107542E-005, rtol = 1e-5)


    def test_fast(self):
        """
        test finding solution of fast waves given a good guess.
        """

        betap = 1.
        t_list=[1.,1]
        a_list=[1., 1.]
        n_list=[1.,1.] 
        q_list=[1.,-1.]
        m_list=[1., 1./1836.]
        v_list=[0.,0.]
        
        theta = 30.

        #--------------------#
        # small k, unity beta
        #--------------------#
        k = 0.1
        kz, kp = kzkp(k, theta)
        fast = 0.128

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = fast
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr,0.12766204160059919, rtol = 1e-5)
        npt.assert_allclose(wi, -1.4220312021487975E-002, rtol = 1e-5)


        #-------------------------#
        # very small k, unity beta
        #-------------------------#
        k = 0.01
        kz, kp = kzkp(k, theta)
        fast = 0.0128

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = fast
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr,1.2738898812001565E-002, rtol = 1e-5)
        npt.assert_allclose(wi,-1.4788747870134837E-003, rtol = 1e-5)


        #--------------------#
        # large k, unity beta
        #--------------------#
        k = 5.
        kz, kp = kzkp(k, theta)
        fast = 22

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = fast
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr,22.245783373592722, rtol = 1e-5)
        npt.assert_allclose(wi,-0.53550638712302845, rtol = 1e-5)        
                

    def test_slow(self):
        """
        test finding solution of slow waves given a good guess.
        """

        betap = 1.
        t_list=[1.,1]
        a_list=[1., 1.]
        n_list=[1.,1.] 
        q_list=[1.,-1.]
        m_list=[1., 1./1836.]
        v_list=[0.,0.]
        
        theta = 30.

        #--------------------#
        # small k, unity beta
        #--------------------#
        k = 0.1
        kz, kp = kzkp(k, theta)
        slow = 0.109 - 0.05j

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = slow
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr,0.10922146808591095, rtol = 1e-5)
        npt.assert_allclose(wi, -4.6967417598900864E-002, rtol = 1e-5)

        #--------------------#
        # unity k, unity beta
        #--------------------#
        k = 1.0
        theta = 60.
        kz, kp = kzkp(k, theta)
        slow = 0.6 - 0.4j

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 4, \
                                      aol=1e-4))
        guess = slow
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr,0.59307352008003522, rtol = 1e-5)
        npt.assert_allclose(wi, -0.39225459969080823, rtol = 1e-5)

        #--------------------#
        # large k, unity beta
        #--------------------#
        k = 3.0
        theta = 60.
        kz, kp = kzkp(k, theta)
        slow = 3. - 1.5j

        f = lambda wrel:\
            real_imag(oblique_wrapper(list_to_complex(wrel), kz, kp, betap,
                                      t_list, a_list, n_list, q_list, \
                                      m_list, v_list, method = 'pade', n = 20, \
                                      aol=1e-4))
        guess = slow
        wr, wi = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(wr,3.0226417896957392, rtol = 1e-5)
        npt.assert_allclose(wi, -1.3824717587864439, rtol = 1e-5)

         
if __name__ == '__main__':
    unittest.main()
