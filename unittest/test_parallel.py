import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.wrapper import parallel_wrapper
from py_vlasov.util import real_imag, list_to_complex
import scipy.optimize

class Test_parallel(unittest.TestCase):
    def test_par_wrapper_firehose_1(self):
        """
        benchmark parallel_wrapper() using dispersion relation of 
        fireose instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
        k=0.5 * np.sqrt(2.)
        kz = k
        kp = 0        
        betap = 3.16
        t_list=[1., 1., 4.]
        a_list=[0.5, 0., 0.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]

        f = lambda wrel: real_imag(parallel_wrapper(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'numpy', aol=1/5000))
        guess = 0.4
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(freq[0], 0.396882, rtol = 3)
        npt.assert_allclose(freq[1], 0.0242365, rtol = 3)

    def test_par_wrapper_firehose_2(self):
        """
        benchmark parallel_wrapper() using dispersion relation of 
        fireose instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
        k=0.4 * np.sqrt(2.)
        kz = k
        kp = 0        
        betap = 3.16
        t_list=[1., 1., 4.]
        a_list=[0.5, 0., 0.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]

        f = lambda wrel: real_imag(parallel_wrapper(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'numpy', aol=1/5000))
        guess = 0.2
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(freq[0], 0.248512, rtol = 3)
        npt.assert_allclose(freq[1], 0.0116537, rtol = 3)        
        
    def test_par_wrapper_firehose_3(self):
        """
        benchmark parallel_wrapper() using dispersion relation of 
        fireose instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
        k=0.7 * np.sqrt(2.)
        kz = k
        kp = 0        
        betap = 3.16
        t_list=[1., 1., 4.]
        a_list=[0.5, 0., 0.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]

        f = lambda wrel: real_imag(parallel_wrapper(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'numpy', aol=1/5000))
        guess = 0.5
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(freq[0], 0.640861, rtol = 3)
        npt.assert_allclose(freq[1], 0.00937915, rtol = 3)         

    def test_par_wrapper_firehose_4(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        fireose instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.2 * np.sqrt(2.)
         kz = k
         kp = 0         
         betap = 3.16
         t_list=[1., 1., 4.]
         a_list=[0.3, 0., 0.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000))
         guess = 0.5
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.142828, rtol = 3)
         npt.assert_allclose(freq[1], -0.00109096, rtol = 3)        

    def test_par_wrapper_firehose_5(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        fireose instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.5 * np.sqrt(2.)
         kz = k
         kp = 0
         betap = 3.16
         t_list=[1., 1., 4.]
         a_list=[0.3, 0., 0.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000))
         guess = 0.5
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.46114682, rtol = 3)
         npt.assert_allclose(freq[1], -0.01967979, rtol = 3)
        
if __name__ == '__main__':
    unittest.main()
