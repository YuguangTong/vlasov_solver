import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.wrapper import parallel_wrapper
from py_vlasov.util import real_imag, list_to_complex, VlasovException
import scipy.optimize

class Test_parallel(unittest.TestCase):

    def test_not_parallel_error(self):
        """
        Check if parallel_wrapper throws VlasovException if kp != 0

        """
        kz = 0.5
        kp = 0.1      
        betap = 3.16
        t_list=[1., 1., 4.]
        a_list=[0.5, 1., 1.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]
        wrel = [0.5, 0.1]
        self.assertRaises(VlasovException, parallel_wrapper,
                         list_to_complex(wrel), kz, kp, betap, t_list,
                         a_list, n_list, q_list, m_list, v_list,
                         method = 'numpy', aol=1/5000)
    
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
        a_list=[0.5, 1., 1.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]

        f = lambda wrel: real_imag(parallel_wrapper(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'numpy', aol=1/5000))
        guess = 0.4
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(freq[0],0.370972, rtol = 1e-3)
        npt.assert_allclose(freq[1], 0.0241541, rtol = 1e-3)

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
        a_list=[0.5, 1., 1.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]

        f = lambda wrel: real_imag(parallel_wrapper(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'numpy', aol=1/5000))
        guess = 0.2
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(freq[0], 0.248512, rtol = 1e-3)
        npt.assert_allclose(freq[1], 0.0116537, rtol = 1e-3)        
        
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
        a_list=[0.5, 1., 1.]
        n_list=[1., 1.10,0.05] 
        q_list=[1., -1., 2.]
        m_list=[1., 1/1836, 4.]
        v_list=[0, 0, 0]

        f = lambda wrel: real_imag(parallel_wrapper(
            list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
            q_list, m_list, v_list, method = 'numpy', aol=1/5000))
        guess = 0.5
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        npt.assert_allclose(freq[0], 0.640861, rtol = 1e-3)
        npt.assert_allclose(freq[1], 0.00937915, rtol = 1e-3)         

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
         a_list=[0.7, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000))
         guess = 0.5
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.142828, rtol = 1e-3)
         npt.assert_allclose(freq[1], -0.00109096, rtol = 1e-3)        

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
         a_list=[0.7, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000))
         guess = 0.5
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.46114682, rtol = 1e-3)
         npt.assert_allclose(freq[1], -0.01967979, rtol = 1e-3)

    def test_par_wrapper_cyclotron_1(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        cyclotron instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.2 * np.sqrt(2.)
         kz = k
         kp = 0
         betap = 1.0
         t_list=[1., 1., 4.]
         a_list=[1.5, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000, mode='L'))
         guess = 0.2
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.230978, rtol = 1e-3)
         npt.assert_allclose(freq[1], -0.0314169, rtol = 1e-3)         

    def test_par_wrapper_cyclotron_2(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        cyclotron instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.35 * np.sqrt(2.)
         kz = k
         kp = 0
         betap = 1.0
         t_list=[1., 1., 4.]
         a_list=[1.5, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000, mode='L'))
         guess = 0.3
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.356764, rtol = 1e-3)
         npt.assert_allclose(freq[1], -0.0398052, rtol = 1e-3)

    def test_par_wrapper_cyclotron_3(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        cyclotron instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.35 * np.sqrt(2.)
         kz = k
         kp = 0
         betap = 1.0
         t_list=[1., 1., 4.]
         a_list=[2.25, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000, mode='L'))
         guess = 0.4
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.4721, rtol = 1e-3)
         npt.assert_allclose(freq[1], 0.0447763, rtol = 1e-3)

    def test_par_wrapper_cyclotron_4(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        cyclotron instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.65 * np.sqrt(2.)
         kz = k
         kp = 0
         betap = 1.0
         t_list=[1., 1., 4.]
         a_list=[2.5, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000, mode='L'))
         guess = 0.4
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.58734 , rtol = 1e-3)
         npt.assert_allclose(freq[1], -0.00603197, rtol = 1e-3)         

    def test_par_wrapper_cyclotron_5(self):
         """
        benchmark parallel_wrapper() using dispersion relation of 
        cyclotron instabilitiy. The benchmark data are taken from a figure in
        BA Maruca's PhD thesis (p.82). Maruca used SP Gary's code to generate
        the plot.
        """
         k=0.65 * np.sqrt(2.)
         kz = k
         kp = 0
         betap = 1.0
         t_list=[1., 1., 4.]
         a_list=[1.5, 1., 1.]
         n_list=[1., 1.10,0.05] 
         q_list=[1., -1., 2.]
         m_list=[1., 1/1836, 4.]
         v_list=[0, 0, 0]

         f = lambda wrel: real_imag(parallel_wrapper(
             list_to_complex(wrel), kz, kp, betap, t_list, a_list, n_list,
             q_list, m_list, v_list, method = 'numpy', aol=1/5000, mode='L'))
         guess = 0.4
         freq = scipy.optimize.fsolve(f, real_imag(guess))
         npt.assert_allclose(freq[0], 0.435854, rtol = 1e-3)
         npt.assert_allclose(freq[1], -0.232166, rtol = 1e-3)         
if __name__ == '__main__':
    unittest.main()
