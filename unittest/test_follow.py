import numpy.testing as npt
import numpy as np
import unittest
from py_vlasov.util import real_imag, kzkp
from py_vlasov.follow_parameter import simple_follow_fn
from py_vlasov.new_follow_parameter import (
    follow_kz, follow_kp, follow_k, follow_angle, follow_beta,
    follow_temperature, follow_anisotropy, follow_drift)


class Test_follow(unittest.TestCase):
    
    def test_follow_kz_alfven_1(self):
        """
        test follow_kz() on an alven mode against WHAMP.
        """
        kz = 0.01
        kp = 0.1       
        seed_freq = 0.01
        
        betap = 1.
        t_list=[1.,1]
        a_list=[1., 1.]
        n_list=[1.,1.] 
        q_list=[1.,-1.]
        m_list=[1., 1./1836.]
        v_list=[0.,0.]
        n = 10
        method = 'pade'
        aol = 1/5000.
        
        param = [kz, kp, betap, t_list, a_list, n_list, q_list,
                 m_list, v_list, n, method, aol]
        guess_fn = simple_follow_fn
        
        target_kz = 0.1
        log_inc = 0.2
        freq = follow_kz(seed_freq, target_kz, param, guess_fn,
                         log_incrmt=log_inc, incrmt_method = 'log')
        w_r, w_i = real_imag(freq)
        npt.assert_allclose(w_r, 9.9115e-2, rtol = 1e-3)
        npt.assert_allclose(w_i, - 2.2500e-4, rtol = 1e-3)

        target_kz = 0.2
        freq = follow_kz(seed_freq, target_kz, param,
                         guess_fn, )
        w_r, w_i = real_imag(freq)
        npt.assert_allclose(w_r, 1.8294e-1, rtol = 1e-3)
        npt.assert_allclose(w_i, - 4.6795e-3, rtol = 1e-3)        


    # def test_follow_kp_alfven_1(self):
    #     """
    #     test follow_kp() on an alven mode.
    #     """
    #     kz = 0.01
    #     kp = 0.1       
    #     seed_freq = 0.01
        
    #     betap = 1.
    #     t_list=[1.,1]
    #     a_list=[1., 1.]
    #     n_list=[1.,1.] 
    #     q_list=[1.,-1.]
    #     m_list=[1., 1./1836.]
    #     v_list=[0.,0.]
    #     n = 10
    #     method = 'pade'

    #     param = [kz, kp, betap, t_list, a_list, n_list, q_list,
    #              m_list, v_list, n, method]

    #     guess_fn = simple_follow_fn
        
    #     target_kp = 0.5
    #     increment = 0.1
    #     freq = follow_kz(seed_freq, target_kp, param,
    #                      increment,guess_fn, show_plot=False)
    #     w_r, w_i = real_imag(freq)
    #     npt.assert_allclose(w_r, 1.0333e-2, rtol = 1e-3)
    #     npt.assert_allclose(w_i, -1.0742-4, rtol = 1e-3)

    #     target_kp = 1.0
    #     increment = 0.1
    #     freq = follow_kz(seed_freq, target_kp, param,
    #                      increment,guess_fn, show_plot=False)
    #     w_r, w_i = real_imag(freq)
    #     npt.assert_allclose(w_r, 1.1366e-2, rtol = 1e-3)
    #     npt.assert_allclose(w_i, -3.2756e-4, rtol = 1e-3)


    # def test_follow_anisotropy_alfven_1(self):
    #     """
    #     test follow_anisotropy() on an alven mode.
    #     """
    #     kz = 0.1
    #     kp = 0.1       
    #     seed_freq = 0.1
        
    #     betap = 1.
    #     t_list=[1.,1]
    #     a_list=[1., 1.]
    #     n_list=[1.,1.] 
    #     q_list=[1.,-1.]
    #     m_list=[1., 1./1836.]
    #     v_list=[0.,0.]
    #     n = 10
    #     method = 'pade'

    #     param = [kz, kp, betap, t_list, a_list, n_list, q_list,
    #              m_list, v_list, n, method]

    #     guess_fn = simple_follow_fn
        
    #     target_anisotropy = [2., 2.]
    #     increment = 0.1
    #     freq = follow_anisotropy(seed_freq, target_anisotropy, param,
    #                      increment,guess_fn, show_plot=False)
    #     w_r, w_i = real_imag(freq)
    #     npt.assert_allclose(w_r, 1.4079e-1, rtol = 1e-3)
    #     npt.assert_allclose(w_i, -5.3245e-4, rtol = 1e-3)

    #     target_anisotropy = [.2, .2]
    #     increment = 0.1
    #     freq = follow_anisotropy(seed_freq, target_anisotropy, param,
    #                      increment, guess_fn, show_plot=False)
    #     w_r, w_i = real_imag(freq)
    #     npt.assert_allclose(w_r, 1.2638e-1, rtol = 1e-3)
    #     npt.assert_allclose(w_i, -2.1521e-4, rtol = 1e-3)        
        

        
if __name__ == '__main__':
    unittest.main()
