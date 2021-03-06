import numpy as np
from .util import VlasovException, real_imag, list_to_complex
from .wrapper import disp_det
from numbers import Number
import scipy.optimize
import matplotlib.pyplot as plt
#---------------------------------------------#
# consider expansion to include changes in mass
#---------------------------------------------#
def follow_eigen_freq(seed_freq, target_param, target_value,
                      param, num, show_plot=False):
    """
    Follow the frequency of an eigenmode of a plasma along a
    parameter ('target_param') till the parameter reach a
    target value ('target_val'). Return the frequency at the new
    set of plasma parameters.
    
    parameters
    ----------
    seed_freq: eigen mode frequency of the initial plasma system
    target_param: the parameter to follow along. Should be a string.
    target_value: the value of the target parameter
    
    param: a set of dimensionless plasma of which the seed_freq is an 
           eigen frequency. It contains the following list
           [kpar, kperp, betap, t_list, a_list, n_list, q_list, m_list, v_list,
            n, method, aol]
    num: number of steps to search
    guess_fn: a function to guess the next frequency
    show_plot: whether make a plot for the frequencies stepped through
    """
    if target_param == 'kz':
        return follow_kz(seed_freq, target_value,param, num)
    elif target_parm == 'kp':
        return follow_kp(seed_freq, target_param, target_value, param, num)
    elif target_parm == 'k':
        return follow_k(seed_freq, target_value, param, num)
    elif target_parm == 'angle':
        return follow_angle(seed_freq, target_value, param, num)
    elif target_parm == 'beta':
        return follow_beta(seed_freq, target_value, param, num)
    elif target_parm == 'temperature':
        return follow_temperature(seed_freq, target_value, param, num)
    elif target_parm == 'anisotropy':
        return follow_anisotropy(seed_freq, target_value, param, num)
    elif target_parm == 'drift':
        return follow_drift(seed_freq, target_value, param, num)
    else:
        raise VlasovException('{0} is not a valid parameter to vary'.
                              format(target_param))

def generate_1d_steps(start, end, log_incrmt, lin_incrmt, incrmt_method):
    """
    Generate a list of steps from START to END.
    Assume that START and END are scalar.

    parameters
    ----------
    start: a list of values to start
    end: a list of values to end
    log_incrmt: incremental steps if using log scale.
                start, start*(1+log_incrmt), start*(1+log_incrmt)**2 ...
    lin_incrmt: incremental steps if using linear scale.
                start, start+lin_incrmt, start+2*lin_incrmt ...
    incrmt_method:
                 'linear' --> linear scale
                 'log' --> log scale
    return
    ------
    a list of steps between start and end
    """
    if np.abs(start - end) < 1e-8:
        return np.array([start, end])
    if incrmt_method == 'log':
        num_step = 1 + np.abs(np.log(start/end))/np.log(1+log_incrmt)
        num_step = round(num_step)
        return np.logspace(np.log10(start), np.log10(end), num=num_step)
    elif incrmt_method == 'linear':
        num_step = 1 + round(np.abs(start - end)/lin_incrmt)
        num_step = round(num_step)
        return np.linspace(start, end, num=num_step)
    
def generate_2d_steps(start, end, log_incrmt, lin_incrmt, incrmt_method):
    """
    Generate a list of steps from START to END.
    Assume that START and END are scalar.

    parameters
    ----------
    start: a list of values to start
    end: a list of values to end
    log_incrmt: incremental steps if using log scale.
                start, start*(1+log_incrmt), start*(1+log_incrmt)**2 ...
    lin_incrmt: incremental steps if using linear scale.
                start, start+lin_incrmt, start+2*lin_incrmt ...
    incrmt_method:
                 'linear' --> linear scale
                 'log' --> log scale
    return
    ------
    a list of steps between start and end
    """
    start = np.array(start)
    end = np.array(end)
    if incrmt_method == 'log':
        assert not False in (start > 0)
        assert not False in (end > 0)
        ratio = np.abs(np.log(start/end))
        num_step = 1 + round(np.amax(ratio) / np.log(1 + log_incrmt))
        res = np.array([np.logspace(
            np.log10(start[i]), np.log10(end[i]), num=num_step)
                        for i in range(len(start))])
        return res.T
    elif incrmt_method == 'linear':
        num_step = 1 + np.amax(np.abs(start - end)/lin_incrmt)
        num_step = round(num_step)
        res = np.array([np.linspace(start[i], end[i], num=num_step)
                        for i in range(len(start))])       
        return res.T

def generate_steps(start, end, log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'linear'):
    """
    Generate a list of steps from START to END.
    START and END are either scalar or 1D list and have the same dimension.

    """
    
    if isinstance(start, Number) and isinstance(end, Number):
        return generate_1d_steps(start, end, log_incrmt,
                                 lin_incrmt, incrmt_method)
    else:
        assert hasattr(start, '__iter__') and hasattr(end, '__iter__')
        return generate_2d_steps(start, end, log_incrmt,
                                 lin_incrmt, incrmt_method)
    
def follow_kz(seed_freq, target_value, param, show_plot=False,
              log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow a mode with frequency SEED_FREQ in a plasma specified by
    PARAM along the kz parameter.
    """
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_kz = kz
    # a list of kz to step through
    kz_list = generate_steps(kz, target_value, log_incrmt=log_incrmt,
                             lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)

    freq_lst = []
    guess = seed_freq
    for kz in kz_list:
        f = lambda wrel: real_imag(disp_det(
            list_to_complex(wrel), kz, kp, beta, t_list, a_list,
            n_list, q_list, m_list, v_list, n=n, method=method, aol=aol))
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        guess = list_to_complex(freq)
        freq_lst += [guess]
    if show_plot:
        plt.plot(kz_list, np.real(freq_lst), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$kz\rho_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $kz\rho_p$ from {0} to {1}'.format(seed_kz, target_value))
        plt.show()        
    return guess

def follow_kp(seed_freq, target_value, param, show_plot=False,
              log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    to implement
    """
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_kp = kp
    # a list of kp to step through
    kp_list = generate_steps(kp, target_value, log_incrmt=log_incrmt,
                             lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)

    freq_lst = []
    guess = seed_freq
    for kp in kp_list:
        f = lambda wrel: real_imag(disp_det(
            list_to_complex(wrel), kz, kp, beta, t_list, a_list,
            n_list, q_list, m_list, v_list, n=n, method=method, aol=aol))
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        guess = list_to_complex(freq)
        freq_lst += [guess]
    if show_plot:
        plt.plot(kp_list, np.real(freq_lst), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$kp\rho_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $kp\rho_p$ from {0} to {1}'.format(seed_kp, target_value))
        plt.show()        
    return guess

def follow_angle(seed_freq, target_value, param, increment,guess_fn, show_plot=False):
    """
    to implement
    """
    
    return 0

def follow_k(seed_freq, target_value, param, increment,guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_beta(seed_freq, target_value, param, increment,guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_temperature(seed_freq, target_value, param, increment,guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_anisotropy(seed_freq, target_value, param, show_plot=False,
              log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow a mode with frequency SEED_FREQ in a plasma specified by
    PARAM along the anisotropy parameter.
    """
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_a_list = a_list
    # a list of kp to step through
    a_list_steps = generate_steps(a_list, target_value, log_incrmt=log_incrmt,
                             lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)

    freq_lst = []
    guess = seed_freq
    for a_list in a_list_steps:
        f = lambda wrel: real_imag(disp_det(
            list_to_complex(wrel), kz, kp, beta, t_list, a_list,
            n_list, q_list, m_list, v_list, n=n, method=method, aol=aol))
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        guess = list_to_complex(freq)
        freq_lst += [guess]
    if show_plot:
        plt.plot(a_list_steps[:,0], np.real(freq_lst), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$T_{p\perp}/T_{p\parallel}$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $T_{s\perp}/T_{s\parallel}$')
        plt.show()        
    return guess
   

def follow_drift(seed_freq, target_value, param, increment, guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0



