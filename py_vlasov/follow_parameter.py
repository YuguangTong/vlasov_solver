import numpy as np
from .util import VlasovException, real_imag, list_to_complex
from .wrapper import disp_det
from numbers import Number
import scipy.optimize
import matplotlib.pyplot as plt
#---------------------------------------------#
# consider expansion to include changes in mass
#---------------------------------------------#

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
    raise NotImplementedError('Does not support increment method {0}.'.format(incrmt_method))
    
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
    raise NotImplementedError('Does not support increment method {0}.'.format(incrmt_method))

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

def solve_disp(guess, kz, kp, beta, t_list, a_list,
               n_list, q_list, m_list, v_list,
               n, method, aol, pol):
    """
    Solve dispersion relation given plasma parameters 
    and an initial guess.

    We apply an artificial Doppler shift velocity vd
    in the case when the eigenfrequency is so small that
    numberical solver cannot correctly determine solution.

    """
    vd, shift_freq = 0, 0
    if np.abs(guess) < 1e-2:
        shift_freq = 0.1
        guess += shift_freq
        vd = shift_freq * np.sqrt(beta)/kz
    temp_v_list = np.array(v_list) + vd
    f = lambda wrel: real_imag(disp_det(
        list_to_complex(wrel), kz, kp, beta, t_list, a_list,
        n_list, q_list, m_list, temp_v_list, n=n, method=method,
        aol=aol, pol=pol))
    freq = scipy.optimize.fsolve(f, real_imag(guess))
    return list_to_complex(freq) - shift_freq

    # vd, shift_freq = 0, 0
    # if np.abs(guess) < 1e-3:
    #     vd = .1
    #     shift_freq = vd * kz/np.sqrt(beta)
    # temp_v_list = np.array(v_list) + vd
    # f = lambda wrel: real_imag(disp_det(
    #     list_to_complex(wrel), kz, kp, beta, t_list, a_list,
    #     n_list, q_list, m_list, temp_v_list, n=n, method=method,
    #     aol=aol, pol=pol))
    # freq = scipy.optimize.fsolve(f, real_imag(guess))
    # return list_to_complex(freq) - shift_freq
    
def follow_kz(seed_freq, target_value, param, pol='r',  show_plot=False,
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
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(kz_list, np.abs(np.real(freq_lst)), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$kz\rho_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $k_z\rho_p$ from {0} to {1}'.format(seed_kz, target_value))
        plt.show()
    new_param = (target_value, kp, beta, t_list, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)

def follow_kp(seed_freq, target_value, param, pol='r', show_plot=False,
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
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(kp_list, np.abs(np.real(freq_lst)), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$kp\rho_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $kp\rho_p$ from {0} to {1}'.format(seed_kp, target_value))
        plt.show()        
    new_param = (kz, target_value, beta, t_list, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)

def follow_angle(seed_freq, target_value, param, pol='r', show_plot=False,
                 log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow mode along angles
    """
    
    return 0

def follow_kzkp(seed_freq, target_value, param, pol='r', show_plot=False,
                 log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow mode in wavenumber plane by incrementally vary
    kz and kp in turns.

    """
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    target_kz, target_kp = target_value
    delt_kz, delt_kp = target_kz- kz, target_kp- kp

    ## see if we can reduce to follow along kz or follow along kp cases.
    
    if delt_kz == 0:
        return follow_kp(seed_freq, target_value, param, pol=pol,
                         show_plot=show_plot, log_incrmt=log_incrmt,
                         lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)
    elif delt_kp == 0:
        return follow_kp(seed_freq, target_value, param, pol=pol,
                         show_plot=show_plot, log_incrmt=log_incrmt,
                         lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)
    
    ## now both kz and kp need to vary. We vary both of them in log scale.
    
    seed_k = np.sqrt(kz **2 + kp **2)
    delt_k = np.sqrt(delt_kz **2 + delt_kp **2)

    ## a list of k to step through
    
    k_inc_list = generate_steps(seed_k, seed_k + delt_k, log_incrmt=log_incrmt,
                            lin_incrmt=lin_incrmt,
                            incrmt_method = incrmt_method) - seed_k
    kz_k = delt_kz/delt_k
    kp_k = delt_kp/delt_k
    freq_lst = []
    guess = seed_freq

    for k_inc in k_inc_list:
        kz_loc = kz + kz_k * k_inc
        kp_loc = kp + kp_k * k_inc
        guess = solve_disp(guess, kz_loc, kp_loc, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
        
    new_param = (target_kz, target_kp, beta, t_list, a_list, n_list,
                 q_list, m_list, v_list, n, method, aol)
    return (guess, new_param, freq_lst)        

def follow_k(seed_freq, target_value, param, pol='r', show_plot=False,
             log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow mode along wavenumber parameter.
    """    
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_k = np.sqrt(kz**2 + kp**2)
    kz_k = kz/seed_k
    kp_k = kp/seed_k
    # a list of k to step through
    k_list = generate_steps(seed_k, target_value, log_incrmt=log_incrmt,
                             lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)
    freq_lst = []
    guess = seed_freq
    for k in k_list:
        kz, kp = k * kz_k, k * kp_k
        try:
            guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                               n_list, q_list, m_list, v_list,
                               n, method, aol, pol)
        except Exception:
            print('kz = {0:.3g}'.format(kz))
            print('kp = {0:.3g}'.format(kp))
            print('guess = {0:.3g}'.format(guess))
        freq_lst.append(guess)
    if show_plot:
        plt.plot(k_list, np.abs(np.real(freq_lst)), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$k\rho_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $k\rho_p$ from {0} to {1}'.format(seed_k, target_value))
        plt.show()        
    new_param = (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)

def follow_beta(seed_freq, target_value, param, pol='r', show_plot=False,
                log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow a mode with frequency SEED_FREQ in a plasma specified by
    PARAM along the beta (proton parallel beta).
    """    
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_beta = beta
    # a list of beta to step through
    beta_list = generate_steps(beta, target_value, log_incrmt=log_incrmt,
                               lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)

    freq_lst = []
    guess = seed_freq
    for beta in beta_list:
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(beta_list, np.abs(np.real(freq_lst)), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\beta_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $\beta_p$ from {0} to {1}'.
                  format(seed_beta, target_value))
        plt.show()        
    new_param = (kz, kp, target_value, t_list, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)

def follow_temperature(seed_freq, target_value, param, pol='r', show_plot=False,
                       log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log'):
    """
    follow a mode with frequency SEED_FREQ in a plasma specified by
    PARAM along the temperature ratio parameter.
    """    
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_t_list = t_list
    # a list of kp to step through
    t_list_steps = generate_steps(t_list, target_value, log_incrmt=log_incrmt,
                             lin_incrmt=lin_incrmt, incrmt_method = incrmt_method)

    freq_lst = []
    guess = seed_freq
    for t_list in t_list_steps:
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(np.abs(np.real(freq_lst)),'o-', markersize= 2)
        plt.yscale('log')
        plt.xlabel('step')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.show()        
    new_param = (kz, kp, beta, target_value, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)

def follow_anisotropy(seed_freq, target_value, param, pol='r', show_plot=False,
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
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(a_list_steps[:,0], np.abs(np.real(freq_lst)),
                 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('step')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $T_{s\perp}/T_{s\parallel}$')
        plt.show()        
    new_param = (kz, kp, beta, t_list, target_value, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)


def generate_vn_steps(lst, target_lst, lin_incrmt):
    """
    Generate search steps when following solution along v_list or n_list.
    Gurantees linear change from 
    """
    start = np.array(lst)
    end = np.array(target_lst)
    num_step = 1 + np.amax(np.abs(start - end)/lin_incrmt)
    num_step = int(round(num_step))
    incrmt_list = (end - start)/num_step
    res = np.array([start + incrmt_list * i  for i in range(num_step+1)])
    return res

def follow_drift(seed_freq, target_value, param, pol='r', show_plot=False,
                 lin_incrmt=0.1):
    """
    follow a mode with frequency SEED_FREQ in a plasma specified by
    PARAM along the drift parameter.
    
    Caution: system with current are difficult to follow since dispersion
    relation changes quickly.
    We do not intend to provide assurance that solution converges.
    """
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_v_list = v_list
    # a list of drift to step through
    v_list_steps = generate_vn_steps(v_list, target_value,
                                    lin_incrmt=lin_incrmt)
    freq_lst = []
    guess = seed_freq
    for v_list in v_list_steps:
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(np.abs(np.real(freq_lst)), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('step')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change drift speed')
        plt.show()        
    new_param = (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)

def follow_density(seed_freq, target_value, param, pol='r', show_plot=False,
                   lin_incrmt=0.1):
    """
    follow a mode with frequency SEED_FREQ in a plasma specified by
    PARAM along the drift parameter.
    
    Caution: system with current are difficult to follow since dispersion
    relation changes quickly.
    We do not intend to provide assurance that solution converges.
    """
    (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
     v_list, n, method, aol) = param
    seed_n_list = n_list
    # a list of relative density to step through
    n_list_steps = generate_vn_steps(n_list, target_value,
                                    lin_incrmt=lin_incrmt)
    freq_lst = []
    guess = seed_freq
    for n_list in n_list_steps:
        guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                           n_list, q_list, m_list, v_list,
                           n, method, aol, pol)
        freq_lst.append(guess)
    if show_plot:
        plt.plot(np.abs(np.real(freq_lst)), 'o-', markersize= 2)
        plt.yscale('log')
        plt.xlabel('step')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change density')
        plt.show()        
    new_param = (kz, kp, beta, t_list, a_list, n_list, q_list, m_list,
                 v_list, n, method, aol)
    return (guess, new_param, freq_lst)


