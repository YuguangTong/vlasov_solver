import numpy as np

#---------------------------------------------#
# consider expansion to include changes in mass
#---------------------------------------------#
def follow_eigen_freq(seed_freq, target_param, target_value,
                      param, num, guess_fn, show_plot=False):
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
        return follow_kz(seed_freq, target_value,
                         param, num, guess_fn, show_plot=False)
    elif target_parm == 'kp':
        return follow_kp(seed_freq, target_param, target_value,
                         param, num, guess_fn, show_plot=False)
    elif target_parm == 'k':
        return follow_k(seed_freq, target_value,
                        param, num, guess_fn, show_plot=False)
    elif target_parm == 'angle':
        return follow_angle(seed_freq, target_value,
                            param, num, guess_fn, show_plot=False)
    elif target_parm == 'beta':
        return follow_beta(seed_freq, target_value,
                           param, num, guess_fn, show_plot=False)
    elif target_parm == 'temperature':
        return follow_temperature(seed_freq, target_value,
                                  param, num, guess_fn, show_plot=False)
    elif target_parm == 'anisotropy':
        return follow_anisotropy(seed_freq, target_value,
                                 param, num, guess_fn, show_plot=False)
    elif target_parm == 'drift':
        return follow_drift(seed_freq, target_value,
                            param, num, guess_fn, show_plot=False)
    else:
        raise VlasovException('{0} is not a valid parameter to vary'.
                              format(target_param))


def follow_kz(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """
    return 0

def follow_kp(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """
    return 0

def follow_angle(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """
    
    return 0

def follow_k(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_beta(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_temperature(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_anisotropy(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0

def follow_drift(seed_freq, target_value, param, num, guess_fn, show_plot=False):
    """
    to implement
    """    
    return 0



