## provide functions to find a certain mode and follow it.

import numpy as np
import scipy.special, scipy.optimize
from .dispersion_tensor import dt_wrapper
from .util import (real_imag, list_to_complex)
import matplotlib.pyplot as plt


"""
dimensionless plasma parameters take the form:
freq, (k, theta, beta, te/tp, method, mratio, n, aol)
"""
def change_angle(seed_freq, param, target_angle, num, follow_angle_fn, show_plot=False):
    """
    Follow the solution and incrementally change angles to the target_angle.
    
    Parameters
    ----------
    freq: w/proton_gyro_frequency
    param: list (k, theta, beta, te/tp, method, mratio, n, aol)
    target_angle: target propogation angle
    num: number of steps
    follow_angle_fn: how to follow from one angle to another angle
    show_plot: whether to show a plot of intermediate steps

    Return
    ------
    freq at the target angle
    """
    k, seed_angle, beta, tetp, method, mass_ratio, n, aol = param
    start = seed_angle
    stop = target_angle
    angle_list_1 = np.linspace(start, stop, num, endpoint=True)
    theta = seed_angle * np.pi/180
    result_1 = [seed_freq]
    for i in list(range(num))[1:]:
        theta = angle_list_1[i]
        prev_angle = angle_list_1[i-1]
        prev_result = result_1[-1]
        guess = follow_angle_fn(prev_result, prev_angle, theta)
        theta = theta * np.pi/180.
        kz = k * np.cos(theta)
        kp = k * np.sin(theta)
        f = lambda wrel: real_imag(dt_wrapper(wrel[0] + 1j * wrel[1], kp, kz, beta, tetp, 'numpy', mass_ratio, n, aol))
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        result_1 += [list_to_complex(freq)]

    if show_plot:
        plt.plot(angle_list_1, np.real(result_1), 'o-', markersize= 2)
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $\theta$ from {0} to {1}'.format(seed_angle, target_angle))
        plt.show()
    return result_1[-1]

def simple_follow_fn(prev_result, prev_param_val, cur_param_val):
    """
    Follow a solution by using the previous result directly as a guess for \
    the new result.

    """
    return prev_result
    
def change_beta(seed_freq, param, target_beta, num, follow_beta_fn=simple_follow_fn, show_plot=True):
    """
    Follow the solution and incrementally change angles to the target_angle.
    
    Parameters
    ----------
    freq: w/proton_gyro_frequency
    param: list (k, theta, beta, te/tp, method, mratio, n, aol)
    target_angle: target propogation angle
    num: number of steps
    follow_beta_fn: how to follow from one beta to another beta
    show_plot: whether to show a plot of intermediate steps

    Return
    ------
    freq at the target angle
    """
    k, theta, seed_beta, tetp, method, mass_ratio, n, aol = param
    start = seed_beta
    stop = target_beta
    beta_list = np.linspace(start, stop, num, endpoint=True)
    theta = theta * np.pi/180
    kz = k * np.cos(theta)
    kp = k * np.sin(theta)
    result = [seed_freq]
    for i in list(range(num))[1:]:
        beta = beta_list[i]
        prev_beta = beta_list[i-1]
        prev_result = result[-1]
        guess = follow_beta_fn(prev_result, prev_beta, beta)
        f = lambda wrel: real_imag(dt_wrapper(wrel[0] + 1j * wrel[1], kp, kz, beta, tetp, 'numpy', mass_ratio, n, aol))
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        result += [list_to_complex(freq)]

    if show_plot:
        plt.plot(beta_list, np.real(result), 'o-', markersize= 2)
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\beta_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $\beta$ from {0} to {1}'.format(seed_beta, target_beta))
        plt.show()
    return result[-1]

def change_k(seed_freq, param, target_k, num_step, follow_k_fn=simple_follow_fn, step_method = 'log',  show_plot=True):
    """
    Follow the solution and incrementally change angles to the target_angle.
    
    Parameters
    ----------
    freq: w/proton_gyro_frequency
    param: list (k, theta, beta, te/tp, method, mratio, n, aol)
    target_angle: target propogation angle
    num_step: number of steps
    follow_beta_fn: how to follow from one beta to another beta
    show_plot: whether to show a plot of intermediate steps

    Return
    ------
    freq at the target angle
    """
    seed_k, theta, beta, tetp, method, mass_ratio, n, aol = param
    start = seed_k
    stop = target_k
    if step_method == 'log':
        k_list = np.logspace(np.log10(start), np.log10(stop), num_step, endpoint=True)
    else:
        k_list = np.linspace(start, stop, num_step, endpoint=True)
    theta = theta * np.pi/180
    result = [seed_freq]
    for i in list(range(num_step))[1:]:
        k = k_list[i]
        prev_k = k_list[i-1]
        prev_result = result[-1]
        guess = follow_k_fn(prev_result, prev_k, k)
        kz = k * np.cos(theta)
        kp = k * np.sin(theta)
        f = lambda wrel: real_imag(dt_wrapper(wrel[0] + 1j * wrel[1], kp, kz, beta, tetp, 'numpy', mass_ratio, n, aol))
        freq = scipy.optimize.fsolve(f, real_imag(guess))
        result += [list_to_complex(freq)]

    if show_plot:
        plt.plot(k_list, np.real(result), 'o-', markersize= 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$k\rho_p$')
        plt.ylabel(r'$\omega/\Omega_{ci}$')
        plt.title(r'Change $k\rho_p$ from {0} to {1}'.format(seed_k, target_k))
        plt.show()
    return result[-1]
