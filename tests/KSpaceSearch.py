import numpy as np
import scipy.interpolate
from py_vlasov.util import kzkp
from py_vlasov.follow_parameter import generate_steps, solve_disp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from math import ceil

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3.5
mpl.rcParams['ytick.minor.size'] = 3.5
mpl.rcParams['xtick.major.width'] = 1.1
mpl.rcParams['ytick.major.width'] = 1.1
mpl.rcParams['xtick.minor.width'] = .8
mpl.rcParams['ytick.minor.width'] = .8
mpl.rcParams['legend.fontsize'] = 14

class KThetaScan:
    """
    Scan a wave mode in wavenumber (k) vector. 
    k vector is parametrized by magnitude k and angle theta w.r.t. mean magnetic field.
    """
    def __init__(self, param,
                 guessFn = None, 
                 incrmt_method = 'linear', 
                 lin_incrmt = 0.05, 
                 log_incrmt = 0.05, 
                 stop_condition = None,
                 diagnostic_dir = '.', 
                 savefig = False):
        """
        Constructor
        
        param: a list containing 
            [kz, kp, betap, t_list, a_list, n_list, q_list, m_list, v_list, n, method, aol]
        """
        self.param = param
        if guessFn:
            self.guessFn = guessFn
        else:
            print("No user provided wave mode guess function. Use default Alfven guess.")
            self.guessFn = self.AlfvenGuess
        self.incrmt_method = incrmt_method
        self.lin_incrmt = lin_incrmt
        self.log_incrmt = log_incrmt
        self.stop_condition = stop_condition
        self.maxGrowthRate = None
        self.maxGrowthK = None
        self.maxGrowthKMag = None
        self.maxGrowthTheta = None    
        self.savefig = savefig
        try: 
            self.diagnostic_dir = os.path.abspath(diagnostic_dir)
        except Exception:
            print("Do not recognize directory {0}".format(diagnostic_dir))
            print("Use current directory instead {0}".format(os.path.curdir))
            self.diagnostic_dir = os.path.abspath(os.path.curdir)
    
    def getParamString(self):
        return self.getElectronDriftParamString()
    
    def getElectronDriftParamString(self):
        betap = self.param[2]
        t_list = self.param[3]
        n_list = self.param[4]
        v_list = self.param[8]
        aol = self.param[11]
        tctp = t_list[1]/t_list[0]
        thtc = t_list[2]/t_list[1]
        vcva = v_list[1]
        va2c2 = aol
        annot = r"$\beta_p = ${0:.3g}".format(betap) + "\n"
        annot += "$v_c/v_A = ${0:.3g}".format(vcva) + "\n"
        annot += "$T_c/T_p = ${0:.2g}".format(tctp) + "\n"
        annot += "$T_h/T_c = ${0:.2g}".format(thtc) + "\n"
        annot += "$v_A/c = ${0:.2g}".format(aol)
        
        title = "betap={0:.2g}_vcva={1:.3g}_tctp={2:.2g}_thtc={3:.2g}_vac={4:.2g}".format(betap, vcva, tctp, thtc, aol)
        return annot, title
        
    def guess(self, kz, kp):
        return self.guessFn(kz, kp)
       
    def AlfvenGuess(self, kz, kp):
        betap = self.param[2]
        return kz / np.sqrt(betap)
 
    def follow_k(self, seed_freq, target_value, param, pol='r', show_plot=False,
                 log_incrmt=0.1, lin_incrmt=0.1, incrmt_method = 'log', stop_condition = None):
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
            guess = solve_disp(guess, kz, kp, beta, t_list, a_list,
                               n_list, q_list, m_list, v_list,
                               n, method, aol, pol)
            if stop_condition and stop_condition(guess):
                break
            freq_lst.append(guess)
        return freq_lst

    def fixThetaScan(self, theta, kmin, kmax):
        kz, kp = kzkp(kmin, theta)
        guess = self.guess(kz, kp)
        self.param[0] = kz
        self.param[1] = kp


        freq = self.follow_k(guess, kmax, self.param,
                        lin_incrmt = self.lin_incrmt, 
                        log_incrmt = self.log_incrmt, 
                        incrmt_method = self.incrmt_method,
                        stop_condition = self.stop_condition)  
        k_arr = generate_steps(kmin, kmax, log_incrmt=self.log_incrmt,
                             lin_incrmt=self.lin_incrmt, incrmt_method = self.incrmt_method)
        kz_arr = k_arr * np.cos(np.deg2rad(theta))
        kp_arr = k_arr * np.sin(np.deg2rad(theta))
        wrel_arr = np.array(freq)
        return kz_arr, kp_arr, k_arr, wrel_arr
    
    def fullScan(self, theta_arr, kmin, kmax):
        k_arr = generate_steps(kmin, kmax, log_incrmt=self.log_incrmt,
                             lin_incrmt=self.lin_incrmt, incrmt_method = self.incrmt_method)
        kz_2d = np.zeros((len(theta_arr), len(k_arr)), dtype = float)
        kp_2d = np.zeros((len(theta_arr), len(k_arr)), dtype = float)
        k_2d = np.zeros((len(theta_arr), len(k_arr)), dtype = float)
        wrel_2d = np.zeros((len(theta_arr), len(k_arr)), dtype = complex)
        wrel_2d.fill(np.nan + 1j * np.nan)
        
        for i, theta in enumerate(theta_arr):
            kz_arr, kp_arr, k_arr, wrel_arr = self.fixThetaScan(theta, kmin, kmax)
            kz_2d[i, :] = kz_arr
            kp_2d[i, :] = kp_arr
            k_2d[i, :] = k_arr
            wrel_2d[i, :len(wrel_arr)] = wrel_arr        
        return kz_2d, kp_2d, k_2d, wrel_2d 

    def edges2grid(self, edges):
        """
        convert edges of histogram bins (from numpy.hist2d output) 
        to grid (center of the bins)
        """
        grid = (np.roll(edges, 1) + edges)/2
        return grid[1:]

    def makeDiagnostics2D(self, theta_arr, kz_2d, kp_2d, wrel_2d):
        mask = ~np.isnan(wrel_2d)
        
        xmin, xmax = np.min(kz_2d[mask]), np.max(kz_2d[mask])
        ymin, ymax = np.min(kp_2d[mask]), np.max(kp_2d[mask])

        x_edges, y_edges = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
        x_grid, y_grid = self.edges2grid(x_edges), self.edges2grid(y_edges)
        x_i, y_i = np.meshgrid(x_grid, y_grid)
        x_2d = kz_2d
        y_2d = kp_2d
        z_2d = wrel_2d.real
        
        mask = ~np.isnan(wrel_2d)
        wrel_real_i = scipy.interpolate.griddata((x_2d[mask].flatten(), y_2d[mask].flatten()), 
                                                 wrel_2d[mask].real.flatten(), (x_i, y_i), 
                                                 method='linear')

        wrel_imag_i = scipy.interpolate.griddata((x_2d[mask].flatten(), y_2d[mask].flatten()), 
                                                 wrel_2d[mask].imag.flatten(), (x_i, y_i), 
                                                 method='cubic')
        wrel_imag_i[np.isnan(wrel_imag_i)] = 0
        
        self.maxGrowthRate = np.max(wrel_imag_i[~np.isnan(wrel_imag_i)])
        peak_pos = np.where(wrel_imag_i == self.maxGrowthRate)
        peak_kz = x_i[peak_pos][-1]
        peak_kp = y_i[peak_pos][-1]
        self.maxGrowthK = [peak_kz, peak_kp]
        self.maxGrowthKMag = np.sqrt(peak_kz**2 + peak_kp**2)
        self.maxGrowthTheta = np.rad2deg(np.arctan(peak_kp/peak_kz))
        # texts to annotate image & for naming image files
        annot, title = self.getParamString()        
    
        z = np.ma.masked_where(np.isnan(wrel_real_i), wrel_real_i)
        plt.pcolormesh(x_edges, y_edges, 
                       z, vmin=1e-2, vmax=np.max(z), cmap = plt.cm.jet)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.plot([peak_kz], [peak_kp], 'k+')
        plt.xlabel(r'$k_\parallel\rho_p$')
        plt.ylabel(r'$k_\perp \rho_p$')
        plt.title(r'$\omega_r/\Omega_p$')
        plt.text(0.5 * xmax, 0.3 * ymax, annot)
        plt.colorbar()
        plt.tight_layout()
        fileName = os.path.join(self.diagnostic_dir, title + '_real_freq_contour.png')
        if self.savefig:
            plt.savefig(fileName, dpi = 200)
        plt.show()  
        
        if self.maxGrowthRate < 1e-6:
            return 
        z = np.ma.masked_where(wrel_imag_i <= 0, wrel_imag_i)
        vmax = np.max(z)
        vmin = np.max([1e-6, vmax/500])
        pcm = plt.pcolormesh(x_edges, y_edges, z, 
                             vmin=vmin, vmax=vmax, cmap = plt.cm.jet, 
                             norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        # we only make two contour lines
        max_contour = np.floor(np.log(self.maxGrowthRate))
        contour_vals = [10**(max_contour-1), 10**max_contour]
        CS = plt.contour(x_grid, y_grid, z, contour_vals, colors = 'k')
        plt.clabel(CS, inline=1, fontsize=10, fmt='%.2g')
        xmax, ymax = np.max(kz_2d), np.max(kp_2d)
        plt.xlim([0, xmax])
        plt.ylim([0, ymax])
        plt.plot([peak_kz], [peak_kp], 'k+')
        # 45 degree reference line
        plt.plot([0, 1], [0, 1], 'k:')
        plt.plot([0, 5], [0, 5 * np.tan(np.deg2rad(60))], 'k:')
        plt.plot([0, 10], [0, 10 * np.tan(np.deg2rad(80))], 'k:')
        plt.plot([0, 10], [0, 10 * np.tan(np.deg2rad(85))], 'k:')  
        plt.plot([0, 10], [0, 10 * np.tan(np.deg2rad(89))], 'k:')        
        plt.xlabel(r'$k_\parallel\rho_p$')
        plt.ylabel(r'$k_\perp \rho_p$')
        plt.title(r'$\omega_i/\Omega_p$')
        plt.axvline(0, color = 'k')
        plt.text(0.5 * xmax, 0.3 * ymax, annot)
        plt.colorbar(pcm)
        plt.tight_layout()
        fileName = os.path.join(self.diagnostic_dir, title + '_growth_rate_contour.png')
        if self.savefig:
            plt.savefig(fileName, dpi = 200)
        plt.show()
        
        
    def makeDiagnostics1D(self, theta_arr, k_2d, wrel_2d):
        """
        Produce diagnostics plots:
            1, real freq vs wave number.
            2, imaginary freq vs wave number.
        """
        # at most label 5 curves
        n = ceil(len(theta_arr)/5)
        for i, wrel_arr in enumerate(wrel_2d):
            if i % n == 0:
                plt.plot(k_2d[i, :], wrel_arr.real, label = '{0:.3g}'.format(theta_arr[i]))
            else:
                plt.plot(k_2d[i, :], wrel_arr.real)
        not_nan = ~np.isnan(wrel_arr)
        wrel_real_pos = 0.2 * np.max(wrel_arr[not_nan].real) + 0.8 * np.min(wrel_arr[not_nan].real)
        wrel_imag_pos = 0.5 * np.max(wrel_arr[not_nan].imag) + 0.5* np.min(wrel_arr[not_nan].imag)
        k_real_pos = 0.3 * k_2d[0, 0] + 0.7 * k_2d[0, -1]
        k_imag_pos = 0.8 * k_2d[0, 0] + 0.2 * k_2d[0, -1]
        annot, title = self.getParamString()
        print(title)
        plt.xlabel(r'$k\rho_p$')
        plt.ylabel(r'$\omega_r/\Omega_p$')
        plt.legend(fontsize = 10, frameon=False)
        plt.text(k_real_pos, wrel_real_pos, annot)
        plt.tight_layout()
        fileName = os.path.join(self.diagnostic_dir, title + '_real_freq.png')
        if self.savefig:
            plt.savefig(fileName, dpi = 100)
        plt.show()
        
        for i, wrel_arr in enumerate(wrel_2d):
            if i % n == 0:
                plt.plot(k_2d[i, :], wrel_arr.imag, label = '{0:.3g}'.format(theta_arr[i]))
            else:
                plt.plot(k_2d[i, :], wrel_arr.imag)
        plt.xlabel(r'$k\rho_p$')
        plt.ylabel(r'$\omega_i/\Omega_p$')
        mask = ~np.isnan(wrel_2d)
        gamma_max = np.max(wrel_2d[mask].imag)
        if gamma_max > 0:
            plt.ylim([-1e-3, 2 * gamma_max])
            plt.text(k_imag_pos, -1e-3, annot)
        else:
            plt.text(k_imag_pos, wrel_imag_pos, annot)
        plt.legend(fontsize = 10, frameon=False)
        plt.tight_layout()
        fileName = os.path.join(self.diagnostic_dir, title + '_growth_rate.png')
        if self.savefig:
            plt.savefig(fileName, dpi = 100) 
        plt.show()