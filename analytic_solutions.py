import numpy as np
import scipy.integrate as integrate
from blast_wave_plots import TS_bench_prime, TS_bench, TS_current, square_blast_phi_vector
from show import show
import math
from sedov_uncollided import sedov_uncollided_solutions
from sedov_funcs import sedov_class
from functions import quadrature
import quadpy
from functions import get_sedov_funcs
from cubic_spline import cubic_spline_ob as cubic_spline
import matplotlib.pyplot as plt

def TS_bench_prime(sigma_t = 1e-3, eblast = 1e20, tstar = 1e-12):
    f_fun, g_fun, l_fun = get_sedov_funcs()
    sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t, eblast = eblast, tstar = tstar)
    foundzero = False
    iterator = 0
    while foundzero == False:
        if g_fun[iterator] == 0.0:
            iterator += 1
        else:
            foundzero = True
        if iterator >= g_fun.size:
                assert(0)
    f_fun = f_fun[iterator:]
    g_fun = g_fun[iterator:]
    l_fun = l_fun[iterator:]
    
    interp_g_fun, interp_v_fun = interp_sedov_selfsim(g_fun, l_fun, f_fun)

    # plt.figure(24)
    # plt.ion()
    # plt.plot(l_fun, interp_g_fun.eval_spline(l_fun))
    # plt.show()
    # plt.figure(25)

    # plt.figure(24)
    # plt.ion()
    # plt.plot(l_fun, interp_v_fun.eval_spline(l_fun))
    # plt.show()
    # plt.figure(25)

    # rs = np.linspace(-0.5, 0.5,100)
    # plt.ion()
    # plt.plot(rs, sedov.interpolate_self_similar(1.0, rs, interp_g_fun))
    # plt.show()
    



    return interp_g_fun, interp_v_fun, sedov
def integrate_sedov_selfsim(beta = 3.5, sigma_t=0.001, x0=0.15):
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
     def integrand(x):
        return g_interp.eval_spline(np.array([x]))**beta
     integral = integrate.quad(integrand, 0,1)[0]
     return integral
def interp_sedov_selfsim(g_fun, l_fun, f_fun):

             l_fun = np.flip(l_fun)
             l_fun[0] = 0.0
             g_fun = np.flip(g_fun)
             f_fun[-1] = 0.0
             l_fun[-1] = 1.0
            #  g_fun[-1] = sedov_class.gpogm
            #  l_fun[-1] = 1.0
             g_fun[-1] = 1.0

             interpolated_g = cubic_spline(l_fun, g_fun)
             interpolated_v = cubic_spline(l_fun, np.flip(f_fun))
            #  print('g interpolated')
             return interpolated_g, interpolated_v

def analytic_detector_lambda(t, x0, mu, E0, beta, sedov_class, omega, sigma_t = 1e-3):
    gamma = 7/5
    gammp1 = (gamma+1)/(gamma-1)
    v = 29.98
    rho0 = 1
    rho2 = 6
    x = x0
    # chi = 
    # print(c1)
    # tt = (integral_bound1-x)/mu + t
    a = x-t*mu
    c1 =  (E0/(sedov_class.alpha*sedov_class.rho0))**(1.0/3)* (10**-9 /v/sigma_t)**(2/3) * sigma_t
    xitest =   -0.3333333333333333*(c1**3 + 3*a*mu**2)/mu**3 - (2**0.3333333333333333*(-c1**6 - 6*a*c1**3*mu**2))/(3.*mu**3*(-2*c1**9 - 18*a*c1**6*mu**2 - 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333) + (-2*c1**9 - 18*a*c1**6*mu**2 - 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333/(3.*2**0.3333333333333333*mu**3)
    # print(tt,xitest)
    xbtest = -0.3333333333333333*(-c1**3 + 3*a*mu**2)/mu**3 - (2**0.3333333333333333*(-c1**6 + 6*a*c1**3*mu**2))/(3.*mu**3*(2*c1**9 - 18*a*c1**6*mu**2 + 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(-4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333) + (2*c1**9 - 18*a*c1**6*mu**2 + 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(-4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333/(3.*2**0.3333333333333333*mu**3)
    # print(-c1 * tt**(2/3), integral_bound1)

    r2a = (xitest-t)*mu + x
    r2b = (xbtest-t)*mu + x
    
    res = rho0**beta * (2*x0 + (r2b-r2a) * ((gammp1)**beta * omega-1))
    return res



def check_lambda(t=2, x0=0.15, E0=1e20, beta = 2.6, sigma_t = 1e-3):
    plt.ion()
    omega = integrate_sedov_selfsim(beta = beta, sigma_t=sigma_t, x0=x0)
    mulist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    lambdalist = mulist * 0
    lambdalistanalytic = mulist * 0

    xs_quad, ws_quad = quadrature(20, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(64)
    mu_quad = res1.points
    mu_ws = res1.weights

    interp_g_fun, interp_v_fun, sedov = TS_bench_prime(sigma_t = sigma_t, eblast = E0, tstar = 1e-12)
    # sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t, eblast = E0, tstar = 0)


    
    sedov_uncol = sedov_uncollided_solutions(xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_t, t0 =1000, transform= True, lambda_1 = beta, relativistic= True, analytic_contact=True)
    for im, mu in enumerate(mulist):
        if t> 2*x0/mu:
            lambdalistanalytic[im] = analytic_detector_lambda(t, x0, mu, E0, beta, sedov, omega)
        
            lambdalist[im] = sedov_uncol.integrate_sigma(x0, mu, t, sedov, interp_g_fun, interp_v_fun)
    
    plt.plot(mulist, lambdalistanalytic, 'b-o')
    plt.plot(mulist, lambdalist, 'k:')
    plt.show()

def analytic_profile(tf = 5.5, sigma_t = 1e-3, x0 = 0.15, beta = 2.7, t0 = 1.0, eblast = 1e20, plotnonrel = False, plotbad = False, relativistic = True, tstar = 1e-12, npts = 250):
    g_interp, v_interp, sedov = TS_bench_prime(sigma_t, eblast, tstar)
    plt.ion()
    f, (a1, a2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    xs = np.linspace(-x0, x0, npts)
    phi_bench = TS_bench(tf, xs, g_interp, v_interp, sedov, beta = beta, transform=True, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = relativistic)
    xrho = np.linspace(-x0, x0, 5000)
    density_final = sedov.interpolate_self_similar(tf, xrho, g_interp)
         
    current = TS_current(tf, xs, g_interp, v_interp, sedov, beta = beta ,transform=True, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = relativistic)
#  if plotbad == True:
    plt.figure(1)
    a2.plot(xs/sigma_t, phi_bench, 'k-', mfc = 'none')
    a1.plot(xrho/sigma_t, density_final, 'k-')
    if plotnonrel == True:
         phi_bench_bad = TS_bench(tf, xs, g_interp, v_interp, sedov,beta = beta, transform=True, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False )
         a2.plot(xs/sigma_t, phi_bench_bad, 'k--', mfc = 'none')
    
    elif plotbad == True:
        phi_bench_nottransformed = TS_bench(tf, xs, g_interp, v_interp, sedov,beta = beta, transform=False, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False )
        a2.plot(xs/sigma_t, phi_bench_nottransformed, 'k--', mfc = 'none')
         
    a2.set_ylabel(r'$\phi$', fontsize = 16)
    a2.set_xlabel('x [cm]', fontsize = 16)
    a1.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
    # a2.ylim(0, 1.1)
    show(f'blast_plots/analytic_phi_t={tf}_E0={eblast}_beta={beta}_x0={x0/sigma_t}_t0={t0}')
    plt.show()
    # plt.savefig(f'analytic_phi_t={tf}_E0={eblast}_beta={beta}_x0={x0/sigma_t}_t0={t0}.pdf')
    
    plt.close()
    f, (a3, a4) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    plt.figure(2)
    
    a4.plot(xs/sigma_t, current, 'k-', mfc = 'none')
    a3.plot(xrho/sigma_t, density_final, 'k-')
    if plotnonrel == True:
         current_bad = TS_current(tf, xs, g_interp, v_interp, sedov, beta = beta ,transform=True, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)
         a4.plot(xs/sigma_t, current_bad, 'k--', mfc = 'none')
    
    elif plotbad == True:
        current_nottransformed = TS_current(tf, xs, g_interp, v_interp, sedov, beta = beta, transform=False, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)
        a4.plot(xs/sigma_t, current_nottransformed, 'k--', mfc = 'none')
         
    a4.set_ylabel(r'$J$', fontsize = 16)
    a3.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
    a4.set_xlabel('x [cm]', fontsize = 16)
    # a2.ylim(0, 0.6)
    
    show(f'blast_plots/analytic_J_t={tf}_E0={eblast}_beta={beta}_x0={x0/sigma_t}_t0={t0}')
    plt.show()
    # plt.savefig(f'analytic_J_t={tf}_E0={eblast}_beta={beta}_x0={x0/sigma_t}_t0={t0}.pdf')
    
    
    plt.close()
    # plt.close()
    # plt.close()

def paper_example_plots():
     analytic_profile(tf=0.1, t0 = 1.00000000001)
    #  plt.close()
    #  plt.close()
     analytic_profile(tf=0.16, t0 = 1.00000000001)
    #  plt.close()
    #  plt.close()
     analytic_profile(tf=0.22, t0 = 1.00000000001)
    #  plt.close()
    #  plt.close()
     analytic_profile(tf=1, t0 = 1.00000000001)
    #  plt.close()
    #  plt.close()
     analytic_profile(tf=1.05, t0 = 1.00000000001)
    #  plt.close()
    #  plt.close()
     analytic_profile(tf=1.15, t0 = 1.00000000001)



def plot_square_blast(t, x0 = -150, v0 = 0.01, t0source = 100, npts = 250):
     if v0 * t > abs(x0):
          raise ValueError('blast has passed the source')
     plt.ion()
     xs = np.linspace(-x0,x0, npts)
     phi = square_blast_phi_vector(t, xs, v0, t0source, x0)
     plt.plot(xs, phi, 'k-')
     plt.show()


def analytic_square(tf = 15.5, sigma_t = 1e-3, x0 = 0.15, v0 = 0.01, t0 = 1000.0, npts = 250):
    if v0 * tf > abs(x0/sigma_t):
          raise ValueError('blast has passed the source')
    plt.ion()
    rho0 = 1
    rho2 = 6
    xs = np.linspace(-x0/sigma_t, x0/sigma_t, npts)
    phi = square_blast_phi_vector(tf, xs, v0, t0, -x0/sigma_t)
    xrho = np.linspace(-x0/sigma_t, x0/sigma_t, 5000)
    def density_func(x):
         res = x*0
         for ix, xx in enumerate(x):
              if abs(xx) < v0 * tf:
                   res[ix] = rho2
              else:
                   res[ix] = rho0
         return res
                   
              
    density_final = density_func(xrho)
         
    
#  if plotbad == True:
    f, (a1, a2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    plt.figure(1)
    a2.plot(xs, phi, 'k-', mfc = 'none')
    a1.plot(xrho, density_final, 'k-')
         
    a2.set_ylabel(r'$\phi$', fontsize = 16)
    a2.set_xlabel('x [cm]', fontsize = 16)
    a1.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
    # a2.ylim(0, 1.1)
    show(f'blast_plots/square_blast_tf={tf}_v0={v0}')
    plt.show()
    # plt.savefig(f'analytic_phi_t={tf}_E0={eblast}_beta={beta}_x0={x0/sigma_t}_t0={t0}.pdf')
    
    # plt.close()
def plot_square_blast_detector(tf, x0 = -150, v0 = 0.01, t0source = 10000, npts = 250):
    # 
     plt.ion()
     xs = np.array([-x0-0.0000000001])
     ts = np.linspace(0.0001, tf, npts)
     phi = ts * 0

     for it, tt in enumerate(ts):
        if v0 * tt > abs(x0):
          raise ValueError('blast has passed the source')
        phi[it] = square_blast_phi_vector(tt, xs, v0, t0source, x0)[0]
     plt.plot(ts/29.98, phi, 'k-')
     plt.xlabel(r'$\phi$', fontsize = 16)
     plt.ylabel('t [ns]', fontsize = 16)
     show(f'blast_plots/square_detector_tf={tf}_v0={v0}')
     plt.show()
     plt.close()