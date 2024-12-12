import numpy as np
import scipy.integrate as integrate
from blast_wave_plots import TS_bench_prime
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

