# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 19:57:02 2025

@author: Joaquin
"""
import numpy as np
import matplotlib.pyplot as plt
r = 1
l = 0.1
c= 0.00001
t= 100000
tau= 1/r*c
omega_cuadr= 1/l*c
def mi_funcion(tau,omega_cuadr,t):
    w = np.arange(start = 0,stop = t , step = 1)
    t = 20*np.log10(w^2/np.sqrt((omega_cuadr-w^2)^2+w^2*tau^2))
    return w,t

xx,tt = mi_funcion(tau, omega_cuadr, t)

plt.plot(xx,tt)