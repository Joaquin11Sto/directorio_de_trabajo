#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:55:33 2023

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación

fs = 250 # Hz
N = fs

delta_f = fs / N # Hz

Ts = 1/fs
T_simulacion = N * Ts # segundo


# Parámetros de la señal
fx = 1 # Hz
ax = 10# V


# grilla temporal
tt = np.arange(start = 0, stop = T_simulacion, step = Ts)

xx = ax * np.sin( 2 * np.pi * fx * tt )

#plt.plot(tt, xx)

x_1 = np.var(xx) 
x_2 = np.mean(xx)
x_3 = np.std(xx)
x_5 = xx/x_3 # para cualquier señal, primero estandarizo el vector de mi grafica,
x_4= np.var(x_5) # luego obtengo la varianza, que 

print("primero: ", x_1)
print("segundo: ", x_2)
print("tercero: ", x_3)
print("cuarto: ", x_4)

x_3
