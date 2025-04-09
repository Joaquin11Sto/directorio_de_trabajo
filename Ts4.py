# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:10:07 2025

@author: Joaquin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
from scipy import signal
from scipy.fft import fft, fftshift


 
N=1000 # Cantidad de muestras
R=200 # Realizaciones
fs=1000 #Frecuecia de muestreo
a1= 1/np.sqrt(2) #Amplitud de la señal
df=fs/N #rResolucion espectral
tt = np.arange(0,1,1/N).reshape((N,1))#Vector de tiempo de columna N
tt = np.tile(tt, (1, R))
Pn=1/10 #Potencia de ruido cuantizado con 10 dB

omega_0=fs/4 # Frecuencia central
fr=np.random.uniform(-0.5,0.5,size=(1,R)) # Frecuencia aletearia

omega_1= omega_0 + fr*df 
xx = a1*np.sin(2*np.pi*omega_1*tt) # Hay que multiplicar por 2pi sino, no queda.

sigma=np.sqrt(1/10)
nn= np.random.normal(0,sigma,size=(N,1))
nn = np.tile(nn, (1, R))

S= xx + nn # Señal de ruido 

S_fft=1/N*np.fft.fft(S,axis=0)
ff=np.linspace(0, (N-1)*df, N)
bfrec= ff <= fs/2
plt.figure()
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(S_fft[bfrec, i])**2), label=f'Señal {i+1}')

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad espectral de potencia (dB)")
plt.legend()
plt.grid(True)
plt.show()