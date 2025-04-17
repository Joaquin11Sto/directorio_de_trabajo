# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 21:13:33 2025

@author: Joaquin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
from scipy import signal
from scipy.fft import fft, fftshift
import random

def mi_funcion_sen (vmax,dc,ff,ph,N,fs):
    
    ts = 1/fs # tiempo de muestreo o periodo
    tt=np.linspace (0, (N-1)*ts, N).flatten() #vector de tiempo

    xx= dc + vmax*np.sin(2*np.pi*ff*tt + ph)
    return tt, xx

#Datos del ADC
fs=1000 #Frecuencia de muestreo
N=1000 #Cantidad de muestras
f0=500 #Frecuencia arbitraria
tt,xx=mi_funcion_sen(4,0,f0,0,N,fs)

xn=xx/np.std(xx) # Normalizo la señal senoidal

B=8 #Cantidad de bits
Vf= 2 # rango simetrico o amplitud
q= Vf/2**(B-1) #Pasos de cuantificacion de q Volts

Pq=(q**2)/12 #Watts
k=1
Pn=k*Pq #

ts=1/fs # tiempo de muestreo
df = fs/N # resolución espectral

 #señal de ruido analogico
nn=np.random.normal(0,np.sqrt(Pn),N) # Ruido 
sr= xn + nn # Señal analogica mas ruido
srq= np.round(sr/q) * q# Señal analogica mas ruido cuantizada
nq= srq - sr


plt.figure(1)
plt.plot(tt,srq,lw=2)
plt.plot(tt,xn,color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot(tt,sr,color='green')
plt.title('Señal muestrada por un ADC')
plt.xlabel('Tiempo [t]')
plt.ylabel('Voltaje [V]')
plt.show()

plt.figure(2)
ft_As = 1/N*np.fft.fft(xn) # Señal analogica normalizada, con la fft
ft_SR = 1/N*np.fft.fft(sr) # Señal analogica con ruido, aplicada la fft
ft_Srq = 1/N*np.fft.fft(srq) # Señal analogica con ruido quantizada, aplicada la fft
ft_Nq = 1/N*np.fft.fft(nq) # piso de ruido digital
ft_Nn = 1/N*np.fft.fft(nn) # Piso de ruido Analogico

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2

Nnq_mean = np.mean(np.abs(ft_Nq)**2) # ??
nNn_mean = np.mean(np.abs(ft_Nn)**2) # ??

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' ) # Grafico de la señal senoidal analogica totalmente pura
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' ) # Grafico de la Señal analogica con ruido
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) ) # Piso de ruido analogico
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' ) # ?????'
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) ) # ????
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Respuesta en frecuencia')
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins,alpha=0.5)
plt.plot(np.array([-1/2, -1/2, 1/2, 1/2]),np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
plt.xlabel('Pasos de cuantización (q) [V]')
plt.show()

