# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:32:17 2025

@author: Joaquin
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import hamming, kaiser, blackmanharris
import matplotlib.pyplot as plt
import scipy.io as sio

from pytc2.sistemas_lineales import plot_plantilla, group_delay

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N=len(ecg_one_lead)
fs_ecg= 1000
# %% Filtro Highpass

fs_hp= 2
f_nyq_hp=fs_hp/2
coef_hp= 5501 #Tiene que ser impar
filter_type_hp= 'highpass'

fpass_hp= 0.1
fstop_hp= 0.001
attenuation= 40 #dB
ripple = 0.5 #dB

#Requisitos de plantilla

frecs=[0, fstop_hp, fpass_hp, 1]
gains=[-np.inf,-ripple,-attenuation,0] #dB
gains = 10**(np.array(gains)/20)

win_name= 'kaiser'
num_hp = sig.firwin2(coef_hp,frecs,gains,window=('kaiser',14))

# Grafico de mi plantilla
plt.figure(1)
plt.cla()

# w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
# w_rad = np.append(w_rad, np.linspace(0.4,f_nyq_hp,500,endpoint=True))/f_nyq_hp * np.pi
w , hh = sig.freqz(num_hp,1,worN=1000)

plt.plot(w/np.pi, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'highpass' , fpass = fpass_hp, ripple = ripple , fstop = fstop_hp, attenuation = attenuation, fs = fs_hp)
plt.legend()
# %% Filtro Lowpass
fs_lp=2
f_nyq_lp=fs_lp/2
coef_lp=9001
fpass_lp= 0.35
fstop_lp= 0.50
attenuation= 40 #dB
ripple = 0.5 #dB
#Requisitos de plantilla
frecs_lp=[0, fstop_lp, fpass_lp, 1]
gains_lp=[0,-ripple, -attenuation,-np.inf] #dB
gains_lp = 10**(np.array(gains)/20)

win_name= 'kaiser'
num_lp = sig.firwin2(coef_lp,frecs_lp,gains_lp,window=('kaiser',14))

# Grafico de mi plantilla
plt.figure(2)
plt.cla()

# w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
# w_rad = np.append(w_rad, np.linspace(0.4,f_nyq_hp,500,endpoint=True))/f_nyq_hp * np.pi
w , hh = sig.freqz(num_lp,1,worN=1000)

plt.plot(w/np.pi, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'lowpass' , fpass = fpass_lp, ripple = ripple , fstop = fstop_lp, attenuation = attenuation, fs = fs_hp)
plt.legend()
