# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:58:30 2025

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

mat_struct = sio.loadmat('C:/Users/Joaquin/Carpeta_APS/ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead)
N=len(ecg_one_lead)
fs_ecg= 1000
f_nyq=fs_ecg/2

# Bandpass
fpass = np.array( [1, 35] ) 
ripple = 1.0 # dB
fstop = np.array( [0.1, 50] ) 
attenuation = 40.0 # dB

my_sos_1 = sig.iirdesign(fpass, fstop, ripple, attenuation,fs=fs_ecg,ftype='cheby2',output='sos')
my_sos_2 = sig.iirdesign(fpass, fstop, ripple, attenuation,fs=fs_ecg,ftype='ellip',output='sos')

w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad, np.linspace(40,f_nyq,500,endpoint=True))/f_nyq * np.pi
# %% Grafico de diseño de cheby2

w , hh = sig.sosfreqz(my_sos_1,worN=w_rad)

plt.figure(1)
plt.cla()

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_ecg)
plt.legend()
# %% Grafico de diseño ellip

w , hh = sig.sosfreqz(my_sos_2,worN=w_rad)

plt.figure(1)
plt.cla()

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs_ecg)
plt.legend()

# %% 

y_ecg = sig.sosfiltfilt(my_sos_1, ecg_one_lead,axis=0)
demora = 0

plt.figure(8)
plt.plot(ecg_one_lead, label='ECG')
plt.plot(y_ecg, label='ECG filtrado',color='green')
plt.title('ECG')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

###################################
#%% Regiones de interés con ruido #
###################################
 
regs_interes = (
        [4000, 5500], # muestras
        # [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure(1)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, y_ecg[zoom_region + demora], label='Filtrado')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
###################################
#%% Regiones de interés sin ruido #
###################################
 
regs_interes = (
        # np.array([5, 5.2]) *60*fs_ecg, # minutos a muestras
        # np.array([12, 12.4]) *60*fs_ecg, # minutos a muestras
        np.array([15, 15.2]) *60*fs_ecg, # minutos a muestras
        )

for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure(2)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, y_ecg[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()