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

mat_struct = sio.loadmat('C:/Users/Joaquin/Carpeta_APS/ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead)
N=len(ecg_one_lead)
fs_ecg= 1000
f_nyq=fs_ecg/2

# %% Filtro Highpass

coef_hp= 9501 #Tiene que ser impar
coef_hp_firls= 4001
coef_hp_remez = 2101

fpass_hp= 1
fstop_hp= 0.1
attenuation= 40 #dB
ripple = 0.1 #dB

#Requisitos de plantilla

frecs=[0, fstop_hp/f_nyq, fpass_hp/f_nyq, 1]
gains=[-np.inf,-attenuation,-ripple,0] #dB
gains = 10**(np.array(gains)/20)

num_hp = sig.firwin2(coef_hp,frecs,gains,window=('kaiser',14)) # Metodo de ventana
firls_hp = sig.firls(coef_hp_firls,frecs,gains,fs=2) # Metodo de cuadrados Minimos
remez_hp= sig.remez(coef_hp_remez, frecs,gains[::2], fs = 2) # Metodo Parks-MC claeran


w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad, np.linspace(40,f_nyq,500,endpoint=True))/f_nyq * np.pi

# %% Grafico la plantilla de ventana highpass
# Grafico de mi plantilla
plt.figure(1)
plt.cla()

w , hh = sig.freqz(num_hp,1,worN=w_rad)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'highpass' , fpass = fpass_hp, ripple = 1 , fstop = fstop_hp, attenuation = attenuation, fs = fs_ecg)
plt.legend()

# %% Grafico de plantilla de cuadrados minimos Highpass
plt.figure(2)
plt.cla()

w , hh = sig.freqz(firls_hp,1,worN=w_rad)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'highpass' , fpass = fpass_hp, ripple = 1 , fstop = fstop_hp, attenuation = attenuation, fs = fs_ecg)
plt.legend()
plt.show()

# %% Grafico de plantilla de Parks Highpass

plt.figure(3)
plt.cla()

w , hh = sig.freqz(remez_hp,1,worN=w_rad)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'highpass' , fpass = fpass_hp, ripple = 1 , fstop = fstop_hp, attenuation = attenuation, fs = fs_ecg)
plt.legend()
plt.show()

# %% Filtro Lowpass
coef_lp=4001
coef_lp_firls= 351
coef_lp_remez= 147

fpass_lp= 35
fstop_lp= 49
attenuation= 40 #dB
ripple = 0.5 #dB
#Requisitos de plantilla
frecs_lp=[0, fpass_lp/f_nyq, fstop_lp/f_nyq, 1]
gains_lp=[0,-ripple, -attenuation,-np.inf] #dB
gains_lp = 10**(np.array(gains_lp)/20)

win_name= 'kaiser'
num_lp = sig.firwin2(coef_lp,frecs_lp,gains_lp,window=('kaiser',14))
firls_lp = sig.firls(coef_lp_firls,frecs_lp,gains_lp,fs=2)
remez_lp = sig.remez(coef_lp_remez,frecs_lp,gains_lp[::3],fs=2)

# %% Grafico mi filtro pasa bajos Ventanas
# Grafico de mi plantilla
plt.figure(4)
plt.cla()

w , hh = sig.freqz(num_lp,1,worN=1000)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'lowpass' , fpass = fpass_lp, ripple = 1 , fstop = 50, attenuation = attenuation, fs = fs_ecg)
plt.legend()
# %% Grafico de mi filtro pasabajo Cuadrados minimos
plt.figure(5)
plt.cla()

w , hh = sig.freqz(firls_lp,1,worN=1000)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'lowpass' , fpass = fpass_lp, ripple = 1 , fstop = 50, attenuation = attenuation, fs = fs_ecg)
plt.legend()

# %% Grafico del filtro pasabajos de Parks Mc cleran
plt.figure(6)
plt.cla()

w , hh = sig.freqz(remez_lp,1,worN=1000)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'lowpass' , fpass = fpass_lp, ripple = 1 , fstop = 50, attenuation = attenuation, fs = fs_ecg)
plt.legend()

# %% Grafico de pasabanda Metodo de Ventanas
num_bp = np.convolve(num_hp, num_lp)

fpass_bp = np.array([fpass_hp , fpass_lp]) 
fstop_bp = np.array([fstop_hp , fstop_lp])
plt.figure(7)
plt.cla()

w , hh = sig.freqz(num_bp,1,worN=w_rad)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'bandpass' , fpass = fpass_bp, ripple = 1 , fstop = fstop_bp, attenuation = attenuation, fs = fs_ecg)
plt.legend()

# %% Grafico bandpass Metodo Cuadrados Minimos
firls_bp = np.convolve(firls_hp, firls_lp)

fpass_bp = np.array([fpass_hp , fpass_lp]) 
fstop_bp = np.array([fstop_hp , fstop_lp])
plt.figure(8)
plt.cla()


w , hh = sig.freqz(firls_bp,1,worN=w_rad)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'bandpass' , fpass = fpass_bp, ripple = 1 , fstop = fstop_bp, attenuation = attenuation, fs = fs_ecg)
plt.legend()

# %% Grafico bandpass Metodo Parks- Mc cleran
remez_bp = np.convolve(remez_hp, remez_lp)

fpass_bp = np.array([fpass_hp , fpass_lp]) 
fstop_bp = np.array([fstop_hp , fstop_lp])
plt.figure(9)
plt.cla()


w , hh = sig.freqz(remez_bp,1,worN=w_rad)

plt.plot(w/np.pi*f_nyq, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'bandpass' , fpass = fpass_bp, ripple = 1 , fstop = fstop_bp, attenuation = attenuation, fs = fs_ecg)
plt.legend()
# %% filtrado de señal

#Filtro primero con el highpass
f_ecg_win=sig.lfilter(num_bp,1,ecg_one_lead,axis=0) # FIR con Metodo de Ventanas
f_ecg_firls=sig.lfilter(firls_bp,1,ecg_one_lead,axis=0) # FIR con Metodo de Cuadrados Minimos
f_ecg_remez=sig.lfilter(remez_bp,1,ecg_one_lead,axis=0) # FIR con Metodo de Parks-McCleran

demora = (len(num_bp) - 1) // 2

# %%
plt.plot(ecg_one_lead,label= 'ECG')
plt.plot(f_ecg_win,label='Filtrado')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid()
plt.legend()
plt.show()

# %%

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
    plt.plot(zoom_region, f_ecg_win[zoom_region + demora], label='FIR Window')
   
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
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
   
    plt.figure(2)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()

