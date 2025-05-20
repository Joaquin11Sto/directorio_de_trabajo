# -*- coding: utf-8 -*-
"""
Created on Wed May  7 20:12:53 2025

@author: Joaquin
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


from scipy import signal
from scipy.fft import fft, fftshift
import pandas as pd
rng = np.random.default_rng()

#Funciones

def blackman_tukey(x,  M = None):    
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;

fs_ecg = 1000 # Hz # Hz

# para listar las variables que hay en el archivo
ecg_sin_ruido = np.load('./ecg_sin_ruido.npy') # cargar el archivo . npy

N = len(ecg_sin_ruido) # Número de muestras
df = fs_ecg/N
ff_ecg = np.linspace(0, (N-1)*df, N) # Vector de frecuencias
nor_ecg = ecg_sin_ruido / np.std(ecg_sin_ruido) # normalizamos el ECG

fw, Pxxw = signal.welch(nor_ecg, fs_ecg, nfft=N, window='hamming', nperseg=(N/20), axis=0) # uso el metodo de Welch para calcular la PSD
Pxxbt_ecg = blackman_tukey(nor_ecg, M=int(N/20)) # uso el metodo de Blackman-Tukey para calcular la PSD

# Pxxw=20*np.log10(np.abs(Pxxw))
# plt.figure(1,figsize=(20, 5))
# plt.plot(ecg_sin_ruido, label='ECG sin ruido')
# plt.title('ECG sin ruido')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.grid()
# plt.legend()
# plt.show()

plt.figure(2,figsize=(20, 5))
plt.plot(fw,10*np.log10(np.abs(Pxxw)**2), label='Welch')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Densidad espectral de potencia del ECG sin ruido con el método de Welch')
plt.grid()
plt.legend()
plt.show()

plt.figure(3,figsize=(20, 5))
plt.plot(ff_ecg[:len(Pxxbt_ecg)//2], 10*np.log10(np.abs(Pxxbt_ecg[:len(Pxxbt_ecg)//2])**2), label='Blackman-Tukey')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Densidad espectral de potencia del ECG sin ruido con el método de Blackman-Tukey')
plt.grid()
plt.legend()
plt.show()

energia_total_bt_ecg=np.sum(Pxxbt_ecg[:len(Pxxbt_ecg)//2]) # La energia total del ECG espectro
energia_acumulada_bt_ecg=np.cumsum(Pxxbt_ecg[:len(Pxxbt_ecg)//2]) # La energia acumulada del ECG espectro
indx_95=np.where(energia_acumulada_bt_ecg>=0.95*energia_total_bt_ecg)[0][0] # El indice de la energia acumulada del ECG espectro
indx_98=np.where(energia_acumulada_bt_ecg>=0.98*energia_total_bt_ecg)[0][0] # El indice de la energia acumulada del ECG espectro
ancho_banda_95_ecg=ff_ecg[indx_95] # El ancho de banda del ECG espectro
ancho_banda_98_ecg=ff_ecg[indx_98] # El ancho de banda del ECG espectro

plt.figure(4,figsize=(20, 5))
plt.plot(ff_ecg[:len(Pxxbt_ecg)//2], 10*np.log10(np.abs(Pxxbt_ecg[:len(Pxxbt_ecg)//2])**2), label='Blackman-Tukey')
plt.axvline(x=ancho_banda_95_ecg, color='r', linestyle='--', label=f'Ancho de banda 95%: {ancho_banda_95_ecg:.2f} Hz')
plt.axvline(x=ancho_banda_98_ecg, color='g', linestyle='--', label=f'Ancho de banda 98%: {ancho_banda_98_ecg:.2f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Densidad espectral de potencia del ECG sin ruido con el método de Blackman-Tukey')
plt.grid()
plt.legend()
plt.show()