# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:51:42 2025

@author: Joaquin
"""
import numpy as np
from scipy import signal as sig
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.signal import stft, welch
import pywt


from scipy import signal
from scipy.fft import fft, fftshift
import pandas as pd
rng = np.random.default_rng()

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
N = len(wav_data) # Número de muestras
nor_audio = wav_data / np.std(wav_data) # normalizamos el audio
fw_audio, Pxxw_audio = signal.welch(nor_audio, fs_audio, nfft=N, window='blackman', nperseg=(N//8), axis=0) # uso el metodo de Welch para calcular la PSD
t = np.linspace(0, N, int(N), endpoint=False)
# %% Aplicacion de la cucaracha en SFTF


f, t_stft, Zxx = stft(wav_data, fs=fs_audio, nperseg=1024)

# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

# Señal
ax1.plot(wav_data[0:17000])
ax1.set_title("Titulo")
ax1.set_ylabel("Amplitud")
# ax1.set_xlim(t[0], t[-1])

# Subplot 2: Welch
ax2.semilogy(fw_audio//2, Pxxw_audio)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")

# Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylim(0,3000)
ax3.set_xlim(0,0.36)


# Colorbar en eje externo
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

# %% Aplicacion de la cucaracha en CWT
# Escalas y CWT
scales = np.logspace(0, np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')
# wavelet = pywt.ContinuousWavelet('mexh')
# wavelet = pywt.ContinuousWavelet('gaus3')

f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
dt = 1.0 / fs_audio
frequencies = f_c / (scales * dt)

coefficients, frec = pywt.cwt(wav_data, scales, wavelet, sampling_period=dt)
t = np.linspace(0, (N-1)/fs_audio, N, endpoint=False)

# %% Grafico de la cucaracha en CWT

# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Señal
ax1.plot(t,wav_data)
ax1.set_title("Señal")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(t[0], t[-1])
plt
pcm = ax2.imshow(np.abs(coefficients),
           extent=[t[0], t[-1], scales[-1], scales[0]],  # nota el orden invertido para eje Y
           cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()
# %% Aplicaicon de PPg en STFT

fs_ppg= 400 # Hz
ppg = np.load('ppg_sin_ruido.npy')
N = len(ppg) # Número de muestras
nor_ppg = ppg / np.std(ppg) # normalizamos el PPG
fw_ppg, Pxxw_ppg = signal.welch(nor_ppg, fs_ppg, nfft=N, window='blackman', nperseg=(N//8), axis=0) # uso el metodo de Welch para calcular la PSD

f, t_stft, Zxx = stft(nor_ppg, fs=fs_ppg, nperseg=548)
t = np.linspace(0, (N-1)/fs_ppg, N, endpoint=False)


# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

# Señal
ax1.plot(t,nor_ppg)
ax1.set_title("Titulo")
ax1.set_ylabel("Amplitud")
# ax1.set_xlim(t[0], t[-1])

# Subplot 2: Welch
ax2.semilogy(fw_ppg, Pxxw_ppg)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")

# Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylim(0,10)
# ax3.set_xlim(0,0.36)


# Colorbar en eje externo
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

# %% DWT Aplicacion a la cucaracha (Audio)

# DWT multiescala hasta nivel 4
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(wav_data), pywt.Wavelet(wavelet).dec_len)

coeffs = pywt.wavedec(wav_data, wavelet, level=8)

# Separar coeficientes
cA8,cD8,cD7,cD6 ,cD5,cD4, cD3, cD2, cD1 = coeffs
t = np.linspace(0, (N-1)/fs_audio, N, endpoint=False)

plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(t, wav_data)
plt.title("Audio original")
plt.grid()

plt.subplot(6, 1, 2)
plt.plot(cD5)
plt.title("Coeficiente de detalle (nivel 5)")

plt.subplot(6, 1, 3)
plt.plot(cD6)
plt.title("Coeficiente de detalle (nivel 6)")

plt.subplot(6, 1, 4)
plt.plot(cD7)
plt.title("Coeficiente de detalle (nivel 7)")

plt.subplot(6, 1, 5)
plt.plot(cD8)
plt.title("Coeficiente de detalle (nivel 8)")

plt.subplot(6, 1, 6)
plt.plot(cA8)
plt.title("Coeficiente de aproximación (nivel 8)")

plt.tight_layout()
plt.show()
