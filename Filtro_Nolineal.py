# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:52:42 2025

@author: Joaquin
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
from scipy.signal.windows import hamming, kaiser, blackmanharris
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline

from pytc2.sistemas_lineales import plot_plantilla, group_delay

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

mat_struct = sio.loadmat('C:/Users/Joaquin/Carpeta_APS/ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
qrs_picks = vertical_flaten(mat_struct['qrs_detections'])
qrs_pattern1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
qrs_pattern2 = vertical_flaten(mat_struct['heartbeat_pattern2'])
qrs_pattern3= vertical_flaten(mat_struct['qrs_pattern1'])

ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead) # normalizacion
qrs_pattern3=qrs_pattern3/np.std(qrs_pattern3)
N=len(ecg_one_lead)
fs_ecg= 1000
tt = np.linspace(0, (N-1)/fs_ecg, N, endpoint=False)

# %% Aplico tecnica de la mediana

b_gorro=sig.medfilt(ecg_one_lead[:,0],kernel_size=201)
b_gorro = sig.medfilt(b_gorro,kernel_size=601)
# %% Grafico
plt.figure(1)
plt.plot(ecg_one_lead,label='ECG')
plt.plot(b_gorro,label='Linea de Base',color='orange')
plt.title('ECG con Mediana')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# %% graficar el ECG con los picos

plt.figure(2)
plt.plot(ecg_one_lead[:, 0],color='green')
plt.plot(qrs_picks[:, 0], ecg_one_lead[qrs_picks[:, 0].astype(int), 0], 'rx', label='QRS detectados')
# plt.plot(ecg_V[:,0:],color ='blue')
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# %% Aplico el metodo de splines cubicos
qrs_indices = qrs_picks[:, 0].astype(int)

# Inicializamos listas para guardar los puntos de spline
tiempos_spline = []
valores_spline = []

# Recorremos cada QRS
for idx in qrs_indices:
    inicio = idx - 90
    fin = inicio + 20

    # Control de bordes por si la ventana queda fuera del inicio de la señal
    if inicio < 0 or fin > N:
        continue  # Salteamos si la ventana es inválida

    ventana = ecg_one_lead[inicio:fin, 0]
    promedio = np.mean(ventana)

    tiempos_spline.append(inicio)
    valores_spline.append(promedio)

cs = CubicSpline(tiempos_spline, valores_spline)

# Generamos la línea de base estimada para toda la señal
tiempos = np.arange(N)
linea_base_spline = cs(tiempos)


# %% Grafico de la interpolacion 
plt.figure(4)
plt.plot(ecg_one_lead[:, 0], label='ECG original', color='green')
plt.plot(tiempos, linea_base_spline, label='Línea de Base (Spline)', color='orange')
plt.scatter(tiempos_spline, valores_spline, color='red', label='Puntos para el spline')
plt.title('Línea de Base con Splines')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# %% Aplico tecnica del filtro matched

corre_ecg = sig.correlate(ecg_one_lead,qrs_pattern3,mode='same')
corre_ecg = corre_ecg / np.std(corre_ecg)
peaks, _ = sig.find_peaks(corre_ecg[:,0],height= 0.2,distance=350)

# %% Grafico las realizaciones
muestras = np.arange(N)

plt.figure(5)
plt.plot(muestras, ecg_one_lead, label='ECG')
plt.plot(muestras, corre_ecg, label='ECG correlado')
plt.plot(peaks, corre_ecg[peaks], 'rx', label='Picos correlación')
plt.plot(qrs_indices, ecg_one_lead[qrs_indices], 'gx', label='Picos originales')
plt.title('Matriz de realizaciones (en muestras)')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# %% Matriz de confusion
# Tolerancia en muestras (ajustar según tu fs)
tolerancia = 100

# Convertimos los arrays en sets para facilitar el conteo
qrs_real = np.array(qrs_indices)
qrs_detectados = np.array(peaks)

# Inicializamos contadores
VP = 0
FP = 0
FN = 0

# Creamos una máscara para marcar qué QRS reales fueron detectados
qrs_detectados_mask = np.zeros(len(qrs_real), dtype=bool)

# Contamos Verdaderos Positivos (VP)
for detected_peak in qrs_detectados:
    # Buscamos el QRS real más cercano
    distancia = np.abs(qrs_real - detected_peak)
    idx_min = np.argmin(distancia)
    
    if distancia[idx_min] <= tolerancia:
        # Si cae dentro de la tolerancia, contamos como VP
        VP += 1
        qrs_detectados_mask[idx_min] = True
    else:
        # Si no coincide con ningún QRS real, es un FP
        FP += 1

# Los FN son los QRS reales que no fueron detectados
FN = np.sum(~qrs_detectados_mask)

# Imprimir tabla tipo matriz de confusión
print("\nMatriz de Confusión:")
print("----------------------------")
print("                | QRS real Sí | QRS real No")
print("Detectado Sí    | {:>10}  | {:>10}".format(VP, FP))
print("Detectado No    | {:>10}  | {:>10}".format(FN, "-"))