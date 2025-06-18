# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:23:52 2025

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
altos=vertical_flaten(mat_struct['qrs_detections'])
hb1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb2 = vertical_flaten(mat_struct['heartbeat_pattern2'])
qrs = vertical_flaten(mat_struct['qrs_pattern1'])
qrs_win= mat_struct['qrs_detections']

N=len(ecg_one_lead)
N_qrs=len(altos)
ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead) # Normalizacion
ecg_corto=ecg_one_lead[720000:740000]
b_gorro=sig.medfilt(ecg_corto[:,0],kernel_size=201) ## metodo de mediana

b_gorro=sig.medfilt(b_gorro,kernel_size=601)


# ecg_filtro = ecg_corto - b_gorro
# cant_ciclos=N/N_qrs + 0.5
# corr = sig.correlate(ecg_one_lead, qrs)
# corr = corr /np.std(corr)

# Definir la cantidad de muestras antes y después del QRS
samples_before = 200
samples_after = 350

# Dimensiones del bloque (ventana)
# window_size = samples_before + samples_after

# Lista para almacenar las ventanas
# ecg_blocks = []

# for i in range(len(altos)):
#     idx = altos[i, 0]  # obtener el valor del índice
#     start = idx - samples_before
#     end = idx + samples_after

#     # Verificar que no se exceda de los límites
#     if start >= 0 and end < len(ecg_one_lead):
#         window = ecg_one_lead[start:end, 0]  # sacar como vector plano
#         ecg_blocks.append(window)

# # Convertir la lista a array: shape (num_latidos, window_size)
# ecg_V = np.array(ecg_blocks)



# for i in 0 : N_qrs
    # ecg_V =  [ecg_one_lead[altos[i]+300]]
# %%
# %%


# plt.figure(1)
# plt.plot(ecg_filtro,color='blue',label='ECG Filtrado')
# plt.plot(ecg_corto,color='yellow',label='ECG')
# plt.title('Señal ECG')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.show()

# %%
plt.figure(2)
# plt.plot(ecg_one_lead[:, 0],color='green')
# plt.plot(altos[:, 0], ecg_one_lead[altos[:, 0].astype(int), 0], 'rx', label='QRS detectados')
plt.plot(ecg_V[:,0:],color ='blue')
plt.title('Señal ECG')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()


# %%

# Arcoiris de realizaciones
plt.figure(3)
plt.plot(ecg_V)
plt.title("Primeras 10 realizaciones de ECG centradas en QRS")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

