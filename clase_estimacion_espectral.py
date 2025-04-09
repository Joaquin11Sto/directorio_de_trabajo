# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:45:45 2025

@author: Notebook
"""
import numpy as np
import matplotlib.pyplot as plt
# %% CLASE 3/4/2025
####### DATO SIMULACION ##############

repes =200
a1= 1/np.sqrt(2) #para q sea normal
N=1000
fs=1000
ts=1/fs
df = fs/N
# %%
########################
##### Mtriz de ruido####
########################

SNR= 10
Pot_nn =10**(-SNR / 10) #Viene del calculo de gretissss
verifico = - 10 * np.log10(Pot_nn)
print(verifico)
N = 1000
sigma = np.sqrt(Pot_nn)

nn = np.random.normal(0, sigma, (N,repes)) #matriz de 1000

# %%
##### Matriz de omegaS####

ff = np.array([])

for i in range(repes):
    fr = np.random.uniform(-0.5,0.5) 
    f0= fs/4
    f1= f0 + fr * df
    ff = np.append(ff,f1)
    
ff = ff.reshape((1,repes))

# %%
####primer termino del xx

t = np.linspace(0,(N-1)*ts,N).reshape((N,1))

xx=a1 * np.sin(2*np.pi*t*ff)
    
# %%
### sumo todo :)
xk= xx + nn

# %%
### ploteo :) 
# for i in range(10):
#    plt.plot(t, xk[:,i])
   
# %% ### FFT
Xk_fft = np.fft.fft(xk, axis=0) /N  # FFT por columnas
fs = 1000  # frecuencia de muestreo
ffx = np.linspace(0, (N-1)*df, N)
bfrec = ffx <= fs/2

plt.figure()
for i in range(repes):
    plt.plot(ffx[bfrec], 10 * np.log10(2 * np.abs(Xk_fft[bfrec, i])**2), label=f'Señal {i+1}')

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad espectral de potencia (dB)")
plt.legend()
plt.grid(True)
plt.show()

# Falta ver q onda si multiplico xk por una window (ahora esta la delta implicita)
# Lo ultimo q djo es q si de  xk_fft(1000x200) tomo la fila donde N/4 (OSEA EL PICO)
# obtengo ccada "a" estimado de cada repe. entonces con las 200(a piquito) se puede
#calcular el sesgo y la varianza :) 
# 
