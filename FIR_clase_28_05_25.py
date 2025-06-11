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

# %%

# frecuencia de muestreo normalizada
fs = 1000
# tamaño de la respuesta al impulso
cant_coef = 2500
nyq_frec=fs/2

filter_type = 'bandpass'


fpass =np.array([1.0,35.0]) # 
ripple = 1.0 # dB
fstop =np.array([0.1,50.0]) # Hz
attenuation = 50.0 # dB

# construyo la plantilla de requerimientos
frecs = [0.0,  0.1, 1.0,35.0,50.0, 500]
gains = [0,0,1,1,0,0] # dB

# gains = 10**(np.array(gains)/20) #Aca lo pasa a veces

# algunas ventanas para evaluar
#win_name = 'boxcar'
#win_name = 
win_name = kaiser
#win_name = 'flattop'

# FIR design
num_bh = sig.firwin2(cant_coef, frecs, gains , window='blackmanharris',fs=fs )
num_hm = sig.firwin2(cant_coef, frecs, gains , window='hamming',fs=fs )
num_ka = sig.firwin2(cant_coef, frecs, gains , window=('kaiser',14),fs=fs)
den = 1.0
# %%
# def plot_freq_resp_fir(this_num, this_desc):

#     wrad, hh = sig.freqz(this_num, 1.0)
#     ww = wrad / np.pi
    
#     plt.figure(1)

#     plt.plot(ww, 20 * np.log10(abs(hh)), label=this_desc)

#     plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef) )
#     plt.xlabel('Frequencia normalizada')
#     plt.ylabel('Modulo [dB]')
#     plt.grid(which='both', axis='both')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()
    
#     plt.figure(2)

#     phase = np.unwrap(np.angle(hh))

#     plt.plot(ww, phase, label=this_desc)

#     plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef))
#     plt.xlabel('Frequencia normalizada')
#     plt.ylabel('Fase [rad]')
#     plt.grid(which='both', axis='both')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()

#     plt.figure(3)

#     # ojo al escalar Omega y luego calcular la derivada.
#     gd_win = group_delay(wrad, phase)

#     plt.plot(ww, gd_win, label=this_desc)

#     plt.ylim((np.min(gd_win[2:-2])-1, np.max(gd_win[2:-2])+1))
#     plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef))
#     plt.xlabel('Frequencia normalizada')
#     plt.ylabel('Retardo [# muestras]')
#     plt.grid(which='both', axis='both')

#     axes_hdl = plt.gca()
#     axes_hdl.legend()

# plot_freq_resp_fir(num_bh, filter_type+ '-blackmanharris')    
# plot_freq_resp_fir(num_hm, filter_type+ '-hamming')    
# plot_freq_resp_fir(num_ka, filter_type+ '-kaiser-b14')    
    
    
# # sobreimprimimos la plantilla del filtro requerido para mejorar la visualización    
# fig = plt.figure(1)    
# plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
# ax = plt.gca()
# ax.legend()

# # reordenamos las figuras en el orden habitual: módulo-fase-retardo
# plt.figure(2)    
# axes_hdl = plt.gca()
# axes_hdl.legend()

# plt.figure(3)    
# axes_hdl = plt.gca()
# axes_hdl.legend()

# plt.show()
# %%
w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad, np.linspace(40,nyq_frec,500,endpoint=True))/nyq_frec * np.pi

plt.figure(1)
plt.cla()

npoints = 1000
a0=1
w,hh = sig.freqz(num_ka,a0, worN=w_rad)
plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+ 1e-15)) # Respuesta de Modulo
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
# plt.legend()

# %%


