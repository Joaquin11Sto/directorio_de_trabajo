import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Gráficos interactivos
#%matplotlib ipympl
# Gráficos estáticos
#%matplotlib inline

from pytc2.sistemas_lineales import plot_plantilla
import scipy.io as sio

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

# %%
# aporx_name = 'butter'
# fs=400
# nqy_frec=fs/2
# fpass=np.array( [0.5,10.0] ) #dB
# ripple = 1.0 #dB
# fstop = np.array( [.01,25.0] ) #dB
# attenuation = 50 #dB

# w_rad=np.append(np.logspace(-3,0.9,100),np.logspace(1,1.8,100))
# w_rad = np.append(w_rad, np.linspace(40,nqy_frec,200,endpoint=True))/nqy_frec * np.pi

# my_sos = sig.iirdesign(fpass,fstop,ripple,attenuation,fs=fs,ftype=aporx_name,output='sos')
# npoits= 400
# plt.figure(1)
# plt.cla()

# w,hh= sig.sosfreqz(my_sos, worN=w_rad)
# plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+ 1e-15), label='mi Sos')
# plt.title('Plantilla diseñada')
# plt.xlabel('Frecuencia normalizada a Nyq [#]')
# plt.ylabel('Amplitud [dB]')
# plt.grid(which='both', axis='both')
# plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
# plt.legend()
# plt.show()

# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

# y_ppg = sig.sosfiltfilt(my_sos, ppg,axis=0)
# ppg_sin=np.load('ppg_sin_ruido.npy')

# plt.figure(2,figsize=(20, 6))
# plt.plot(ppg, label='PPG original',color='orange')
# plt.plot(y_ppg, label='PPG filtrado',color='blue')
# # plt.plot(ppg_sin,label='PPG Sin ruido Mariano',color = 'green')
# plt.title('PPG')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.grid(which='both', axis='both')
# plt.legend()
# plt.show()

# %%
aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

# Por qué no hay bessel ?
#aprox_name = 'bessel'

# Requerimientos de plantilla

# plantillas normalizadas a Nyquist y en dB
fs = 1000 # Frecuencia de muestreo en Hz
nyq_frec = fs/2
fpass = np.array( [1.0, 35.0] ) 
ripple = 1.0 # dB
fstop = np.array( [0.1, 50.0] ) 
attenuation = 40.0 # dB

w_rad = np.append(np.logspace(-2,0.8,250), np.logspace(0.9,1.6,250))
w_rad = np.append(w_rad, np.linspace(40,nyq_frec,500,endpoint=True))/nyq_frec * np.pi

my_sos= sig.iirdesign(fpass,fstop,ripple,attenuation,fs=fs,ftype=aprox_name,output='sos')

# Plantilla de diseño
plt.figure(1)
plt.cla()


npoints = 1000
w,hh = sig.sosfreqz(my_sos, worN=w_rad)
hh_ph = np.angle(hh) # Respuesta de fase
hh_ph= np.unwrap(hh_ph)
der_ph = np.diff(hh_ph)
plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+ 1e-15), label='mi Sos') # Respuesta de Modulo
# plt.plot(w/np.pi*fs/2,hh_ph)
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')
plt.legend()

plot_plantilla(filter_type = 'bandpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)



mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N= len(ecg_one_lead)

y_ecg = sig.sosfiltfilt(my_sos, ecg_one_lead,axis=0)


# plt.figure(2,figsize=(20, 6))
# plt.plot(ecg_one_lead, label='ECG original',color='yellow')
# plt.plot(y_ecg, label='ECG filtrado',color='')
# plt.title('ECG original')
# plt.xlabel('Muestras')
# plt.ylabel('Amplitud')
# plt.grid(which='both', axis='both')
# plt.legend()

# %%


regs_interes = ( 
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
demora = 0
for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(10, 6), dpi= 150, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    # plt.plot(zoom_region, y_ecg[zoom_region], label='Butter')
    plt.plot(zoom_region, y_ecg[zoom_region + demora], label='Win')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.grid()
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()

# %%


regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
demora= 0
for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(12, 6), dpi= 150, facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2,color='blue')
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, y_ecg[zoom_region + demora], label='Win',color='yellow')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.grid()
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()