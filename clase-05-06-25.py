# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:29:37 2025

@author: Joaquin
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)


fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')

# plt.figure()
# plt.plot(wav_data)
X=wav_data
# %% Interpolacion
N=len(wav_data)
L=10
Z=np.zeros(N*L)
Z[::L]= X
fs_x10=fs_audio*10

f_nyq= fs_x10/2


fpass = 0.1*f_nyq # 
ripple = 0.5 # dB
fstop = 0.6*f_nyq # Hz
attenuation = 40 # dB

# %%