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

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
ecg_one_lead=ecg_one_lead/np.std(ecg_one_lead)
N=len(ecg_one_lead)
fs_ecg= 1000