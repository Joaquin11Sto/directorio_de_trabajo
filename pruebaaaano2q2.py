# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:18:16 2025

@author: Joaquin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
from scipy import signal
from scipy.fft import fft, fftshift


fs=1000
N=1000
window_1 = signal.windows.boxcar(51)


