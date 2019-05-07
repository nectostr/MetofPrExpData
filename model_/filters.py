import datetime
import numpy as np
import math
import in_out
from in_out import *
from analysys import *
from numba import jit
import copy
import collections

@jit()
def create_low_pass_filter(fc, m, dt):
    d = [0.35577019, 0.24369830, 0.07211497, 0.00630165]
    lpf = np.zeros(m+1)
    arg = 2 * fc * dt
    lpf[0] = arg
    arg *= np.pi
    for i in range(1,m+1):
        lpf[i] = np.sin(arg*i)/(np.pi*i)
    lpf[m] /= 2
    sumg = lpf[0]
    for i in range(1, m+1):
        sum = d[0]
        arg = np.pi * i / m
        for k in range(1,4):
            sum += 2 * d[k] * np.cos(arg * k)
        lpf[i] *= sum
        sumg += 2 * lpf[i]
    lpf /= sumg
    lpf = np.append(lpf[::-1], lpf[1:])
    #lpf *= len(lpf)
    return lpf

@jit()
def create_hight_pass_filter(fc, m, dt):
    lpf = create_low_pass_filter(fc, m, dt)
    lpw = -lpf.copy()
    lpw[m] = 1 - lpf[m]
    return lpw

@jit()
def create_band_pass_filter(fc1, fc2, m, dt):
    lpf1 = create_low_pass_filter(fc1, m, dt)
    lpf2 = create_low_pass_filter(fc2, m, dt)
    bpw = lpf2 - lpf1
    return bpw

@jit()
def create_band_stop_filter(fc1, fc2, m, dt):
    lpf1 = create_low_pass_filter(fc1, m, dt)
    lpf2 = create_low_pass_filter(fc2, m, dt)
    bsw = lpf1 - lpf2
    bsw[m] = 1 + lpf1[m] - lpf2[m]
    return bsw

@jit()
def convolutional(x, h):
    M = len(h)
    N = len(x)
    y = np.zeros((N+M))
    for k in range(0, N + M):
        for m in range(0, M):
            y[k] += x[(k-m)%N] * h[m]
    return y[M//2:-M//2]

@jit()
def convolutional_2d(_x, h):
    x = _x.copy()
    for i in range(x.shape[0]):
        x[i] = convolutional(x[i], h)
    for i in range(x.shape[1]):
        x[:,i] = convolutional(x[:,i], h)
    return x

def threshold_filter_2d(_x, threshold, mul=1):
    _x = np.array(_x > threshold, dtype=np.float)
    # for i in range(_x.shape[0]):
    #     for j in range(_x.shape[1]):
    #         _x[i,j] = 1 if _x[i,j] > threshold else 0
    return _x * mul