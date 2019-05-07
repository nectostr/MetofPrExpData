import datetime
import numpy as np
import math
import in_out
from in_out import *
from analysys import *
from numba import jit
import copy
import collections

@jit(parallel=True)
def _fourier_step(x,N,n):
    k = np.arange(0,N)
    Re = np.sum(x * np.cos((2 * np.pi * n * k) / N), dtype=np.double)
    Im = np.sum(x * np.sin((2 * np.pi * n * k) / N), dtype=np.double)
    return Re, Im

@jit(parallel=True)
def _fourier_step_complex(y, N, n, rev=False):
    k = np.arange(0, N)
    if not rev:
        return np.sum(y * np.exp(-1j * 2 * math.pi * k * n / N))
    else:
        return np.sum(y * np.exp(1j * 2 * math.pi * k * n / N))


@jit
def fourier_complex(y):
    N = len(y)
    x = np.zeros(N, dtype=np.complex)
    for n in range(0, N):
       x[n] = _fourier_step_complex(y, N, n)
    x /= N
    return x

@jit
def fourier(xk, ret_phaze=False): #furie
    N = len(xk)
    Re = np.zeros(N, dtype=np.double)
    Im = np.zeros(N, dtype=np.double)
    for n in range(N):
        Re[n], Im[n] = _fourier_step(xk,N,n)
    Re /= N
    Im /= N
    Cn  = (Re**2+Im**2)**(1/2)
    CSn = Re+Im
    if not ret_phaze:
        return Cn, CSn
    else:
        return Cn, CSn, Im, Re


@jit
def fourier_2D(img, test=1, complex=False):
    if not complex:
        img_f = np.empty(img.shape)
        for i in range(img.shape[0]):
            img_f[i] = fourier(img[i])[test]

        for j in range(img_f.shape[1]):
            img_f[:, j] = fourier(img_f[:, j])[test]
    else:
        img_f = np.empty(img.shape, dtype=np.complex)
        for i in range(img.shape[0]):
            img_f[i] = fourier_complex(img[i])

        for j in range(img_f.shape[1]):
            img_f[:, j] = fourier_complex(img_f[:, j])
    return img_f


@jit
def fourier_2D_back(img, complex=False):
    if not complex:
        for j in range(img.shape[0]):
            img[j] = reverse_fourier(img[j])
        for i in range(img.shape[1]):
            img[:, i] = reverse_fourier(img[:, i])
    else:
        for j in range(img.shape[0]):
            img[j] = reverse_fourier_complex(img[j])
        for i in range(img.shape[1]):
            img[:, i] = reverse_fourier_complex(img[:, i])
    return img


@jit
def reverse_fourier_complex(CSn): #back furie
    N = len(CSn)
    x = np.zeros(N, dtype=np.complex)
    for n in range(0, N):
       x[n] = _fourier_step_complex(CSn, N, n, rev=True)
    x /= N
    return x

@jit
def reverse_fourier(CSn): #back furie
    N = len(CSn)
    Re = np.zeros(N)
    Im = np.zeros(N)
    for k in range(N):
        Re[k], Im[k] = _fourier_step(CSn, N, k)
    xk = Re + Im
    return xk