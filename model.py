import datetime
import numpy as np
import math
import in_out
from model import *
from in_out import *
from analysys import *
from numba import jit


def add_shift(x, shi):
    return x + shi

def generate_x(fromm, to, step):
    x = np.empty(int((to - fromm) / step))
    for i in range(len(x)):
        x[i] = fromm + step * i
    return x

def my_harmonic(dt=0.002, N=1000, A0 = 100, f0 = 37):
    k = generate_x(0, N, 1)
    y = A0 * np.sin(2*np.pi*f0*k*dt)
    return k,y

@jit(parallel=True)
def _fourier_step(x,N,n):
    k = np.arange(0,N)
    Re = np.sum(x * np.cos((2 * np.pi * n * k) / N))
    Im = np.sum(x * np.sin((2 * np.pi * n * k) / N))
    return Re, Im

@jit
def fourier(xk): #furie
    N = len(xk)
    Re = np.zeros(N)
    Im = np.zeros(N)
    for n in range(N):
        Re[n], Im[n] = _fourier_step(xk,N,n)

    Re /= N
    Im /= N
    Cn  = (Re**2+Im**2)**(1/2)
    CSn = Re+Im
    return Cn, CSn

@jit
def reverse_fourier(CSn): #back furie
    N = len(CSn)
    Re = np.zeros(N)
    Im = np.zeros(N)
    for k in range(N):
        Re[k], Im[k] = _fourier_step(CSn, N, k)
    xk = Re + Im
    return xk


def add_pikes(x, min_am_prc, max_am_prc, min_size, max_size):
    peaks_i = [np.random.randint(low = 0, high=len(x)) for _ in range(np.random.randint(low = len(x) * (min_am_prc), high=len(x) * (max_am_prc) )) ]
    new_x = x.copy()
    for i in peaks_i:
        new_x[i] =  ((-1) ** np.random.randint(low = 0, high=2)) *((max_size-min_size) * abs(x.max() + np.random.sample()) * np.random.random() + min_size * abs(x.max() + np.random.sample()))

    # заменен знак
    # импульсный шум и т.д ()
    # положение - случайно, количество - в рамках, но случайно
    # по значению существенно отличаться от данных
    # знак случайный

    return new_x

def myrandom(N):
    y = np.empty(N)
    cell = datetime.datetime.now().microsecond
    cell = cell ** 4
    for i in range(len(y)):
        to = max(int(str(cell)[-1]), int(str(cell)[1])) % 3 + 9
        fro = min(int(str(cell)[-1]), int(str(cell)[1])) % 3
        if fro == to:
            to += 1
        while float(str(cell)[fro:to]) in (0,1):
            to += 1
        cell =  abs(math.log(float(str(cell)[fro:to])))
        y[i] = cell
        cell = int('{:.20f}'.format(cell-int(cell))[2:])
    return y

def normalize_maxmin(x, S):
    xk = ((x - x.min()) / (x.max() - x.min()) - 0.5 ) * 2 * S
    return xk

def defrandom(N):
    x = np.empty(N)
    for i in range(len(x)):
        x[i] = np.random.random()
    return x

def count_sum(x, y):
    k = []
    b = []

    for i in range(len(y) - 1):
        if i % 2 == 0:
            k.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            b.append(y[i + 1] - k[i] * x[i + 1])
        else:
            k.append((math.log(y[i + 1]) - math.log(y[i])) / (x[i + 1] - x[i]))
            b.append(math.log(y[i + 1]) - k[i] * x[i + 1])

    return k, b

def lineral(k,b):
    def fun(x):
        return k * x + b
    return fun

def exponential(a, b):
    def fun(x):
        return math.e ** (a * x + b)
    return fun

def funstar(k,b):
    def fun(x):
        a = []
        for i in range(len(k)):
            if i % 2 == 0:
                a.append(lineral(k[i],b[i])(x[i*250:(i+1)*250]))
            else:
                a.append(exponential(k[i],b[i])(x[i*250:(i+1)*250]))
        a = tuple(a)
        return np.concatenate(a)
    return fun

if __name__ == '__main__':
    x = np.array([i for i in range(10000)])
    y = np.zeros_like(x)
    in_out.plot_functions([(x,y),(x,y), (x, add_pikes(y, 0.01, 0.02, 5, 10)), (x, add_shift(y, 5))])
