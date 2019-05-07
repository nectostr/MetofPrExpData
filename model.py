import datetime
import numpy as np
import math
import in_out
from in_out import *
from analysys import *
from numba import jit
import copy
import collections
from model_.filters import *
from model_.fourier import *

def anti_shift(x):
    return x - x.mean()
    # должно уйти 0 в фурье, но если процесс не стационарен, то среднее теряетсмысл

def add_shift(x, shi):
    return x + shi

def generate_x(fromm, to, step):
    x = np.empty(int((to - fromm) / step), dtype=np.double)
    for i in range(len(x)):
        x[i] = fromm + step * i
    return x

@jit
def my_harmonic(dt=0.002, N=1000, A0 = 200, f0 = 37, opp=0):
    mov = int(round(opp))
    print("\tMOV: {}".format(mov))
    k = np.arange(0, N + mov, dtype=np.double)
    y = A0 * np.sin(2 * np.pi * f0 * k * dt)
    y = y[mov:]
    k = k[:N]
    return k, y

def anti_pikes(x, S):
    x_new = x.copy()
    if abs(x[0]) > S:
        x_new[0] = x[1] - (x[2] - x[1])
    if abs(x[-1]) > S:
        x_new[-1] = x[-2] - (x[-3] - x[-2])
    for i in range(1,len(x)-1):
        if abs(x[i]) > S:
            x_new[i] = (x[i-1]+x[i+1])/2
    return x_new

def add_pikes(x, min_am, max_am, min_size, max_size):
    size = np.random.randint(low=min_am, high=max_am)
    peaks_i = np.random.randint(low=0, high=len(x), size=size)
    new_x = np.zeros(x.shape)
    for i in peaks_i:
        new_x[i] = ((-1) ** np.random.randint(low=0, high=2)) * \
                    (min_size + (max_size - min_size) * np.random.sample())
                    #((max_size-min_size) * abs(x.max()) + np.random.sample() * np.random.random() + min_size * abs(x.max()) + np.random.sample())
    return new_x
    # заменен знак
    # импульсный шум и т.д ()
    # положение - случайно, количество - в рамках, но случайно
    # по значению существенно отличаться от данных
    # знак случайный

def anti_random(x,L=2):
    new_x = x.copy()
    for i in range(L, len(x)-L):
        new_x[i] = x[i-L:i+L].mean()
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


def normalize(x, S=255):
    return (x - x.min()) / (x.max() - x.min()) * S

def defrandom(N):
    x = np.empty(N)
    for i in range(len(x)):
        x[i] = np.random.random()
    return x

def tiks(N, amount, size_from, size_to):
    y = np.zeros(N)
    step = N // (amount+1)
    for i in range(step, N, step):
        y[i] = np.random.randint(size_from, size_to+1)
        # if i // step == 2:
        #     y[i] *= -1
    return y

def heart_module(M, f0=7, dt=0.005, al=25):
    k = np.arange(0,M,1)
    sin_part = np.sin(2*np.pi*f0*k*dt)
    exsp_part = exponential(-al, 0)(k*dt)
    ret =  sin_part*exsp_part
    return ret / ret.max()


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

def anti_trend1(x): #lineral
    L = 1
    new_x = x.copy()
    for i in range(L,len(x) - L):
        new_x[i] = (x[i+1] - new_x[i-1])/(2*L)
    return new_x



def anti_trend2(x, L=10): # univers
    trend_x = x.copy()
    for i in range(len(x)-L):
        trend_x[i] = x[i:i+L].mean()
    trend_x2 = trend_x.copy()
    for i in range(len(x)-1,len(x)-L-2, -1):
        trend_x2[i] = x[i-L:i].mean()
    trend_x[len(x)-L:] = trend_x2[len(x)-L:] - (trend_x2[len(x)-L-1] - trend_x[len(x)-L-1])
    # for i in range(len(x)-1,len(x)-L-2, -1):
    #     trend_x[i] = x[i-L:i].mean()
    return trend_x

def anti_trend21(x, L=10): # univers
    trend_x = x.copy()
    for i in range(L//2,len(x)-L//2):
        trend_x[i] = x[i-L//2:i+L//2].mean()
    trend_x2 = trend_x.copy()
    for i in range(len(x)-1,len(x)-L-2, -1):
        trend_x2[i] = x[i-L:i].mean()
    for i in range(0,L+1, 1):
        trend_x2[i] = x[i:i+L].mean()
    trend_x[len(x)-L:] = trend_x2[len(x)-L:] - (trend_x2[len(x)-L-1] - trend_x[len(x)-L-1])
    trend_x[:L] = trend_x2[:L] - (trend_x2[L - 1] - trend_x[L - 1])
    # for i in range(len(x)-1,len(x)-L-2, -1):
    #     trend_x[i] = x[i-L:i].mean()
    return trend_x




def lineral(k,b):
    def fun(x):
        return k * x + b
    return fun

def exponential(a, b):
    def fun(x):
        return math.e ** (a * x + b)
    return fun

def funstar(k,b,xs):
    def fun(x):
        a = []
        for i in range(len(k)):
            if i % 2 == 0:
                a.append(lineral(k[i],b[i])(x[xs[i]:xs[i+1]]))
            else:
                a.append(exponential(k[i],b[i])(x[xs[i]:xs[i+1]]))
        a = tuple(a)
        return np.concatenate(a)
    return fun


def norm_pic(img, S = 255):
    img = S * (img - img.min()) / (img.max() - img.min)
    return img

@jit()
def hist(arr, h):
    hi = np.zeros(h)
    mi = arr.min()
    ma = arr.max()
    for i in arr.flatten():
        hi[int(h*(i - mi)/(ma - mi))] += 1
    return hi

def draw_line(_img, x1, y1, x2, y2):
    img = copy.deepcopy(_img)
    if (1 > len(img.shape) > 3):
        return False
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            if (x2 == x1):
                    if x == x1 and x <= img.shape[1]:
                        img[y, x, 0] = 255
                        img[y, x, 1] = 0
                        img[y, x, 2] = 0
            elif y2 == y1:
                if y == y1 and y <= img.shape[0]:
                    img[y, x, 0] = 255
                    img[y, x, 1] = 0
                    img[y, x, 2] = 0
            elif abs((x - x1)/(x2-x1) - (y - y1)/(y2 - y1)) <= 0.01:
                img[y, x, 0] = 255
                img[y, x, 1] = 0
                img[y, x, 2] = 0
    return img


def luminance_distribution(_img, max_coef=255):
    #TODO - more efficient
    x = np.zeros(int(round(max_coef))+1)
    c = collections.Counter(_img.flatten())
    for i in c:
        x[int(round(i))] = c[i]
    for i in range(1, len(x)):
        x[i] += x[i-1]
    return x / 255

def back_luminance_distribution(_LD = None, _img=None):
    if _LD is None:
        _LD = luminance_distribution(_img)
    _LD /= _LD.max()
    _LD *= 255
    _LD = np.array(_LD, dtype=np.int)
    xEy = np.zeros(256)
    for i in range(len(_LD)):
        xEy[_LD[i]] = i
    return xEy

def apply(_img, _LD):
    img = np.zeros(_img.shape, dtype=np.int)
    for i in range(_img.shape[0]):
        for j in range(_img.shape[1]):
            img[i,j] = _LD[int(float(_img[i,j]))]
    return img
    # if __name__ == '__main__':
#     x = np.array([i for i in range(10000)])
#     y = np.zeros_like(x)
#
#     in_out.plot_functions([(x,y),(x,y), (x, add_pikes(y, 0.01, 0.02, 5, 10)), (x, add_shift(y, 5))])


def add_random_noise_2d(img):
    prob_of_empty = 0.3
    shum_len = 100
    img_random_noise = np.zeros(img.shape)
    for i in range(len(img)):
        img_random_noise[i] = img[i] + \
                              np.random.choice(
                                  [i for i in range(shum_len)],
                                  p=[prob_of_empty] + [(1 - prob_of_empty) / (shum_len - 1)
                                                       for _ in range(shum_len - 1)],
                                  size=img.shape[1])

    return normalize(img_random_noise, 255)


def add_salt_pepper_noise_2d(img):
    img_SP_noise = np.zeros(img.shape)
    for i in range(len(img)):
        img_SP_noise[i] = add_pikes(img[i].copy(), len(img[i]) // 6, len(img[i]) // 5, 100, 200)

    img_SP_noise += img
    return normalize(img_SP_noise, 255)
