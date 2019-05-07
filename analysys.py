import numpy as np
import math
from sympy import *
from numba import jit

def check_stationarity(x, interval_am, var = 0.01):
    interval_median = np.empty(interval_am)
    interval_variance = np.empty(interval_am)
    for i in range(interval_am ):
        interval_median[i] =  x[int(i * (len(x) / interval_am)) : int((i + 1) * (len(x) / interval_am))].mean()
        interval_variance[i] = x[int(i * (len(x) / interval_am)) : int((i + 1) * (len(x) / interval_am))].std()
    stationarity = True
    for i in range(interval_am):
        if abs(interval_median[i] - interval_median.mean()) > interval_median.mean() * var:
            stationarity = False
            print("Means: Wrong place is {}, values is {} > {}".format(i,abs(interval_median[i] - interval_median.mean()), interval_median.mean() * var))
        if abs(interval_variance[i] - interval_variance.mean()) > interval_variance.mean() * var:
            print("Variance: Wrong place is {}, values is {} > {}".format(i,abs(interval_variance[i] - interval_variance.mean()), interval_variance.mean() * var))
            stationarity = False
    print("Stationarity is {}".format(stationarity))
    print("means:")
    print(interval_median)
    print("variances:")
    print(interval_variance)



def statistics(x):
    x_min = x.min()
    x_max = x.max()
    x_mid = x.sum() / len(x)
    variance = 0
    mean_square = 0
    skewness = 0
    eccentricity = 0
    for i in x:
        variance += (i - x_mid)**2
        mean_square += i**2
        skewness += (i - x_mid) ** 3
        eccentricity += (i - x_mid) ** 4
    variance /= len(x)
    mean_square /= len(x)
    standart_deviation = variance ** (1/2)
    root_mean_square = mean_square ** (1/2)
    gamma1 = skewness / standart_deviation ** 3
    eccentricity /= len(x)
    kurtosis = eccentricity / standart_deviation ** 4 - 3
    return {"minimum" : x_min, "maximum":x_max, "variance": variance, "standart_deviation":standart_deviation, "mean_square":mean_square, "root_mean_square":root_mean_square, "Skewness":skewness, 'gamma1' : gamma1, 'eccentricity':eccentricity, "kurtosis":kurtosis}

# def correlation(function, tau):
#     t,T = symbols('t T')
#     p1 = limit(1 / T * integrate(function(t)*function(t+tau),(t, 0, 100)), T, oo)
#     return p1
    # def fun():
    #     def f(x):
    #         return myrandom(1)[0]
    #     return
    # print(autocorrelation(fun(), 1))
    #

def covariation(y1, y2, shift):
    sum = 0
    for i in range(len(y1) - shift):
        sum +=( y1[i] - y1.mean()) * (y2[i+shift] - y2.mean())
    return sum / len(y1)

#@jit(parallel = True)
def correlation(y1, y2, shift):
    div = 0
    y1_mean = y1.mean()
    y2_mean = y2.mean()
    for i in range(len(y1) - shift):
        div +=( y1[i] - y1_mean) * (y2[i+shift] - y2_mean)
    divider = (y1 - y1_mean) ** 2
    return div / divider.sum()

# @jit()
def correliation_shifts(y1, y2, from_shift=0, to_shift=1000):
    corr = np.zeros(to_shift - from_shift)
    for i in range(len(corr)):
        corr[i] = correlation(y1, y2, i)
    return corr


if __name__ == '__main__':
    intervals = 10
    x = np.array([i for i in range(1000)])
    for i in range(intervals - 1):
        print(i, statistics(x[int(i * len(x) / intervals): int((i+1) * len(x) / intervals)]))

