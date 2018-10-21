import matplotlib.pyplot as plt
import numpy as np
import math
import struct


def calc_fun(fun, interval=(0,1000), step=1):
    x = np.empty(int((interval[1] - interval[0])/step))
    for i in range(len(x)):
        x[i] = interval[0]+step*i
    y = fun(x)
    return (x,y)

def plot_functions(plots, fun_names=()):
    cols = 1
    rows = 1
    for i in range(math.floor(math.sqrt(len(plots))), 0, -1):
        if len(plots) / i == int(len(plots) / i):
            rows = i
            cols = int(len(plots) / i)
            break


    f, axarr = plt.subplots(cols, rows)
    print(f)
    print()
    print(axarr)
    for i, xy in enumerate(plots):
        if rows == 1:
            if cols == 1:
                axarr.plot(xy[0], xy[1])
                if len(fun_names) != 0:
                    axarr.set_title(fun_names[i])
                else:
                    axarr.set_title(str(i) + " function")
            else:
                axarr[i].plot(xy[0], xy[1])
                if len(fun_names) != 0:
                    axarr[i].set_title(fun_names[i])
                else:
                    axarr[i].set_title(str(i) + " function")
        else:
            axarr[int(i / cols), i % cols].plot(xy[0], xy[1])
            if len(fun_names) != 0:
                axarr[int(i / cols), i % cols].set_title(fun_names[i])
            else:
                axarr[int(i / cols), i % cols].set_title(str(i) + " function")

    f.subplots_adjust(hspace=0.7)
    plt.show()

def read_from_file(file_path, length=1000):
    with open(file_path, 'br') as f:
        ans = struct.unpack(str(length) + "f", f.read())
    return ans

