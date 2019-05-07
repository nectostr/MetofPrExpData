import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import struct
import scipy
from scipy import misc
from scipy.io.wavfile import read
import os
import matplotlib.gridspec as gridspec


def read_xcr(file_name, size=16, shape=(300,400), reverse=False):
    with open(file_name, 'rb') as f:
        data = f.read()
    image_len = shape[0]*shape[1]
    arr = list(data[len(data)-image_len*2:])
    if reverse:
        for i in range(0, len(arr)-1, 2):
            arr[i], arr[i+1] = arr[i+1], arr[i]

    arr = np.array(struct.unpack(str(image_len) + "H", bytes(arr))).reshape(shape)
    # arr = np.flipud(arr)
    return arr

sound_max_val = 32768
def read_audio(file_name):
    a = read(file_name)
    rate = a[0]
    a = np.array(a[1], dtype=np,)
    #print(a[0])
    if type(a[0]) != float:
        a = a.transpose()
        a = a[0]
        print("bi")
    return a / sound_max_val, rate

def save_audio(y, rate, filename='res.wav'):
    a = np.array(y, dtype=np.float)
    scipy.io.wavfile.write(filename, rate, a)


def calc_fun(fun, interval=(0,1000), step=1):
    x = np.empty(int((interval[1] - interval[0])/step))
    for i in range(len(x)):
        x[i] = interval[0]+step*i
    y = fun(x)
    return (x,y)

def plot_functions(plots, fun_names=(), types = (), fig_name='figure 0'):
    # plt.xkcd()
    # mpl.rcParams['axes.color_cycle'] = ['violet', 'lightblue', 'magenta']
    for i in range(len(plots)):
        if plots[i][0] is None:
            plots[i] = (np.arange(len(plots[i][1])),plots[i][1])
    cols = 1
    rows = 1
    for i in range(math.floor(math.sqrt(len(plots))), 0, -1):
        if len(plots) / i == int(len(plots) / i):
            rows = i
            cols = int(len(plots) / i)
            break


    f, axarr = plt.subplots(cols, rows)
    f.suptitle(fig_name, fontsize=12)
    # f.patch.set_facecolor('lightblue')

    # print(f)
    # print()
    # print(axarr)
    for i, xy in enumerate(plots):
        if rows == 1:
            if cols == 1:
                if not types:
                    axarr.plot(xy[0], xy[1])
                elif types[i] == "plot":
                    axarr.plot(xy[0], xy[1])
                elif types[i] == "hist":
                    axarr.hist(xy[0], xy[1])
                elif types[i] == "imag":
                    axarr.imshow(xy[0], xy[1],
                                 interpolation="none",vmin=0, vmax=255, cmap='gist_gray')
                if len(fun_names) != 0:
                    axarr.set_title(fun_names[i])
                else:
                    axarr.set_title(str(i) + " function")
            else:
                if not types:
                    axarr[i].plot(xy[0], xy[1])
                elif types[i] == "plot":
                    axarr[i].plot(xy[0], xy[1])
                elif types[i] == "hist":
                    axarr[i].hist(xy[0], xy[1])
                elif types[i] == "imag":
                    axarr[i].imshow(xy[0],
                                 interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
                if len(fun_names) != 0:
                    axarr[i].set_title(fun_names[i])
                else:
                    axarr[i].set_title(str(i) + " function")
        else:
            if not types:
                axarr[i % cols, i // cols].plot(xy[0], xy[1])
            elif types[i] == "plot":
                axarr[i % cols, i // cols].plot(xy[0], xy[1])
            elif types[i] == "hist":
                axarr[i % cols, i // cols].hist(xy[0], xy[1])
            elif types[i] == "imag":
                axarr[i % cols, i // cols].imshow(xy[0],
                             interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
            if len(fun_names) != 0:
                axarr[i % cols, i // cols].set_title(fun_names[i])
            else:
                axarr[i % cols, i // cols].set_title(str(i) + " function")

    f.subplots_adjust(hspace=0.7)

    plt.show()

def show_images(plots, fun_names = (), types = (), fig_name='figure 0'):

    # plt.xkcd()
    # mpl.rcParams['axes.color_cycle'] = ['violet', 'lightblue', 'magenta']
    for i in range(len(plots)):
        if plots[i][0] is None:
            plots[i] = (np.arange(len(plots[i][1])),plots[i][1])
    cols = 1
    rows = 1
    for i in range(math.floor(math.sqrt(len(plots))), 0, -1):
        if len(plots) / i == int(len(plots) / i):
            rows = i
            cols = int(float(len(plots) / i))
            break


    f, axarr = plt.subplots(rows, cols)
    f.suptitle(fig_name, fontsize=12)

    # f.patch.set_facecolor('lightblue')
    bigger = False
    if types == "sizing":
        space = 100
        tiles_x = sum([i.shape[0] for i in plots]) #max(plots[0].shape[0], plots[1].shape[0])
        tiles_y = sum([i.shape[1] for i in plots]) + len(plots) * space
        gridspec.GridSpec(tiles_x, tiles_y)
        plt.subplot2grid((tiles_x, tiles_y), (0, 0),
                             colspan=plots[0].shape[1], rowspan=plots[0].shape[0])
        plt.imshow(plots[0], interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
        wii = plots[0].shape[1]
        for i in range(1,len(plots)):
            plt.subplot2grid((tiles_x, tiles_y), (0, wii),
                                 colspan=plots[i].shape[1], rowspan=plots[i].shape[0])
            wii += plots[i].shape[1] + space
            plt.imshow(plots[i], interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
        print(tiles_x, tiles_y, plots[0].shape[0], plots[0].shape[1])
        print(tiles_x, tiles_y, plots[1].shape[0], plots[1].shape[1])
        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        mng.full_screen_toggle()
        plt.show()
        return 0
        # for i in range(2):
        #     if len(fun_names) != 0:
        #         axarr[i].set_title(fun_names[i])
        #     else:
        #         axarr[i].set_title(str(i) + " function")
    # print(f)
    # print()
    # print(axarr)
    for i, xy in enumerate(plots):
        if rows == 1:
            if cols == 1:
                if not types or types[i] == "imag":
                    axarr.imshow(xy,
                                 interpolation="none",vmin=0, vmax=255, cmap='gist_gray')
                if len(fun_names) != 0:
                    axarr.set_title(fun_names[i])
                else:
                    axarr.set_title(str(i) + " function")
            else:
                if not types or types[i] == "imag":
                        axarr[i].imshow(xy,
                                     interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
                if len(fun_names) != 0:
                    axarr[i].set_title(fun_names[i])
                else:
                    axarr[i].set_title(str(i) + " function")
        else:
            if not types or types[i] == "imag":
                axarr[i // cols, i % cols].imshow(xy,
                             interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
            if len(fun_names) != 0:
                axarr[i // cols, i % cols].set_title(fun_names[i])
            else:
                axarr[i // cols, i % cols].set_title(str(i) + " function")

    f.subplots_adjust(hspace=0.7)
    plt.show()

def read_from_file(file_path, length=1000, typ='f'):
    if type(length) == int:
        with open(file_path, 'br') as f:
                ans = struct.unpack(str(length) + typ, f.read())
    else:
        with open(file_path, 'br') as f:
            ans = np.array(struct.unpack(str(length[0]*length[1]) + typ, f.read()))
            ans = ans.reshape(length)
    return ans

def read_image(file_name, directory=r"./"):
    return scipy.misc.imread(os.path.join(directory, file_name))

