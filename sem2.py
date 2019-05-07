import in_out as IO
import model as m
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import gradational_transformations as GT
from scipy.misc import imresize
import resizing as reS
import analysys
import pic_filters as PF
import matplotlib
matplotlib.rcParams['image.cmap'] = 'gray'
np.random.seed(8)


def task1():
    """
    reading picture, find its row and col variance and median, plot histogram
    """
    img = IO.read_image("photo1.jpg", r".\data")
    ma_col = img.max()
    mi_col = img.min()
    avg_r = np.empty(img.shape[0])
    disp_r = np.empty(img.shape[0])
    for i in range(img.shape[0]):
        avg_r[i] = img[i].mean()
        disp_r[i] = img[i].var()

    avg_c = np.empty(img.shape[1])
    disp_c = np.empty(img.shape[1])
    for i in range(img.shape[1]):
        avg_c[i] = img[:, i].mean()
        disp_c[i] = img[:, i].var()

    plt.figure()
    plt.imshow(img, cmap="Greys")
    # plt.show()
    IO.plot_functions([(None, avg_r), (None, disp_r), (None, avg_c), (None, disp_c)],
                      ["avg rows", "disp rows", "avg cols", "disp cols"])
    print(mi_col, ma_col)

    plt.figure()
    plt.plot(m.hist(img, 255))
    plt.figure()
    plt.hist(img.flatten(), 255)
    plt.show()

def find_face():
    """
    draws lines ower hight change periods on face
    """
    img = IO.read_image("photo1.jpg", r".\data")
    ma = img.max()
    mi = img.min()

    avg_r = np.empty(img.shape[0])
    for i in range(img.shape[0]):
        avg_r[i] = img[i].mean()

    avg_c = np.empty(img.shape[1])
    for i in range(img.shape[1]):
        avg_c[i] = img[:, i].mean()

    IO.plot_functions([(None, avg_r), (None, avg_c)],
                      ["avg rows", "avg cols"])
    xs = []
    ys = []
    has_prev = False
    step = 5
    for i in range(step, len(avg_r) - step, step // 2):
        if abs(avg_r[i] - avg_r[i + step]) > ma / 30:
            xs.append(i + step // 2)

    step = 5
    for i in range(step, len(avg_c) - step, step // 2):
        if abs(avg_c[i] - avg_c[i + step]) > ma / 30:
            ys.append(i + step // 2)

    print(len(xs), len(ys))
    img_l = img
    for x in range(len(xs) - 1):
        for y in range(len(ys) - 1):
            print(xs[x], ys[y])
            img_l = m.draw_line(img_l, ys[y], xs[x], ys[y], xs[x + 1])
            img_l = m.draw_line(img_l, ys[y], xs[x], ys[y + 1], xs[x])
            img_l = m.draw_line(img_l, ys[y], xs[x + 1], ys[y + 1], xs[x + 1])
            img_l = m.draw_line(img_l, ys[y + 1], xs[x], ys[y + 1], xs[x + 1])

    plt.imshow(img_l)
    plt.show()

@jit
def to_one_chanel(_img) -> np.array:
    img = np.empty((_img.shape[0], _img.shape[1]),dtype=np.int)
    img[:,:] = _img[:,:,0]
    return img

def find_edges():
    img = IO.read_image("photo1.jpg", r".\data")
    img = to_one_chanel(img)
    points = np.zeros(img.shape)

    step = 3
    for i in range(step, img.shape[0] - step):
        for j in range(step, img.shape[1] - step):
            a = img[i - step:i + step, j - step:j + step]
            a: np.ndarray = np.greater(img[i, j], a)
            if a.sum() > a.size * 0.8:
                # if  len([[1 for x in range(-step//2, step//2) if img[i,j] > img[i+x, j+y]]
                #          for y in range(-step//2, step//2)]) > step*step*0.3:
                # img[i,j] > img[i-step:i+step, j-step:j+step].mean():
                points[i, j] = img[i, j]
    print(len(points))
    plt.imshow(img, cmap='gist_gray')
    plt.figure()
    plt.imshow(points)
    plt.show()

def task2():
    """
    show resizing in + and -
    :return:
    """
    img = IO.read_image("photo1.jpg", r".\data")
    img = to_one_chanel(img)
    print_size = 3
    mul = 2.7
    img2 = reS.resize_next_neigbour(img, mul)
    plt.figure(figsize=(print_size,print_size))
    plt.imshow(img, cmap='gist_gray')
    plt.figure(figsize=(print_size*mul,print_size*mul))
    plt.imshow(img2, cmap='gist_gray')
    plt.show()

DPI = 111.94

def task4():
    img = IO.read_image("photo1.jpg", r".\data")
    img = to_one_chanel(img)
    mul = 1.7
    img2 = reS.resize_next_neigbour(img, mul)
    img3 = reS.resize_bilinear_interpolation(img, mul)
    IO.show_images([img, img2, img3], types="sizing")
    plt.figure(figsize=(img.shape[1] / DPI, img.shape[0] / DPI))
    plt.imshow(img, aspect="auto", interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
    plt.figure(figsize=(img2.shape[1] / DPI, img2.shape[0] / DPI))
    plt.imshow(img2, aspect="auto", interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
    plt.figure(figsize=(img3.shape[1] / DPI, img3.shape[0] / DPI))
    plt.imshow(img3, aspect="auto", interpolation="none", vmin=0, vmax=255, cmap='gist_gray')
    plt.show()

def task5():
    img = IO.read_image("image2.jpg", r".\data")
    # img = IO.read_image("image2.jpg", r".\data")
    # print(img.shape)
    img = to_one_chanel(img)
    i_img = GT.invert(img)
    gc_img = GT.gamma_correction(img, 0.5)
    gc_img = np.array((gc_img-gc_img.min())/(gc_img.max() - gc_img.min())*255, dtype=np.int)
    l_img = GT.logarifmic_correction(img, 1, base = 1.5)
    l_img = np.array((l_img - l_img.min())/(l_img.max()-l_img.min())*255, dtype=np.int)
    IO.show_images([img, i_img, gc_img, l_img],
                   ["image", "invert image", "image gamma", "image logarifm"])

def task6():
    pass
    """
    1. ??????????? ?????????
    2. ??????????


    ????????? ??????? (????? > 1), ?????? ??????? ? ?? - ?? ????????
    ?????? - ??? ?????? ?????????????... - ?????????? ?? ? 0-1 ????? ???????????? ??
    ? ??? ?????????? - ?????????? ????? ???????? ?????? ? ?? ????????????
    :return:
    """
    img = IO.read_image("image2.jpg", r".\data")
    img = to_one_chanel(img)
    print(img.shape)
    LD = m.luminance_distribution(img)
    img2 = m.apply(img, LD)
    IO.show_images([img, img2], ["image", "image vs luminance distribution"])
    bLD1 = m.back_luminance_distribution(LD)
    bLD2 = m.luminance_distribution(img2)
    IO.plot_functions([(None, LD)])
    IO.plot_functions([(None, LD), (None, bLD1)])
    # IO.plot_functions([(None, LD), (None, bLD1), (None, bLD2)])

def task7():
    """
    was to 26/3
    :return:
    """
    # xcr = IO.read_xcr(r"data\h400x300.xcr", 1024)
    xcr = IO.read_xcr(r"data\c12-85h.xcr", shape=(1024, 1024), reverse=True)
    plt.imshow(xcr / xcr.max() * 255, cmap='gray')
    plt.show()
    line_fs = []
    for i in range(1, xcr.shape[0], xcr.shape[0]//20):
        a = m.fourier(xcr[i])
        line_fs.append(a[0][1:len(a[0])//2])

    line_fs = np.array(line_fs)
    IO.plot_functions([(np.arange(0, 0.5, 0.5/(len(line_fs[0]))),line_fs[0])]) # 0ю5 - это то, что он нам сказал как предел
    print(len(line_fs[0]))
    corr = analysys.correliation_shifts(line_fs[0], line_fs[0], len(line_fs[0]))

    plt.show()
    dt = 1
    f1 = 0.29-0.08
    f2 = 0.29+0.08
    bsf = m.create_band_stop_filter(f1, f2, dt=dt, m=32)
    xcr2 = np.zeros(xcr.shape)
    for i in range(len(xcr)):
        xcr2[i] = m.convolutional(xcr[i], bsf)
    plt.imshow(xcr, cmap='gray')
    plt.figure()
    plt.imshow(xcr2, cmap="gray")
    plt.show()

def task8(not_show=True):
    """
    model_.jpg
    ???????????  - изображение
    1. ????????? ??????????? гистограмма
    2. ????????? ???????? - ?????? ?? ???? ??????? ? ????????? ??????(???????? ? ????)
    случаный шум - идем по строкам и делам + какой то рандом * S . потом скайл
    3. ??????? ???? ? ?????
    шум - соль перец
    4. зашумит спайками (для каждой строки + и -)
    впрос про scale - как его делать - обрезая или /max ?
    """
    img = IO.read_image("data/MODEL.jpg")
    plt.imshow(img, cmap="gray")
    plt.show()
    if not not_show:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.figure()
        plt.hist(img.flatten(), 255)
        plt.show()

    img_random_noise = np.zeros(img.shape)
    prob_of_empty = 0.3
    shum_len = 100
    for i in range(len(img)):
        img_random_noise[i] = img[i] + \
                              np.random.choice(
                                  [i for i in range(shum_len)],
                                  p=[prob_of_empty]+[(1-prob_of_empty)/(shum_len-1)
                                                     for _ in range(shum_len-1)],
                                  size=img.shape[1])

    img_random_noise = m.normalize(img_random_noise)

    img_SP_noise = np.zeros(img.shape)

    for i in range(len(img)):
        img_SP_noise[i] =  m.add_pikes(img[i].copy(), len(img[i])//6, len(img[i])//5, 100, 200)

    img_SP_noise += img
    img_SP_noise = m.normalize(img_SP_noise)
    if not not_show:
        plt.imshow(img_random_noise, cmap="gray")
        plt.show()
        plt.imshow(img_SP_noise, cmap="gray")
        plt.show()
    return img_random_noise, img_SP_noise




def apply_2D_f():
    #xcr = IO.read_xcr(r"data\h400x300.xcr")
    xcr = IO.read_xcr(r"data\c12-85h.xcr")
    plt.imshow(xcr / xcr.max() * 255, )
    plt.show()
    a = m.fourier(xcr)

def task9():
    """
    Усредняющий и медианный филтр к шумам
    :return:
    """
    img = IO.read_image("data/MODEL.jpg")
    r_n, sp_n = task8()
    filetered_r_n = m.normalize(PF.averaging_filter(r_n))
    filetered_sp_n = m.normalize(PF.averaging_filter(sp_n))
    filetered_med_r_n = m.normalize(PF.median_filter(r_n))
    filetered_med_sp_n = m.normalize(PF.median_filter(sp_n, 7))
    plt.imshow(r_n, cmap="gray")
    plt.figure()
    plt.imshow(filetered_r_n, cmap="gray")
    plt.figure()
    plt.imshow(filetered_med_r_n, cmap="gray")
    plt.show()
    plt.imshow(img, cmap='gray')
    plt.figure()
    plt.imshow(sp_n, cmap="gray")
    plt.figure()
    plt.imshow(filetered_sp_n, cmap="gray")
    plt.figure()
    plt.imshow(filetered_med_sp_n, cmap="gray")
    plt.show()
    # TODO - pic with both noise - what filter or what order combine filter

def task10():
    """
    Восстановление изображений
    :return:
    """
    img = IO.read_image("data/image2.jpg")
    img = to_one_chanel(img)
    img_f = np.empty(img.shape)
    for i in range(img.shape[0]):
        img_f[i] = m.fourier(img[i])[1]

    for j in range(img_f.shape[1]):
        img_f[:,j] = m.fourier(img_f[:,j])[1]

    # plt.imshow(m.normalize(img_f), cmap='gray')
    back_f_img = np.empty(img.shape)


    for j in range(img.shape[0]):
        back_f_img[j] = m.reverse_fourier(img_f[j])
    for i in range(img.shape[1]):
        back_f_img[:,i] = m.reverse_fourier(back_f_img[:,i])


    f = plt.figure()
    plt.imshow(back_f_img, cmap='gray')
    plt.show()

def task11():
    """
    Distortion image filtration
    :return:
    """
    img = IO.read_from_file(r"data/InverseFilter/blur307x221D.dat", (221, 307), typ='f')
    img = img[::-1,:]
    plt.title("Исходное изображение")
    plt.imshow(img)
    f = plt.figure()
    f.suptitle("Восстановление не зашумленного изображения")
    moduler = IO.read_from_file(r"data/InverseFilter/kernD76_f4.dat", 76, typ='f')
    add = np.full(img.shape[1]-len(moduler), 0)

    moduler = np.append(moduler, add)
    fou_img = m.fourier_2D(img)
    fou_mod = m.fourier(moduler)[1]


    for i in range(len(fou_img)):
        fou_img[i] = fou_img[i] / fou_mod

    img2 = m.fourier_2D_back(fou_img)
    plt.imshow(img2)
    plt.show()

def task11_c():
    """
    Distortion image filtration
    :return:
    """
    img = IO.read_from_file(r"data/InverseFilter/blur307x221D.dat", (221, 307), typ='f')
    img = img[::-1,:]
    plt.title("Исходное изображение")
    plt.imshow(img)
    plt.figure()
    plt.title("Восстановленное, не зашумленное")
    moduler = IO.read_from_file(r"data/InverseFilter/kernD76_f4.dat", 76, typ='f')
    add = np.full(img.shape[1]-len(moduler), 0)
    moduler = np.append(moduler, add)

    fou_img = m.fourier_2D(img, complex=True)
    fou_mod = m.fourier_complex(moduler)

    res = np.zeros(fou_img.shape, dtype=np.float)
    res = fou_img / fou_mod

    img2 = m.fourier_2D_back(res, complex=True)
    img2 = np.array(img2.real, dtype=np.float)
    plt.imshow(img2)
    plt.show()

def task12():
    """\
    Distortion image filtration
    :return:
    """
    img = IO.read_from_file(r"data/InverseFilter/blur307x221D_N.dat", (221, 307), typ='f')
    img = img[::-1,:]
    plt.title("Исходное изображение")
    plt.imshow(img)
    plt.figure()
    moduler = IO.read_from_file(r"data/InverseFilter/kernD76_f4.dat", 76, typ='f')
    add = np.full(img.shape[1]-len(moduler), 0)
    moduler = np.append(moduler, add)

    fou_img = m.fourier_2D(img, complex=True)
    fou_mod = m.fourier_complex(moduler)

    res = fou_img * ((fou_mod.conj()) / ((np.absolute(fou_mod)**2) + 0.00001))

    img2 = m.fourier_2D_back(res, complex=True)
    img2 = np.array(img2.real, dtype=np.float)
    plt.title("Восстановленное, зашумленное")
    plt.imshow(img2)

    img3 = PF.averaging_filter(img2, 3)
    # img3 = PF.median_filter(img3, 5)
    plt.figure()
    plt.title("Применение медианного филтра")
    plt.imshow(img3)
    plt.show()

def task13():
    """
    Выделить контуры объекта MODEL (должны быть как бинарное изображение):
    ФНЧ (фильтр низких чатсот расфокусирует изображение),
    вычитаем из исходного применяем пороговое преобразование.
    ФВЧ (фильтр высоких частот к тому же
    изображению построчно или по столбцам), аналогично
    1. Попробовать с шумами
    """
    img = IO.read_image(r"data/MODEL.jpg")

    lpf = m.create_low_pass_filter(0.05, 8, 1)
    img2 = m.convolutional_2d(img, lpf)
    img3 = img - img2
    img3 = m.normalize(m.threshold_filter_2d(img3, 240))
    hpf = m.create_hight_pass_filter(0.2, 8, 1)
    img4 = m.convolutional_2d(img, hpf)
    img4 = m.normalize(m.threshold_filter_2d(img4, 100))
    IO.show_images([img, img2, img3, img4], ["start", "conv res", "low filter", "hight filter"],
                   fig_name="Контур с помощью фильров")

    img_n = m.add_random_noise_2d(img)
    img_n = m.add_salt_pepper_noise_2d(img_n)
    img_n = PF.median_filter(img_n)
    img_n = PF.averaging_filter(img_n)

    lpf = m.create_low_pass_filter(0.05, 8, 1)
    img2 = m.convolutional_2d(img_n, lpf)
    img3 = img - img2
    # img3 = m.normalize(img3, 255)
    # print(img3.min(), img3.max())
    img3 = m.normalize(m.threshold_filter_2d(img3, 78))
    hpf = m.create_hight_pass_filter(0.2, 4, 1)
    img4 = m.convolutional_2d(img_n, hpf)
    # img4 = m.normalize(img4, 255)
    # print(img4.min(), img4.max())
    # img4 = m.threshold_filter_2d(img4, 0)
    IO.show_images([img_n, img2, img3, m.normalize(img4)],
                   ["start", "conv res", "low filter", "hight filter"],
                   fig_name="Контур филтрами зашумленного изображения")


def task14():
    """
    1 оригонал (оригинал + шум (sp + noise + both))
    grad (sobel)
    laplasian
    :return:
    """
    img = IO.read_image(r"data/MODEL.jpg")
    img2 = PF.gradient_sobel(img, "h")
    img3 = PF.gradient_sobel(img, "v")
    img4 = PF.gradient_sobel(img, "a")
    img5 = m.normalize(np.absolute(img2)) +\
                    m.normalize(np.absolute(img3))
    IO.show_images([img, m.normalize(img2), m.normalize(img3),
                m.normalize(img4), img5, img-img5]
                   ,["Изначальное изображение", "Маска собела горизонт", "Маска собела верт",
                     "Сумма нескольких масок", "Применение горизонтальной и вертикальной масок",
                     "Суммируем ради красоты"],
                   fig_name="Маски Собела")
    img2 = PF.laplassian(img, "a")
    img3 = PF.laplassian(img, "b")
    IO.show_images([m.normalize(img2), m.normalize(img3),
                    img+m.normalize(img2), img+m.normalize(img3)],
                   ["Лаплассиан, маска1", "Лапл, маска 2","Сумма с маск 1", "Сумма с маской 2"],
                   fig_name="Лаплассиан")

def task15():
    """
    1.oroginal
    2. original + noise
    Контуры
    а)erosion
    b)silatation
    :return:
    """
    img = IO.read_image(r"data/Dots.jpg")
    img = to_one_chanel(img)
    print(img.max(), img.mean(), img.min())
    img = 255 - img
    print(set(img.flatten()))
    img = m.threshold_filter_2d(img, 100)
    print(set(img.flatten()))
    img2 = PF.erosion(img, 17)
    img3 = PF.dilatation(img2, 17)
    IO.show_images([m.normalize(img), m.normalize(img2), m.normalize(img3)], fig_name="Эрозия, пример")

    # with model
    img = IO.read_image(r"data/MODEL.jpg")
    img2 = m.threshold_filter_2d(img, 100)
    img3 = PF.erosion(img2, 10)
    img4 = img2 - img3
    img5 = PF.dilatation(img2, 10)
    img6 = img5 - img2
    IO.show_images([m.normalize(img2), m.normalize(img4), m.normalize(img6)]
                   ["Изначальное бинаризированное", "Эрозия", "Дилатация"],
                   fig_name="Эрозия, детектирования края")


if __name__ == "__main__":
    # task1()
    # task2()
    # task4()
    # find_face()
    # find_edges()
    # task5()
    # task6()
    # task7()
    # task8()
    # task9()
    # task10()
    # task11()


    task11_c()
    task12()
    task13()
    task14()
    task15()
