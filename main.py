from model import *
from in_out import *
from analysys import *
from processes import *

if __name__ == "__main__":
    pass
# Task 1
    # plot_functions([calc_fun(lineral(1.4, 0)), calc_fun(lineral(-1.4, 1000)), calc_fun(exponential(0.0015, 0)),
    #                 calc_fun(exponential(-0.0015, 5))])

    # k, b = count_sum([0,250,500,750,1000],[0,900,400,200,800])
    # plot_functions([calc_fun(funstar(k,b))])

# Task 2
    # x = generate_x(0, 1000, 1)
    # y = normalize_maxmin(myrandom(len(x)), 1)
    #
    # plot_functions([(x, y), (x, add_shift(y,100)), (x, add_pikes(y,0.003,0.005,5,11)), (x, add_pikes(y,0.01,0.05,3,4))],
    #                ["normal fun", "shifted fun", "peaks 0.3% - 0.5% & x10", "peaks 1% - 5% & x3"])
    #
    # plt.subplot(211)
    # plt.hist(defrandom(10000),100)
    # plt.subplot(212)
    # plt.hist(normalize_maxmin(myrandom(10000),1),100)
    # plt.show()

# TAsk 3
    # print("Стационарность моего рандома")
    # check_stationarity(myrandom(10000),10, 0.03)
    # print("Стационарность встроенного рандома")
    # check_stationarity(defrandom(10000), 10, 0.03)
    # print("Статистика моего рандома")
    # print(statistics(normalize_maxmin(myrandom(len(x)), 1)))
    # print("Статистика встроенного рандома")
    # print(statistics(normalize_maxmin(defrandom(len(x)), 1)))

# Task 4 Correliations
    # x = generate_x(0, 10000, 1)
    # y1 = normalize_maxmin(myrandom(len(x)), 1)
    # y11 = normalize_maxmin(myrandom(len(x)), 1)
    # y2 = normalize_maxmin(defrandom(len(x)), 1)
    # y22 = normalize_maxmin(defrandom(len(x)), 1)
    # max_shift = 100
    # avcr = np.zeros(max_shift)
    # cr = np.zeros(max_shift)
    # avcv = np.zeros(max_shift)
    # cv = np.zeros(max_shift)
    # vcr = np.zeros(max_shift)
    # vcv = np.zeros(max_shift)
    # x = generate_x(0, max_shift, 1)
    # for i in range(len(x)):
    #     print("counting iter {}".format(i))
    #     avcr[i] = correlation(y1,y1,i)
    #     cr[i] = correlation(y1,y2,i)
    #     avcv[i] = covariation(y1,y1,i)
    #     cv[i] = covariation(y1,y2,i)
    #     vcv[i] = covariation(y1,y11,i)
    #     vcr[i] = covariation(y2,y22,i)
    #
    # plot_functions([(x,avcr),(x,cr),(x, vcr), (x, vcv)],["автокореляция", "корреляция рандома моего и встроенного",  "взаимная корреляция моего рандома", "взаимная корреляция случайного"])

# Вывести гармоники и заметить эффект перекрытия при слишком больших частотах
    # plot_functions([my_harmonic(), my_harmonic(f0=137), my_harmonic(f0=237), my_harmonic(f0=337)])
    # dt = 0.002
    # N = 1000
    # A0 = 100
    # f0 = 3
    # harm = my_harmonic(dt, N, A0, f0)
    # frq = fourier(harm[1])
    # harm2 = reverse_fourier(frq[1])
    # plot_functions([(harm),(generate_x(0, 1/(2*dt), 1/dt/N), frq[0][0:len(harm[1]) // 2]),(harm[0],harm2)])
    # plt.plot(harm[1])
    # plt.plot(harm2,'--')
    # plt.show()
    # plt.plot(harm[1])
    # plt.plot(harm2,'--')
    # plt.show()

# Task 7 summ of harmonics
    # dt = 0.002
    # N = 1000
    # harm1 = my_harmonic(dt, N, A0 = 15, f0 = 3)
    # harm2 = my_harmonic(dt, N, A0 = 100, f0 = 37)
    # harm3 = my_harmonic(dt, N, A0 = 25, f0 = 137)
    # harm_y = harm1[1]+harm2[1]+harm3[1]
    # frq = fourier(harm_y)
    # harm_after_ft = reverse_fourier(frq[1])
    # plot_functions([(harm1[0], harm_y),(generate_x(0, 1/(2*dt), 1/dt/N), frq[0][0:len(harm_y) // 2]),(harm1[0],harm_after_ft)])

# Task 7.5 read_from file
    file_path = r"./data/pgp_1ms.dat"
    y = read_from_file(file_path, 1000)
    print(y)
    frq = fourier(y)
    harm_after_ft = reverse_fourier(frq[1])
    x = generate_x(0,len(y),1)
    N = 1000
    dt = 1
    plot_functions([(x, y),(generate_x(0, 1/(2*dt), 1/dt/N), frq[0][0:len(y) // 2]),(x,harm_after_ft)])
