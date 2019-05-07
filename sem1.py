from model import *
from in_out import *
from analysys import *



def create_print_pair_f(f, dt ):
    print(int((1/dt)//2))
    return generate_x(0, 1 / (2*dt), 1/(dt*len(f))), f[:len(f)//2]

""" trend lines """
def task1():
    plot_functions([calc_fun(lineral(1.4, 0)), calc_fun(lineral(-1.4, 1000)), calc_fun(exponential(0.0015, 0)),
                    calc_fun(exponential(-0.0015, 5))])
    xs = [0,250,500,750,1000]
    k, b = count_sum(xs,[0,900,400,200,800])
    plot_functions([calc_fun(funstar(k,b, xs))])

""" Shift and pikes """
def task2():
    x = generate_x(0, 1000, 1)
    y = normalize_maxmin(myrandom(len(x)), 1)

    plot_functions([(x, y), (x, add_shift(y,100)), (x, add_pikes(y,0.003,0.005,5,11)),
    (x, add_pikes(y,0.01,0.05,3,4))],
     ["normal fun", "shifted fun",
     "peaks 0.3% - 0.5% & x10", "peaks 1% - 5% & x3"])

    plt.subplot(211)
    plt.hist(defrandom(10000),100)
    plt.subplot(212)
    plt.hist(normalize_maxmin(myrandom(10000),1),100)
    plt.show()

""" Statistics of random funcs """
def task3():
    x = normalize_maxmin(defrandom(100))
    print("Стационарность моего рандома")
    check_stationarity(myrandom(10000),10, 0.03)
    print("Стационарность встроенного рандома")
    check_stationarity(defrandom(10000), 10, 0.03)
    print("Статистика моего рандома")
    print(statistics(normalize_maxmin(myrandom(len(x)), 1)))
    print("Статистика встроенного рандома")
    print(statistics(normalize_maxmin(defrandom(len(x)), 1)))

""" Coreliations and covariations """
def task4():
    x = generate_x(0, 10000, 1)
    y1 = normalize_maxmin(myrandom(len(x)), 1)
    y11 = normalize_maxmin(myrandom(len(x)), 1)
    y2 = normalize_maxmin(defrandom(len(x)), 1)
    y22 = normalize_maxmin(defrandom(len(x)), 1)
    max_shift = 100
    avcr = np.zeros(max_shift)
    cr = np.zeros(max_shift)
    avcv = np.zeros(max_shift)
    cv = np.zeros(max_shift)
    vcr = np.zeros(max_shift)
    vcv = np.zeros(max_shift)
    x = generate_x(0, max_shift, 1)
    for i in range(len(x)):
        print("counting iter {}".format(i))
        avcr[i] = correlation(y1,y1,i)
        cr[i] = correlation(y1,y2,i)
        avcv[i] = covariation(y1,y1,i)
        cv[i] = covariation(y1,y2,i)
        vcv[i] = correlation(y1,y11,i)
        vcr[i] = correlation(y2,y22,i)

    plot_functions([(x,avcr),(x,cr),(x, vcr), (x, vcv)],\
    ["автокореляция", "корреляция рандома моего и встроенного",  \
    "взаимная корреляция моего рандома", "взаимная корреляция случайного"])

""" Harmonics and "lepestki effect" """
def task5():
    plot_functions([my_harmonic(), my_harmonic(f0=137), my_harmonic(f0=237), my_harmonic(f0=337)])
    dt = 0.002
    N = 1000
    A0 = 100
    f0 = 3
    harm = my_harmonic(dt, N, A0, f0)
    frq = fourier(harm[1])
    harm2 = reverse_fourier(frq[1])
    plot_functions([(harm),(generate_x(0, 1/(2*dt), 1/dt/N), frq[0][0:len(harm[1]) // 2]),(harm[0],harm2)])
    plt.plot(harm[1])
    plt.plot(harm2,'--')
    plt.show()
    plt.plot(harm[1])
    plt.plot(harm2,'--')
    plt.show()

""" Sum of harmonics """
def task6():
    dt = 0.002
    N = 1000
    harm1 = my_harmonic(dt, N, A0 = 15, f0 = 3)
    harm2 = my_harmonic(dt, N, A0 = 100, f0 = 37)
    harm3 = my_harmonic(dt, N, A0 = 25, f0 = 137)
    harm_y = harm1[1]+harm2[1]+harm3[1]
    frq = fourier(harm_y)
    harm_after_ft = reverse_fourier(frq[1])
    plot_functions(\
    [(harm1[0], harm_y),\
    (generate_x(0, 1/(2*dt), 1/dt/N),\
     frq[0][0:len(harm_y) // 2]),(harm1[0],harm_after_ft)])

""" Read from file """
def task7():
    file_path = r"./data/pgp_1ms.dat"
    y = read_from_file(file_path, 1000)
    print(y)
    frq = fourier(y)
    harm_after_ft = reverse_fourier(frq[1])
    x = generate_x(0,len(y),1)
    N = 1000
    dt = 0.001
    plot_functions([(x, y),(generate_x(0, 1/(2*dt), 1/dt/N), frq[0][0:len(y) // 2]),(x,harm_after_ft)])

""" Auto coreliation of harmonics """
def task8():
    N = 1000
    dt = 0.001
    harm2 = my_harmonic(dt, N, A0=100, f0=37)
    corr = np.zeros(1000)
    for i in range(len(corr)):
        corr[i] = correlation(harm2[1], harm2[1], i)
    plt.plot(corr)
    plt.show()

""" Auto coreliation of polyharmonics """
def task8():
    file_path = r"./data/pgp_1ms.dat"
    y = np.array(read_from_file(file_path, 1000))
    print(y)
    corr = np.zeros(1000)
    for i in range(len(corr)):
        print(i)
        corr[i] = correlation(y, y, i)
    plt.plot(corr)
    plt.show()

""" Plotnost veroyatnosty harmonic """
def task9():
    N = 1000
    dt = 0.001
    harm2 = my_harmonic(dt, N, A0=100, f0=37)
    plt.hist(harm2[1],100)
    plt.show()

""" Spectr of my random and pikes """
def task10():
    y = add_shift(normalize_maxmin(defrandom(1000),1),0)
    y = np.zeros(1000)
    y = add_pikes(y,0.001,0.002,10,10)
    frq = fourier(y)
    x = generate_x(0,1000,1)
    plot_functions([(x, y), (x, frq[0])])

""" for trends plotnost veroyat, autocorrelia, fourie """
def task11():
    y = lineral(1.4,0)
    x = generate_x(0,1000,1)
    y = y(x)
    plt.hist(y,1000)
    plt.show()
    corr = np.zeros(1000)
    for i in range(len(corr)):
        corr[i] = correlation(y, y, i)
    plot_functions([(x,y),(x,corr), (x[:len(x)//2], fourier(y)[0][:len(x)//2])])

    y = exponential(0.0015, 0)
    x = generate_x(0, 1000, 1)
    y = y(x)
    plt.hist(y, 1000)
    plt.show()
    corr = np.zeros(1000)
    for i in range(len(corr)):
        corr[i] = correlation(y, y, i)
    plot_functions([(x, y), (x, corr), (x[:len(x) // 2], fourier(y)[0][:len(x) // 2])])

""" Summarizing all """
def task12():
    xs = [0,250,400,600,1000]
    k, b = count_sum(xs,[0,5000,2000,100,1800])
    x,y = calc_fun(funstar(k,b,xs))
    # plt.plot(y)
    N = 1000
    dt = 0.001
    y = add_pikes(#my_harmonic(dt + 0.002, N, A0=100, f0=20)[1]  \
                  + my_harmonic(dt, N, A0=50, f0=50)[1]  \
                  + y \
                  + my_harmonic(dt, N, A0=500, f0=4)[1],10,15,500,600) \
    + normalize_maxmin(defrandom(1000),50)


    print(check_stationarity(y,10))
    print(statistics(y))
    corr = correliation_shifts(y,y)
    frq = fourier(y)
    plot_functions([(x,y), (x,corr),
        (generate_x(0, 1/(2*dt), 1/dt/N), frq[0][0:len(y) // 2]),
        (y,100)],["func","autocorr", "fourie","hist"],
        ["plot", "plot","plot", "hist"])

""" on lesson test antidhift """
def task13():
    N = 1000
    dt = 0.001
    harm = my_harmonic(dt, N, A0=50, f0=50)
    y = add_shift(harm[1],40)
    x = harm[0]
    ft1 = fourier(y)
    y_s = anti_shift(y)
    ft2 = fourier(y_s)
    plot_functions([(x,y),(x[0:500], ft1[0][0:500]),(x,y_s),(x[0:500], ft2[0][0:500])])

""" test anti pikes """
def task14():
    N = 1000
    dt = 0.001
    y = normalize_maxmin(myrandom(1000),5)
    y = add_pikes(y,6,10,10,20)
    x = generate_x(0,1000,1)
    y_1 = anti_pikes(y, 5)
    plot_functions([(x,y),(x,y_1)])

    N = 1000
    dt = 0.001
    harm = my_harmonic(dt, N, A0=5, f0=50)
    y = add_pikes(harm[1],6,10,10,20)
    x = harm[0]
    y_1 = anti_pikes(y, 5)
    ft1 = fourier(y)
    ft2 = fourier(y_1)
    plot_functions([(x,y),(x[0:500], ft1[0][0:500]),(x,y_1),(x[0:500], ft2[0][0:500])])

""" test antitrend 2 """
def task15():
    xs = [0, 500, 1000]
    k, b = count_sum(xs,[0,50,20])
    x,y = calc_fun(funstar(k,b,xs))
    N = 1000
    dt = 0.001
    harm = my_harmonic(dt, N, A0=5, f0=50)
    rand = harm[1]#normalize_maxmin(defrandom(1000),5)
    y += rand
    ant_t = anti_trend21(y,40)
    y_1 = y - ant_t
    plot_functions([(x,y),(x,y_1),(x,ant_t),(x, rand)])

""" test antitrend 1 """
def task16():
    xs = [0, 500, 1000]
    k, b = count_sum(xs,[0,50,20])
    x,y = calc_fun(funstar(k,b,xs))
    rand = normalize_maxmin(defrandom(1000),5)
    y += rand
    y_1 = y - anti_trend1(y)
    plot_functions([(x,y),(x,y_1),(x, rand)])

""" all above shum """
def task17():
    xs = [0, 250, 400, 600, 1000]
    k, b = count_sum(xs, [0, 5000, 2000, 100, 1800])
    x, y = calc_fun(funstar(k, b, xs))
    # plt.plot(y)
    N = 1000
    dt = 0.001
    y = add_pikes(   # my_harmonic(dt + 0.002, N, A0=100, f0=20)[1] + \
                  my_harmonic(dt, N, A0=50, f0=50)[1] + \
                  y + \
                  my_harmonic(dt, N, A0=500, f0=4)[1], 10, 15, 500, 600) + \
        normalize_maxmin(defrandom(1000), 20)

    # print(check_stationarity(y,10))
    # print(statistics(y))
    corr = correliation_shifts(y, y)
    frq = fourier(y)
    plot_functions([(x, y), (x, corr),
                    (generate_x(0, 1/(2*dt), 1/dt/N),
                     frq[0][0:len(y) // 2]), (y, 100)],
                   ["func", "autocorr", "fourie", "hist"],
                   ["plot", "plot", "plot", "hist"], 'initial')

    y = y - anti_trend21(y, 30)

    corr = correliation_shifts(y, y)
    frq = fourier(y)
    plot_functions([(x, y), (x, corr), (generate_x(0, 1 / (2 * dt), 1 / dt / N), frq[0][0:len(y) // 2]), (y, 100)],
                   ["func", "autocorr", "fourie", "hist"], ["plot", "plot", "plot", "hist"], 'no trend')

    y = anti_pikes(y, y.mean()+1.5*y.std())
    corr = correliation_shifts(y, y)
    frq = fourier(y)
    plot_functions([(x, y), (x, corr), (generate_x(0, 1 / (2 * dt), 1 / dt / N), frq[0][0:len(y) // 2]), (y, 100)],
                   ["func", "autocorr", "fourie", "hist"], ["plot", "plot", "plot", "hist"], 'no pikes')

    y = anti_shift(y)
    corr = correliation_shifts(y, y)
    frq = fourier(y)
    plot_functions([(x, y), (x, corr), (generate_x(0, 1 / (2 * dt), 1 / dt / N), frq[0][0:len(y) // 2]), (y, 100)],
                   ["func", "autocorr", "fourie", "hist"], ["plot", "plot", "plot", "hist"], 'no shift')

    y = anti_random(y)
    corr = correliation_shifts(y, y)
    frq = fourier(y)
    plot_functions([(x, y), (x, corr), (generate_x(0, 1 / (2 * dt), 1 / dt / N), frq[0][0:len(y) // 2]), (y, 100)],
                   ["func", "autocorr", "fourie", "hist"], ["plot", "plot", "plot", "hist"], 'no random')

""" delete shum """
def task18():
    M = [1,2,3,10,50,100]
    rs = []
    sigmas = []
    for i in M:
        rs.append(np.zeros(1000))
        for j in range(i):
            rs[-1] += defrandom(1000)
        rs[-1] /= i
        plt.plot(rs[-1],label=str(i) + ', std:' + str(rs[-1].std()))
        sigmas.append(rs[-1].std())
    plt.legend()
    plt.show()
    plt.plot(M, sigmas)
    #plt.plot(M, np.sqrt(M))
    plt.title("standart deviation on amount of aggregation random")
    plt.show()
    M = [1,2,3,10,50,100]
    rs = []
    S = 5
    sigmas = []
    for i in M:
        rs.append(np.zeros(1000))
        for j in range(i):
            rs[-1] += normalize_maxmin(defrandom(1000),S) + my_harmonic(0.001, 1000, S/20, 5)[1]
        rs[-1] /= i
        plt.plot(rs[-1],label=str(i) + ', std:' + str(rs[-1].std()))
        sigmas.append(rs[-1].std())
    plt.legend()
    plt.show()
    plt.plot(M, sigmas)
    #plt.plot(M, np.sqrt(M))
    plt.title("standart deviation on amount of aggregation random")
    plt.show()

    S = 5
    M = [1,2,3,10,50,100]
    rs = []
    sigmas = []
    for i in M:
        rs.append([np.arange(1000),np.zeros(1000)])
        for j in range(i):
            rs[-1][1] += normalize_maxmin(defrandom(1000),S)
        rs[-1][1] /= i
        sigmas.append(rs[-1][1].std())
    plot_functions(rs, M)

    M = [1,2,3,10,50,100]
    rs = []
    sigmas = []
    for i in M:
        rs.append([np.arange(1000),np.zeros(1000)])
        for j in range(i):
            rs[-1][1] += normalize_maxmin(defrandom(1000),S) + my_harmonic(0.001, 1000, S/20, 5)[1]
        rs[-1][1] /= i
        sigmas.append(rs[-1][1].std())
    plot_functions(rs, M)

""" ubiranie 1 chastotyi """
def task19():
# 20 Nov
    dt = 0.001
    N = 1000
    harm =  my_harmonic(dt, N, A0=10, f0=50)[1] +  my_harmonic(dt, N, A0=10, f0=5)[1]
    filt = create_low_pass_filter(20, 128, 0.002)

    f1 = fourier(harm)[0]
    #f1 = f1[:len(f1)//2]
    c = convolutional(harm, filt)
    f2 = fourier(c)[0]
    #f2 = f2[:len(f2)//2]
    f3 = fourier(filt)[0]
    f3 = f3 * len(filt)
    plot_functions([(None,harm),  (None, c), (None, filt), (None, f1), (None, f2),  (None, f3) ],
                   ["sum of 2 harm", "filter apply result", "filter form", "fourie of 2 harm",
                                                            "fourie of result", "fourie of filter"])

""" test audio """
def task20():
    a, rate = read_audio(r"data\Recording.wav")
    print(len(a), rate)
    print(a)
    plt.plot(a)
    plt.show()
    a = a[int(input()):int(input())]
    # filt = create_band_pass_filter(300, 600, 128, 1 / rate)
    # #filt1 = create_band_stop_filter(400, 400, 64, 1/rate)
    # con = convolutional(a, filt)
    # con = convolutional(con, filt1)
    # f1 = fourier(a)[0]
    # ff1 = fourier(filt)[0]
    # # ff2 = fourier(filt1)[0]
    # f2 = fourier(con)[0]
    dt = 1 / rate
    # N = len(f1)*2
    # plot_functions([(None,a), (None,con),
    #                 (generate_x(0, 1 / (2* dt), 1 / dt / (len(f1)*2)),f1),
    #                 (generate_x(0, 1 / (2* dt), 1 / dt / (len(f2)*2)),f2),
    #                 # (generate_x(0, 1 / (2* dt), 1 / dt / (len(ff1)*2)), ff1),
    #                 (generate_x(0, 1 / (2* dt), 1 / dt / (len(ff1)*2)), ff1)])
    # print(len(a))
    #a[50000:150000] = np.zeros(len(a[50000:150000]))
    #
    print(rate)
    save_audio(a, rate, r'data\start.wav')
    plt.show()

""" show all filters """
def task21():
    width = 128
    dt = 0.002
    N = width*2+1
    lpw = create_low_pass_filter(15, width, dt)
    hpw = create_hight_pass_filter(15, width, dt)
    bpw = create_band_pass_filter(15, 30, width, dt)
    bsw = create_band_stop_filter(15,30, width, dt)
    f_lpw = fourier(lpw)[0] * (2*width+1)
    f_hpw = fourier(hpw)[0]* (2*width+1)
    f_bpw = fourier(bpw)[0]* (2*width+1)
    f_bsw = fourier(bsw)[0]* (2*width+1)
    plot_functions([(None, lpw), (None, hpw), (None, bpw), (None, bsw),
    (generate_x(0, 1 / (2*dt), 1 / dt / N),f_lpw),(generate_x(0, 1 / (2*dt), 1 / dt / N), f_hpw), (generate_x(0, 1 / (2*dt), 1 / dt / N), f_bpw), (generate_x(0, 1 / (2*dt), 1 / dt / N), f_bsw)],
                   ["lpw","hpw",'bpf','bsf', "f lpw","f hpw",'f bpf','f bsf'])

""" apply one after another filters"""
def task22():
    dt = 0.002
    N = 1000
    # harm = my_harmonic(dt, N, A0=10, f0=10)[1] + my_harmonic(dt, N, A0=50, f0=30)[1] \
    #        + my_harmonic(dt, N, A0=150, f0=40)[1] + my_harmonic(dt, N, A0=150, f0=20)[1] \
    #        + my_harmonic(dt, N, A0=50, f0=50)[1] + my_harmonic(dt, N, A0=10, f0=60)[1]
    harm = my_harmonic(dt, N, A0=10, f0=5)[1] + my_harmonic(dt, N, A0=10, f0=150)[1] \
           + my_harmonic(dt, N, A0=50, f0=50)[1]

    m = 128
    lpw = create_low_pass_filter(70, m, dt)
    harm1 = convolutional(harm, lpw)

    hpw = create_hight_pass_filter(15, m, dt)
    harm2 = convolutional(harm1, hpw)

    bpf = create_band_pass_filter(25, 70, m, dt)
    harm3 = convolutional(harm, bpf)

    bsf = create_band_stop_filter(25, 70, m, dt)
    harm4 = convolutional(harm, bsf)

    f = fourier(harm)[0]
    f1 = fourier(harm1)[0]
    f2 = fourier(harm2)[0]
    f3 = fourier(harm3)[0]
    f4 = fourier(harm4)[0]
    xf = generate_x(0, 1 / (2 * dt), 1 / dt / (len(f) * 2))

    plot_functions([(None, harm), (None, harm1), (None, harm2), (None, harm3), (None, harm4),
                    (xf, f), (xf, f1), (xf, f2), (xf, f3), (xf, f4)],
                   ["all", "with low filt", "with hight filt", "with bandw filt", "with stop b filt",
                    "fourie", "fourie", "fourie", "fourie", "fourie"])


""" Save recording with and without formants """
def task23():
    a, rate = read_audio(r"Recording.wav")
    plt.plot(a)
    plt.show()
    a = a[19000:28000]
    #filt = create_band_pass_filter(300, 600, 128, 1 / rate)
    # filt1 = create_band_stop_filter(400, 400, 64, 1/rate)
    #con = convolutional(a, filt)
    # con = convolutional(con, filt1)
    f0 = fourier(a)[0]
    f1 = fourier(a[:len(a)//2])[0]
    f2 = fourier(a[len(a)//2:])[0]
    dt = 1 / rate
    N = len(f1) * 2
    plot_functions([(None, a), (None, a[:len(a)//2]), (None, a[len(a)//2:]),
                    (generate_x(0, 1 / (2 * dt), 1 / dt / (len(f0) * 2)), f0),
                    (generate_x(0, 1 / (2 * dt), 1 / dt / (len(f1) * 2)), f1),
                    (generate_x(0, 1 / (2 * dt), 1 / dt / (len(f2) * 2)), f2)],
                   ["full", "pri", "vet", 'spek full', 'spek pri', 'spek vet'])

    # hpf = create_hight_pass_filter(1000, 128, dt)
    # pri_formamt = convolutional(a[:len(a)//2], hpf)
    # f_pri_formamt = fourier(pri_formamt)[0]
    # plot_functions([(None, a[:len(a)//2]), (None, pri_formamt),
    #                 (generate_x(0, 1 / (2 * dt), 1 / dt / (len(f1) * 2)), f1),
    #                 (generate_x(0, 1 / (2 * dt), 1 / dt / (len(f_pri_formamt) * 2)), f_pri_formamt)], ["pri", 'after con', 'spek', 'after con sprec'])

    # a[50000:150000] = np.zeros(len(a[50000:150000]))
    #
    bsf = create_band_stop_filter(1000, 3000, 24, dt)
    bpf = create_band_pass_filter(1000, 3000, 24, dt)
    con = convolutional(a, bsf)
    form = convolutional(a, bpf)
    f_con = fourier(con)[0]
    f_form = fourier(form)[0]
    plot_functions([(None, a), (None, con), (None, form),
                    create_print_pair_f(f0,dt),
                    create_print_pair_f(f_con, dt),
                    create_print_pair_f(f_form, dt)])
    save_audio(con * 100, rate, 'witout f1.wav')
    save_audio(a, rate, 'start.wav')
    save_audio(form * 100, rate, 'only f1.wav')


def task24():
    dt = 0.002
    sign = normalize_maxmin(defrandom(1000),2) + my_harmonic(f0=300,dt=dt)[1]
    corr = correliation_shifts(sign, sign)
    fou = fourier(corr)[0]
    plot_functions([(None, sign), (None, corr), create_print_pair_f(fou,dt)])

if __name__ == "__main__":
    pass
    task23()

    # rate = 60000
    # noise = defrandom(3 * rate) * 10000
    # dt = 1 / rate
    # main_thone = create_band_pass_filter(145, 155, 256, 1 / rate) * 5
    # f1 = create_band_pass_filter(550, 570, 256, 1 / rate) * 3
    # f2 = create_band_pass_filter(1100, 1500, 256, 1 / rate) * 2
    # f3 = create_band_pass_filter(1600, 2000, 256, 1 / rate)
    # # fou0 = fourier(main_thone*f1*f2*f3)[0]
    # # plot_functions([fou_pair(fou0, dt)])
    # con = convolutional(noise, main_thone*f1*f2*f3)
    # f = fourier(con)[0]
    # plot_functions([(None, con), fou_pair(f, dt)])
    # save_audio(normalize_maxmin(con,1), rate, "my_A.wav")
# берем шум, апплаим на фильтры - получаем фонему


#
# примеры курсача
# экономика
# амплитудная моделяция + шумы?
# частотная модуляция + шумы?
# восстановить от шумов и трендов и тд
# лодка передает сигнал пакетами с шумами
# диагностика двигателей 4 датчика, циклический процесс каждого циллиндра. ПЕРИОДОГРАММЫ
# плюс - минус одинаковая частота двигателей, амплитудный спектр, фазовый спектр (arct(Im/Re))
# моделирование вибро-аккустического сигнала поезда. Возрастание в нужной полосе
# мониторинг приближающихся-удаляющихся обхектов
# подавление тональных помех в аккустических сигналах. Например сымитировать гудки. Не замтны в общем, но заметны в спектре
# моделирование и подавление эффекта ревербирации в аккустических сигналах






#

#
