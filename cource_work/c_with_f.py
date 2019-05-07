from model import *
from in_out import *
from analysys import *
import time
import msvcrt
import time
import sounddevice as sd
sd.default.device = 7
def fourier_print(f, dt):
    return generate_x(0, 1 / (2*dt), 1 /(2 * dt * len(f))), f

""" load audio"""
sound, rate = read_audio("data/start.wav")
length = len(sound)#rate * 0.5
dt = 1 / rate
print("Rate: {}, Descr step: {}".format(rate, dt))

""" create all possible functions """
notes_freqs = [400, 466, 440, 415, 392, 370,
               349, 330, 311, 294, 277, 200]
notes_arr = [my_harmonic(dt, length, A0=max(sound), f0=i, opp=0)[1] for i in notes_freqs]
note = 4#np.random.randint(0,len(notes_arr))

""" create sound with noises """
sound_with_notes_noise = sound.copy() + notes_arr[0] + notes_arr[-1] #+ notes_arr[len(notes_arr)//2]

""" fourie transform for nosise data"""
fou, temp, Im_f, Re_f = fourier(sound_with_notes_noise.copy(), ret_phaze=True)

""" detect noise freq """
maxxs = []
phazes = []
print("std {}, mean {}, m+4s {}".format(fou.std(), fou.mean(), fou.mean() + 4*fou.std()))
for i in range(1,len(fou)-1):
    if (fou[i] - fou.mean() > 5 * fou.std())\
            and (fou[i] > fou[i-1]) and (fou[i] > fou[i+1]):
        print(i, fou[i])
        maxxs.append(i)
        arc = np.arctan2(Re_f[i], Im_f[i])
        phazes.append(arc if arc > 0 else -arc)

""" cope with freq in more than one point problem"""
i = 0
while i < len(maxxs)-1:
    if maxxs[i+1] - 1 == maxxs[i]:
        avg = (maxxs[i+1] + maxxs[i])/2
        maxxs[i] = avg
        del maxxs[i+1]
    i += 1
maxxs = np.array(maxxs)
print(maxxs)
print(phazes)

""" deletening audio with bsw of filter"""
delete_noise = sound_with_notes_noise.copy()
for i in maxxs:
    f = i * (1 / dt / (len(fou) * 2))
    bsw = create_band_stop_filter(f-10, f+10, 2048, dt)
    delete_noise = convolutional(delete_noise, bsw)

""" create fourie transform to check result """
fou1 = fourier(delete_noise.copy())[0]
fou2 = fourier(bsw)[0]
plot_functions([fourier_print(fou, dt), fourier_print(fou2, dt), fourier_print(fou1,dt)],
               ["Исходное распр Фурье", "Распределение фурье фильтра", 'Итоговое распределение Фурье'])

""" printing differense """
plot_functions([(None, normalize_maxmin(sound,1)), (None, normalize_maxmin(delete_noise,1)),
                (None, normalize_maxmin(sound,1)-normalize_maxmin(delete_noise,1))],
               ["Изначальный файл", "Файл после подавления", "Разница между ними"])


""" save results"""
save_audio(sound_with_notes_noise * 10000 , rate, r"data\tonenoise.wav")
save_audio(delete_noise*10000, rate, r"data\deletingnoise.wav")