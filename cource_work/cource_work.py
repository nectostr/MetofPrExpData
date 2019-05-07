from model import *
from in_out import *
from analysys import *
import time
import sounddevice as sd
sd.default.device = 7
def fourier_print(f, dt):
    return generate_x(0, 1 / (2*dt), 1 /(2 * dt * len(f))), f

def get_shift(arct, fr):
    real_shift = arct / np.pi
    needed_shift_pi = ((2.5 - real_shift / np.pi) % 2) + real_shift
    print(needed_shift_pi)
    opp = needed_shift_pi / (dt * fr)
    return opp

""" load audio"""
sound, rate = read_audio("data/start.wav")
length = len(sound)#rate * 0.5
dt = 1 / rate
print("Rate: {}, Descr step: {}".format(rate, dt))

""" create all possible functions """
notes_freqs = [465.777, 440.8888, 414.222, 400, 391.111, 369.7777,
               348.444, 330.666, 311.11, 293.333, 277.33, 96]
notes_arr = [my_harmonic(dt, length, A0=max(sound), f0=i, opp=0)[1] for i in notes_freqs]
note = 4#np.random.randint(0,len(notes_arr))

""" create sound with noises """
# sound_with_notes_noise = sound.copy()
# for i in notes_arr:
#     sound_with_notes_noise += i

sound_with_notes_noise = sound + notes_arr[0] #+ notes_arr[-1] + notes_arr[len(notes_arr)//2]
""" fourie transform for nosise data"""
fou, temp, Im_f, Re_f = fourier(sound_with_notes_noise, ret_phaze=True)

""" detect noise freq """
maxxs = []
phazes = []
window = 200
print("std {}, mean {}, m+4s {}".format(fou.std(), fou.mean(), fou.mean() + 4*fou.std()))
for i in range(1,len(fou)-1):
    if (fou[i] - fou.mean() > 5 * fou.std())\
            and (fou[i] > fou[i-1]) and (fou[i] > fou[i+1]):
        print(i, fou[i])
        maxxs.append(i)
        arc = np.arctan2(Re_f[i], Im_f[i])
        phazes.append(arc if arc > 0 else -arc)


print(maxxs)
print(phazes)
""" cope with freq in more than one point problem"""
i = 0
while i < len(maxxs)-1:
    if maxxs[i+1] - 1 == maxxs[i]:
        avg = (maxxs[i+1] + maxxs[i])/2
        maxxs[i] = avg
        phazes[i] = (phazes[i+1] + phazes[i])/2
        del maxxs[i+1]
        del phazes[i+1]
    i += 1
maxxs = np.array(maxxs)
print(maxxs)
print(phazes)

""" deletening audio with summarizing opposit waves"""
to_delete = np.zeros(len(sound_with_notes_noise))
for ind, i in enumerate(maxxs):
    f = i * (1/dt/(len(fou)*2))
    to_delete += my_harmonic(dt, length, A0=fou[i]*2, #A0=max(sound),
                              f0=f, opp=get_shift(phazes[ind],f))[1]
    print(fou[int(i)-5:int(i)+5].max(), max(sound), f)
delete_noise = sound_with_notes_noise + to_delete

""" create dourie transform to check result """
fou1 = fourier(delete_noise)[0]
plot_functions([fourier_print(fou, dt), fourier_print(fou1, dt)],
               ["Фурье до подавления", "Фурье после подавления"])


""" printing differense """
plot_functions([(None, sound), (None, delete_noise), (None, sound - delete_noise)],
               ["Изначальный файл", "Файл после подавления", "Разница между ними"])

""" save results"""
save_audio(sound_with_notes_noise * 10000 , rate, r"data\tonenoise.wav")
save_audio(delete_noise*10000, rate, r"data\deletingnoise.wav")