from model import *
from in_out import *
from analysys import *
import msvcrt
import time
import sounddevice as sd
sd.default.device = 7

sound, rate = read_audio("data/start.wav")
length = rate * 0.5
dt = 1 / rate

notes_freqs = [493.88, 466.16, 440.00, 415.30, 392.00, 369.99,
               349.23, 329.63, 311.13, 293.66, 277.18, 261.63]
notes_base = {"B": 493.88, "A#": 466.16, "A": 440.00, "G#": 415.30, "G": 392.00, "F#": 369.99,
              "F": 349.23, "E": 329.63, "D#": 311.13, "D": 293.66, "C#": 277.18, "C": 261.63}
notes_arr = [my_harmonic(dt, length, A0=max(sound), f0=i)[1] for i in notes_freqs]
notes_arr_base = {list(notes_base.keys())[i]: notes_arr[i] for i in range(len(notes_base))}
notes = np.concatenate([notes_arr[i] for i in np.random.randint(0, len(notes_arr), 60)])


keys = ["B","A#", "A", "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C"]
keys_base = {'99': 'B', '102': 'A#', '118': 'A', '103': 'G#', '98': 'G', '110': 'F#', '106': 'F', '109': 'E', '107': 'D#', '44': 'D',
             '108': 'C#', '46': 'C'}

print("GOOO")
prints = 0
while prints < 1000:
    k = str(ord(msvcrt.getch()))
    try:
        to_play = notes_arr_base[keys_base[k]]
        sd.play(to_play * 1000, rate)
    except:
        print("Unknown: {}".format(k))

    time.sleep(0.1)
