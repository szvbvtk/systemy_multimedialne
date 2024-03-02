import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

data, fs = sf.read("SOUND_INTRO/sound1.wav", dtype="float32")

# left channel, right channel, and monofonic sound
sound_left_channel = data[:, 0]
sound_right_channel = data[:, 1]
sound_mono = np.mean(data, axis=1)

# Save the sound to file
sf.write("sound_L.wav", sound_left_channel, fs)
sf.write("sound_R.wav", sound_right_channel, fs)
sf.write("sound_mix.wav", sound_mono, fs)

fig, ax = plt.subplots(3, 1)

# x (time in seconds) is the same for all channels
x = np.arange(len(sound_left_channel)) / fs

ax[0].plot(x, sound_left_channel)
ax[0].set_title("Left Channel")
ax[0].set_ylabel("Amplitude")

ax[1].plot(x, sound_right_channel)
ax[1].set_title("Right Channel")
ax[1].set_ylabel("Amplitude")

ax[2].plot(x, sound_mono)
ax[2].set_title("Mix Channel")
ax[2].set_ylabel("Amplitude")

plt.subplots_adjust(hspace=0.5)
plt.xlabel("Time (s)")

# plt.show()
plt.savefig("zadanie_1.png")
