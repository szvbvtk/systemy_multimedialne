import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack as fft

# Read the data from file
# data is the sound data, fs is the sampling frequency
data, fs = sf.read("SOUND_INTRO/sound1.wav", dtype="float32")


print(f"Dtype: {data.dtype}\nShape: {data.shape}")

# Play the sound
# sd.play(data, fs)

# Wait until the sound is finished
# status = sd.wait()


# left channel, right channel, and mix channel
sound_left_channel = data[:, 0]
sound_right_channel = data[:, 1]
sound_mix_channel = np.mean(data, axis=1)

# Save the sound to file
sf.write('sound_L.wav', sound_left_channel, fs)
sf.write('sound_R.wav', sound_right_channel, fs)
sf.write('sound_mix.wav', sound_mix_channel, fs)

# Plot the sound
# fig, ax = plt.subplots(3, 1)
# x = np.arange(len(sound_left_channel)) / fs

# ax[0].plot(x, sound_left_channel)
# ax[0].set_title("Left Channel")
# ax[0].set_xlabel("Time (s)")
# ax[0].set_ylabel("Amplitude")

# ax[1].plot(x, sound_right_channel)
# ax[1].set_title("Right Channel")
# ax[1].set_xlabel("Time (s)")
# ax[1].set_ylabel("Amplitude")

# ax[2].plot(x, sound_mix_channel)
# ax[2].set_title("Mix Channel")
# ax[2].set_xlabel("Time (s)")
# ax[2].set_ylabel("Amplitude")

# plt.show()

# Plot the spectrum
fsize = 2**8
fig, ax = plt.subplots(3, 1)
yf_left = fft.fft(sound_left_channel, fsize)
yf_right = fft.fft(sound_right_channel, fsize)
yf_mix = fft.fft(sound_mix_channel, fsize)

x = np.arange(0, fs/2, fs/fsize)

ax[0].plot(x, 20*np.log10(np.abs(yf_left[:fsize//2])))
ax[0].set_title("Left Channel")
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Amplitude (dB)")

ax[1].plot(x, 20*np.log10(np.abs(yf_right[:fsize//2]) + 1e-10))
ax[1].set_title("Right Channel")
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude (dB)")

ax[2].plot(x, 20*np.log10(np.abs(yf_mix[:fsize//2])))
ax[2].set_title("Mix Channel")
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Amplitude (dB)")

plt.show()

print(sound_right_channel)
