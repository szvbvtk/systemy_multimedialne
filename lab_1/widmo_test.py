import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack


data, fs = sf.read("SOUND_SIN/sin_440HZ.wav", dtype=np.int32)

fsize=2**8

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)
plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data,fsize)
# plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))

# dB scale
plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
plt.show()