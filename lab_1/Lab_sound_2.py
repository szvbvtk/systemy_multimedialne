import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack as fft


# uwzględnić generowanie dla różnych fsize
def plotAudio(Signal, fs, TimeMargin=[0, 0.02], fsize=[2**8, 2**12, 2**16]):
    fig, ax = plt.subplots(2, 1)
    x = np.arange(len(Signal)) / fs
    ax[0].plot(x, Signal)
    ax[0].set_title("Audio Signal")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlim(TimeMargin)

    fsize = 2**8
    yf = fft.fft(Signal, fsize)
    x = np.arange(0, fs / 2, fs / fsize)
    ax[1].plot(x, 2 / fsize * np.abs(yf[: fsize // 2]))
    ax[1].set_title("Spectrum")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Amplitude")
    # ax[1].set_xlim([0, 20000])

    plt.show()
