import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.fftpack as fft


def plotAudio(Signal, Fs, TimeMargin=[0, 0.02]):
    fig, axs = plt.subplots(2, 1)

    x_time = np.arange(len(Signal)) / Fs
    x_frequency = np.arange(0, Fs / 2, Fs / len(Signal))

    spectrum_halved = fft.fft(Signal)[: len(Signal) // 2]
    spectrum_dB = 20 * np.log10(np.abs(spectrum_halved))

    axs[0].plot(x_time, Signal)
    axs[0].set_title("Audio Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(TimeMargin)
    axs[0].grid()

    axs[1].plot(x_frequency, spectrum_dB)
    axs[1].set_title("Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid()

    plt.subplots_adjust(hspace=0.5)

    plt.show()
    # plt.savefig("zadanie_2.png", dpi=300)


if __name__ == "__main__":
    # Load the sound
    data, fs = sf.read("SOUND_SIN/sin_440Hz.wav")

    # Plot the sound
    plotAudio(data, fs)
