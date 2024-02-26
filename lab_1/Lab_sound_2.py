import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack as fft


# uwzględnić generowanie dla różnych fsize
def plotAudio(Signal, fs, TimeMargin=[0, 0.02], fsize=2**8):
    fig, axs = plt.subplots(2, 1)
    x = np.arange(len(Signal)) / fs
    axs[0].plot(x, Signal)
    axs[0].set_title("Audio Signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(TimeMargin)
    axs[0].grid()

    fsize = 2**8
    spectrum = fft.fft(Signal, fsize)
    spectrum_dB = 20 * np.log10(np.abs(spectrum[: fsize // 2]))
    x_freqs = np.arange(0, fs / 2, fs / fsize)
    axs[1].plot(x_freqs, spectrum_dB)
    axs[1].set_title("Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid()

    plt.show()


if __name__ == "__main__":
    # Load the sound
    data, fs = sf.read("SOUND_SIN/sin_440Hz.wav")

    # Plot the sound
    plotAudio(data, fs)
