import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d
import scipy.fftpack as fft


def quantize(signal, bits):

    if bits < 2 or bits > 32:
        raise ValueError("Number of bits must be in range [2, 32]")

    data = signal.copy()

    if np.issubdtype(data.dtype, np.integer):
        m = np.iinfo(data.dtype).min
        n = np.iinfo(data.dtype).max
    elif np.issubdtype(data.dtype, np.floating):
        m = -1
        n = 1

    # nie wiem skąd sie bierze to d
    d = 2**bits - 1

    data = data.astype(float)
    data = data - m
    data = data / (n - m)
    data = data * d
    data = np.round(data)
    data = data / d
    data = data * (n - m)
    data = data + m
    data = data.astype(signal.dtype)

    return data


def decimate(signal, factor, fs):
    fs1 = fs / factor
    return signal[::factor], int(fs1)


def interpolate(signal, fs, N1, kind="linear"):
    N = len(signal)
    t = np.linspace(0, N / fs, N)
    t1 = np.linspace(0, N / fs, N1)

    if kind == "linear":
        f = interp1d(t, signal, kind="linear")
    elif kind == "cubic":
        f = interp1d(t, signal, kind="cubic")
    else:
        raise ValueError("Invalid interpolation type")

    y = f(t1)
    fs1 = N1 / (N / fs)

    return y, int(fs1)


def plotAudio(Signal, fs, fsize, axs, TimeMargin=[0, 0.02]):

    x_time = np.arange(len(Signal)) / fs
    x_frequency = np.arange(0, fs / 2, fs / fsize)
    spectrum_halved = fft.fft(Signal, fsize)[: fsize // 2]
    spectrum_dB = 20 * np.log10(np.abs(spectrum_halved))

    axs[0].scatter(x_time, Signal)
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

    max_amplitude_index = np.argmax(spectrum_dB)
    peak_amplitude = spectrum_dB[max_amplitude_index]
    peak_frequency = x_frequency[max_amplitude_index]

    return peak_frequency, peak_amplitude


if __name__ == "__main__":
    # sig = np.round(np.linspace(0, 255, 255, dtype=np.uint8))

    # sig_quantized = quantize(sig, 2)

    # plt.plot(sig_quantized)
    # plt.grid(True)
    # plt.show()

    sig, fs = sf.read("SIN/sin_60Hz.wav")
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    sig_decimated, fs1 = decimate(sig, 3, fs)
    sig_interpolated, fs2 = interpolate(sig, fs, len(sig) - 40000, kind="linear")

    print(fs, fs1, fs2)

    peak_frequency, peak_amplitude = plotAudio(sig, fs, 1024, axs)
    peak_frequency, peak_amplitude = plotAudio(sig_interpolated, fs2, 1024, axs)
    # peak_frequency, peak_amplitude = plotAudio(sig_decimated, fs1, 1024, axs)

    fig.suptitle(f"Rozmiar okna FFT: {1024}")
    fig.tight_layout(pad=1.5)
    plt.show()



    
