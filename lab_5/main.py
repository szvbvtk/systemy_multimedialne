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

    y = f(t1).astype(signal.dtype)
    fs1 = N1 / (N / fs)

    return y, int(fs1)


def plotAudio(Signal, fs, fsize, axs, TimeMargin=[0, 0.02]):

    x_time = np.arange(len(Signal)) / fs
    x_frequency = np.arange(0, fs / 2, fs / fsize)
    spectrum_halved = fft.fft(Signal, fsize)[: fsize // 2]
    spectrum_dB = 20 * np.log10(np.abs(spectrum_halved) + np.finfo(np.float32).eps)

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

    max_amplitude_index = np.argmax(spectrum_dB)
    peak_amplitude = spectrum_dB[max_amplitude_index]
    peak_frequency = x_frequency[max_amplitude_index]

    return peak_frequency, peak_amplitude


def main_quantization():
    dir = "SIN/"
    filenames = ["sin_60Hz.wav", "sin_440Hz.wav", "sin_8000Hz.wav", "sin_combined.wav"]
    output_dir = "OUTPUT/quantization"

    filename = filenames[0]
    sig, fs = sf.read(dir + filename)

    for bits in (4, 8, 16, 24):
        sig_quantized = quantize(sig, bits)

        fig, axs = plt.subplots(2, 1, figsize=(10, 9))
        # fig.tight_layout(pad=1)
        _, _ = plotAudio(sig_quantized, fs, 1024, axs, TimeMargin=[0, 0.1])

        plt.suptitle(f"Quantization: {bits} bits")
        # plt.show()
        plt.savefig(
            f"{output_dir}/{filename[:-4]}_{bits}bits.png", dpi=300, format="png"
        )


def main_decimation():
    dir = "SIN/"
    filenames = ["sin_60Hz.wav", "sin_440Hz.wav", "sin_8000Hz.wav", "sin_combined.wav"]
    output_dir = "OUTPUT/decimation"

    filename = filenames[3]
    sig, fs = sf.read(dir + filename)

    for step in (2, 4, 6, 10, 24):
        sig_decimated, fs1 = decimate(sig, step, fs)

        fig, axs = plt.subplots(2, 1, figsize=(10, 9))
        # print(fs, fs1)
        _, _ = plotAudio(sig_decimated, fs1, 1024, axs, TimeMargin=[0, 0.001])

        plt.suptitle(f"Decimation: {step}x")

        # plt.show()

        plt.savefig(f"{output_dir}/{filename[:-4]}_{step}x.png", dpi=300, format="png")


def main_interpolation():
    dir = "SIN/"
    filenames = ["sin_60Hz.wav", "sin_440Hz.wav", "sin_8000Hz.wav", "sin_combined.wav"]
    output_dir = "OUTPUT/interpolation"

    filename = filenames[3]

    sig, fs = sf.read(dir + filename)

    for _kind in ("linear", "cubic"):
        for _fs in (2000, 4000, 8000, 11999, 16000, 16953, 24000, 41000):
            sig_interpolated, fs1 = interpolate(sig, fs, _fs, kind=_kind)

            fig, axs = plt.subplots(2, 1, figsize=(10, 9))
            _, _ = plotAudio(sig_interpolated, fs1, 1024, axs, TimeMargin=[0, 0.01])

            plt.suptitle(f"Interpolation: {fs}Hz -> {_fs}Hz")
            # plt.show()

            plt.savefig(
                f"{output_dir}/{filename[:-4]}_{_kind}_{_fs}Hz.png",
                dpi=300,
                format="png",
            )


def main_2():
    sig, fs = sf.read("SING/sing_high1.wav")
    print(fs)
    # sig = quantize(sig, 4)
    # sig, fs = decimate(sig, 24, fs)

    # sd.play(sig, fs)
    # sd.play(sig, fs)
    # sd.wait()
    sig, fs = interpolate(sig, fs, 8000, kind="linear")

    sd.play(sig, fs)
    sd.wait()


if __name__ == "__main__":
    # main_quantization()
    # main_decimation()
    # main_interpolation()

    main_2()
    pass
