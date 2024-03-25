import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.interpolate import interp1d


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


def decimate(signal, factor):
    return signal[::factor]


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

    return y, fs1


if __name__ == "__main__":
    sig = np.round(np.linspace(0, 255, 255, dtype=np.uint8))

    sig_quantized = quantize(sig, 2)

    plt.plot(sig_quantized)
    plt.grid(True)
    plt.show()
