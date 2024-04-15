import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.stats.mstats import gmean, hmean


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


def muLaw_encode(data, mu=255):
    # data = data.astype(float)
    data = data.copy()
    data = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(1 + mu)

    return data


def muLaw_decode(data, mu=255):
    data = data.copy()
    # data = data.astype(float)
    data = np.sign(data) * (1 / mu) * (np.power(1 + mu, np.abs(data) - 1))

    return data


def DPCM_predict_encode(data, bit, predictor, n):
    y = np.empty(data.shape)
    xp = np.empty(data.shape)
    e = 0

    for i in range(1, data.shape[0]):
        y[i] = quantize(data[i] - e, bit)
        xp[i] = y[i] + e

        idx = np.arange(i - n, i, 1, dtype=int) + 1
        idx = np.delete(idx, idx < 0)

        e = predictor(xp[idx])

    return y

def DPCM_predict_decode(data, predictor, n):
    y = data.copy()
    xp = data.copy()
    e = 0

    for i in range(1, data.shape[0]):
        y[i] = data[i] + e
        xp[i] = data[i] + e

        idx = np.arange(i - n, i, 1, dtype=int) + 1
        idx = np.delete(idx, idx < 0)

        e = predictor(xp[idx])

    return y


def predictor(X):
    return np.median(X)
    # return X[-1]


    

if __name__ == "__main__":
    x = np.linspace(-1, 1, 1000)
    y = 0.9 * np.sin(np.pi * x * 4)

    y_muLaw_encoded = muLaw_encode(y)
    y_quantized = quantize(y_muLaw_encoded, 6)
    y_muLaw_decoded = muLaw_decode(y_quantized)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[1].plot(x, y_muLaw_encoded)
    axs[1].set_title("Mu-Law Encoded Signal")
    axs[2].plot(x, y_muLaw_decoded)
    axs[2].set_title("Mu-Law Decoded Signal")
    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[1].plot(x, y_muLaw_encoded)
    axs[1].set_title("Mu-Law Encoded Signal")
    axs[2].plot(x, y_muLaw_decoded)
    axs[2].set_title("Mu-Law Decoded Signal")

    for ax in axs.flatten():
        ax.set_xlim([0, 1])

    plt.show()


    y_DPCM_encoded = DPCM_predict_encode(y, 6, predictor, 3)
    y_DPCM_decoded = DPCM_predict_decode(y_DPCM_encoded, predictor, 3)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(x, y)
    axs[0].set_title("Original Signal")
    axs[1].plot(x, y_DPCM_encoded)
    axs[1].set_title("DPCM Encoded Signal")
    axs[2].plot(x, y_DPCM_decoded)
    axs[2].set_title("DPCM Decoded Signal")

    for ax in axs.flatten():
        ax.set_xlim([0, 1])

    plt.show()
