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


# zmienic
# def quantize(data, bit):
#     res = data
#     start = -1
#     end = 1
#     q_range = 2**bit - 1

#     res = (res - start) / (end - start)
#     res = np.round(res * q_range) / q_range
#     res = (res * (end - start)) + start
#     return res


def muLaw_encode(data, mu=255):
    data = data.copy()
    data = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(1 + mu)

    return data


def muLaw_decode(data, mu=255):
    data = data.copy()
    data = np.sign(data) * (1 / mu) * (np.power(1 + mu, np.abs(data)) - 1)

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
    # return np.mean(X)
    # return X[-1]


def DPCM_predict(data, bit, predictor, n):
    y = DPCM_predict_encode(data, bit, predictor, n)
    y = DPCM_predict_decode(y, predictor, n)
    return y


def MuLaw(data, bits=6, mu=255):
    data = muLaw_encode(data, mu)
    data = quantize(data, bits)
    data = muLaw_decode(data, mu)
    return data


def draw_plot(titles, x, y, xlim=(-1, 1), margin=0.1):
    fig, axs = plt.subplots(len(titles), 1, figsize=(15, 10))

    for i, title in enumerate(titles):
        axs[i].plot(x, y[i])
        axs[i].set_title(title)

    for ax in axs.flatten():
        ax.set_xlim([xlim[0], xlim[1]])
        current_xlim = ax.get_xlim()
        new_xlim = (current_xlim[0] - margin, current_xlim[1] + margin)
        ax.set_xlim(new_xlim)

    return fig


def save_plot(fig, filename):
    fig.savefig(filename, dpi=600)


def main_comparison():
    x = np.linspace(-1, 1, 1000)
    y = 0.9 * np.sin(np.pi * x * 4)

    bits = range(2, 9)
    margin = 0.1
    for _bits in bits:
        y_muLaw_compressed = MuLaw(y, _bits)
        y_dpcm_compressed = DPCM_predict(y, _bits, predictor, 6)

        fig = draw_plot(
            [
                "Original Signal",
                "Mu-Law compressed Signal",
                "DPCM (predict) compressed  Signal",
            ],
            x,
            [y, y_muLaw_compressed, y_dpcm_compressed],
            xlim=[-1, 1],
            margin=margin,
        )

        # plt.show()
        save_plot(fig, f"OUTPUT/plot_{_bits}bits.png")

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(x, y, label="Original Signal")
        ax.plot(x, y_muLaw_compressed, label="Mu-Law compressed Signal")
        ax.plot(x, y_dpcm_compressed, label="DPCM (predict) compressed  Signal")
        ax.legend()
        ax.set_xlim([-1, 1])
        current_xlim = ax.get_xlim()
        new_xlim = (current_xlim[0] - margin, current_xlim[1] + margin)
        ax.set_xlim(new_xlim)
        save_plot(fig, f"OUTPUT/x_plot_{_bits}bits.png")


def main_sing():
    file, fs = sf.read("SING/sing_high1.wav")

    bits = 2
    n = 6

    filem_muLaw = MuLaw(file, bits)
    filem_DPCM = DPCM_predict(file, bits, predictor, n)

    # sd.play(file, fs)
    # sd.wait()

    sd.play(filem_muLaw, fs)
    sd.wait()

    sd.play(filem_DPCM, fs)
    sd.wait()


if __name__ == "__main__":
    # main_comparison()
    main_sing()

    # x = np.linspace(-1, 1, 1000)
    # y = 0.9 * np.sin(np.pi * x * 4)

    # y_muLaw_encoded = muLaw_encode(y)
    # y_quantized = quantize(y_muLaw_encoded, 6)
    # y_muLaw_decoded = muLaw_decode(y_quantized)

    # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    # axs[0].plot(x, y)
    # axs[0].set_title("Original Signal")
    # axs[1].plot(x, y_muLaw_encoded)
    # axs[1].set_title("Mu-Law Encoded Signal")
    # axs[2].plot(x, y_muLaw_decoded)
    # axs[2].set_title("Mu-Law Decoded Signal")
    # axs[0].plot(x, y)
    # axs[0].set_title("Original Signal")
    # axs[1].plot(x, y_muLaw_encoded)
    # axs[1].set_title("Mu-Law Encoded Signal")
    # axs[2].plot(x, y_muLaw_decoded)
    # axs[2].set_title("Mu-Law Decoded Signal")

    # print(np.max(y), np.min(y), np.max(y_muLaw_decoded), np.min(y_muLaw_decoded))

    # for ax in axs.flatten():
    #     ax.set_xlim([0, 1])

    # plt.show()

    # y_DPCM_encoded = DPCM_predict_encode(y, 6, predictor, 3)
    # y_DPCM_decoded = DPCM_predict_decode(y_DPCM_encoded, predictor, 3)

    # fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    # axs[0].plot(x, y)
    # axs[0].set_title("Original Signal")
    # axs[1].plot(x, y_DPCM_encoded)
    # axs[1].set_title("DPCM Encoded Signal")
    # axs[2].plot(x, y_DPCM_decoded)
    # axs[2].set_title("DPCM Decoded Signal")

    # for ax in axs.flatten():
    #     ax.set_xlim([0, 1])

    # def encode_DPCM(data, bit):
    #     res = data.copy()
    #     E = res[0]
    #     for x in range(1, data.shape[0]):
    #         diff = data[x] - E
    #         diff = quantize(diff, 16)
    #         res[x] = diff
    #         E += diff
    #     return res

    # def decode_DPCM(data):
    #     res = data.copy()
    #     for x in range(1, data.shape[0]):
    #         res[x] = res[x - 1] + data[x]
    #     return res

    # def DPCM_compression(data, bit):
    #     res = encode_DPCM(data, bit)
    #     res = decode_DPCM(res)
    #     return res

    # X = np.array([15,16,20,14,5,10,15,13,11,7,10,11,20,1,23])

    # y = DPCM_predict(X, 7, predictor, 3)
    # print(y)
    # y = DPCM_predict_decode(y, predictor, 3)
    # print(y)

    # plt.show()
