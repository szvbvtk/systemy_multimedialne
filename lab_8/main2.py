import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.fftpack
import numpy as np
import cv2


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class ver1:
    Y = np.array([])
    Cb = np.array([])
    Cr = np.array([])
    ChromaRatio = "4:4:4"
    QY = np.ones((8, 8))
    QC = np.ones((8, 8))
    shape = (0, 0, 3)


QY = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

QC = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)

QN = np.ones((8, 8))


def dct2(block):
    return scipy.fftpack.dct(
        scipy.fftpack.dct(block.astype(float), axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )


def idct2(block):
    return scipy.fftpack.idct(
        scipy.fftpack.idct(block.astype(float), axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )


def quantize(block, Q):
    return np.round(block / Q).astype(int)


def dequantize(block, Q):
    return block * Q


def zigzag(A):
    template = np.array(
        [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
    )
    if len(A.shape) == 1:
        B = np.zeros((8, 8))
        for r in range(0, 8):
            for c in range(0, 8):
                B[r, c] = A[template[r, c]]
    else:
        B = np.zeros((64,))
        for r in range(0, 8):
            for c in range(0, 8):
                B[template[r, c]] = A[r, c]
    return B


def chromaSubsample(channel, Ratio="4:4:4"):
    if Ratio == "4:4:4":
        return channel
    elif Ratio == "4:2:2":
        return channel[:, ::2]
    else:
        raise ValueError("Invalid Chroma Subsampling Ratio")


def chromaResample(channel, Ratio="4:4:4"):
    # sprobowac zmienic na np.repeat jesli bede mial czas, nie jest to wymagane
    if Ratio == "4:4:4":
        return channel
    elif Ratio == "4:2:2":
        rows, cols = channel.shape
        output = np.empty((rows, cols * 2))

        for row in range(rows):
            for col in range(cols):
                output[row, [col * 2, col * 2 + 1]] = channel[row, col]

        return output
    else:
        raise ValueError("Invalid Chroma Resampling Ratio")


def CompressBlock(block, Q):
    pass


def CompressLayer(L, Q):
    S = np.array([])
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w : (w + 8), k : (k + 8)]
            S = np.append(S, CompressBlock(block, Q))


def compress_image(img_rgb, QY=np.ones((8, 8)), QC=np.ones((8, 8)), Ratio="4:4:4"):
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    JPEG = ver1()
    JPEG.Y = img_ycrcb[:, :, 0]
    JPEG.Cr = img_ycrcb[:, :, 1]
    JPEG.Cb = img_ycrcb[:, :, 2]
    JPEG.ChromaRatio = Ratio
    JPEG.QY = QY
    JPEG.QC = QC
    JPEG.shape = img_rgb.shape

    JPEG.Cr = chromaSubsample(JPEG.Cr, JPEG.ChromaRatio)
    JPEG.Cb = chromaSubsample(JPEG.Cb, JPEG.ChromaRatio)

    return JPEG


def decompress_image(JPEG):

    JPEG.Cr = chromaResample(JPEG.Cr, JPEG.ChromaRatio)
    JPEG.Cb = chromaResample(JPEG.Cb, JPEG.ChromaRatio)

    result_ycrcb = np.dstack([JPEG.Y, JPEG.Cr, JPEG.Cb]).clip(0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2RGB)

    return result_rgb


def main():
    img = read_image("IMG/1.jpg")
    img = img[100:356, 100:356, :]

    compressed = compress_image(img, QY, QC, "4:4:4")
    decompressed = decompress_image(compressed)

    print(np.array_equal(img, decompressed))

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(decompressed)
    plt.show()


if __name__ == "__main__":
    main()
