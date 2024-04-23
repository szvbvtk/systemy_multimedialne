import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.fftpack
import numpy as np
import cv2


class ver1:
    Y = np.array([])
    Cb = np.array([])
    Cr = np.array([])
    ChromaRatio = "4:4:4"
    QY = np.ones((8, 8))
    QC = np.ones((8, 8))
    shape = (0, 0, 3)


def RLE_encode(img):
    shape = np.array([len(img.shape)])
    shape = np.concatenate([shape, img.shape])

    img = img.flatten()

    output = np.empty(np.prod(img.shape) * 2, dtype=int)
    j = 0
    count = 1

    print("RLE encoding...")
    for i in tqdm(range(1, len(img))):
        if img[i] == img[i - 1]:
            count += 1
        else:
            output[[j, j + 1]] = img[i - 1], count
            j += 2
            count = 1

    output[[j, j + 1]] = img[-1], count
    j += 2
    output = output[:j]

    output = np.concatenate([shape, output])
    return output


def RLE_decode(data):
    if data[0] == 2:
        shape = data[1:3]
        data = data[3:]
    elif data[0] == 3:
        shape = data[1:4]
        data = data[4:]
    else:
        raise ValueError("Invalid data")

    # copilot
    # shape = data[1 : data[0] + 1]

    output = np.empty(np.prod(shape), dtype=int)
    j = 0
    print("RLE decoding...")
    for i in tqdm(range(0, len(data), 2)):
        output[j : j + data[i + 1]] = data[i]
        j += data[i + 1]

    output = np.reshape(output, shape)

    return output


# copilot, prawdopodobnie dobrze
def chroma_subsampling(Cr, Cb, Ratio="4:4:4"):
    if Ratio == "4:4:4":
        return Cr, Cb
    elif Ratio == "4:2:2":
        return Cr[:, ::2], Cb[:, ::2]
    elif Ratio == "4:2:0":
        return Cr[::2, ::2], Cb[::2, ::2]


# dobrze
def dct2(block):
    return scipy.fftpack.dct(
        scipy.fftpack.dct(block.astype(float), axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )


# dobrze
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


def compress_block(L, Q):
    L -= 128
    D = dct2(L)
    Q = quantize(D, Q)
    Z = zigzag(Q)
    return Z


def decompress_block(L, _Q):
    Q = zigzag(L)
    D = dequantize(Q, _Q)
    L = idct2(D)
    L += 128
    return L


# to na pewno dobrze
def compress_layer(L, Q):
    S = np.array([])
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w : (w + 8), k : (k + 8)]
            S = np.append(S, compress_block(block, Q))

    return S


def decompress_layer(S, Q):
    # tu być może L jest złe
    L = np.zeros((Q.shape[0] * 8, Q.shape[1] * 8))

    for idx, i in enumerate(range(0, S.shape[0], 64)):
        vector = S[i : (i + 64)]
        m = L.shape[0] / 8
        k = int((idx % m) * 8)
        w = int((idx // m) * 8)
        L[w : (w + 8), k : (k + 8)] = decompress_block(vector, Q)

    return L


def JPEG_compress(RGB, Ratio="4:4:4", QY=np.ones((8, 8)), QC=np.ones((8, 8))):
    YCrCb = cv2.cvtColor(RGB, cv2.COLOR_BGR2YCrCb).astype(int)

    JPEG = ver1()
    JPEG.QY = QY
    JPEG.QC = QC
    JPEG.shape = YCrCb.shape
    JPEG.ChromaRatio = Ratio

    JPEG.Y = YCrCb[:, :, 0]
    JPEG.Cr = YCrCb[:, :, 1]
    JPEG.Cb = YCrCb[:, :, 2]

    JPEG.Cr, JPEG.Cb = chroma_subsampling(JPEG.Cr, JPEG.Cb, Ratio)

    JPEG.Y = compress_layer(JPEG.Y, QY)
    JPEG.Cr = compress_layer(JPEG.Cr, QC)
    JPEG.Cb = compress_layer(JPEG.Cb, QC)

    JPEG.Y = RLE_encode(JPEG.Y)
    JPEG.Cr = RLE_encode(JPEG.Cr)
    JPEG.Cb = RLE_encode(JPEG.Cb)

    return JPEG


# copilot, do sprawdzenia calosc
def chroma_resampling(Cr, Cb, Ratio="4:4:4"):
    if Ratio == "4:4:4":
        return Cr, Cb
    elif Ratio == "4:2:2":
        Cr = np.repeat(Cr, 2, axis=1)
        Cb = np.repeat(Cb, 2, axis=1)
        return Cr, Cb
    elif Ratio == "4:2:0":
        # najpier wiersze potem kolumny, sprawdzic czy axis sie zgadzają
        Cr = np.repeat(Cr, 2, axis=0)
        Cr = np.repeat(Cr, 2, axis=1)
        Cb = np.repeat(Cb, 2, axis=0)
        Cb = np.repeat(Cb, 2, axis=1)
        return Cr, Cb


def JPEG_decompress(JPEG):
    JPEG.Y = RLE_decode(JPEG.Y)
    JPEG.Cr = RLE_decode(JPEG.Cr)
    JPEG.Cb = RLE_decode(JPEG.Cb)

    Y = decompress_layer(JPEG.Y, JPEG.QY)
    Cr = decompress_layer(JPEG.Cr, JPEG.QC)
    Cb = decompress_layer(JPEG.Cb, JPEG.QC)

    Cr, Cb = chroma_resampling(Cr, Cb, JPEG.ChromaRatio)

    YCrCb = np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)
    RGB = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)

    return RGB


def main():
    img = plt.imread("IMG/1.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    compressed = JPEG_compress(img)
    decompressed = JPEG_decompress(compressed)

    plt.figure()
    plt.imshow(img)
    plt.title("Original")

    plt.figure()
    plt.imshow(decompressed)
    plt.title("Decompressed")

    plt.show()


if __name__ == "__main__":
    main()
