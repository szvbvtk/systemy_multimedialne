import matplotlib.pyplot as plt
import scipy.fftpack
import numpy as np
import cv2


def RLE_encode(img):
    shape = np.array([len(img.shape)])
    shape = np.concatenate([shape, img.shape])

    img = img.flatten()

    output = np.empty(np.prod(img.shape) * 2, dtype=int)
    j = 0
    count = 1

    for i in range(1, len(img)):
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
    elif data[0] == 1:
        shape = data[1:2]
        data = data[2:]
    else:
        raise ValueError("Invalid data")

    output = np.empty(np.prod(shape), dtype=int)
    j = 0

    for i in range(0, len(data), 2):
        output[j : j + data[i + 1]] = data[i]
        j += data[i + 1]

    output = np.reshape(output, shape)

    return output


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def clip_image(img, row_start, row_end, col_start, col_end):
    return img[row_start:row_end, col_start:col_end, :]


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
        scipy.fftpack.dct(block.astype(float) - 128, axis=0, norm="ortho"),
        axis=1,
        norm="ortho",
    )


def idct2(block):
    return (
        scipy.fftpack.idct(
            scipy.fftpack.idct(block.astype(float), axis=0, norm="ortho"),
            axis=1,
            norm="ortho",
        )
        + 128
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


def chromaResample(channel, Ratio="4:4:4"):
    if Ratio == "4:4:4":
        return channel
    elif Ratio == "4:2:2":
        return np.repeat(channel, 2, axis=1)


def CompressBlock(block, Q):
    block_dct = dct2(block)
    block_quantized = quantize(block_dct, Q)
    block_zigzag = zigzag(block_quantized)

    return block_zigzag


def CompressLayer(L, Q):
    S = np.array([])
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w : (w + 8), k : (k + 8)]
            S = np.append(S, CompressBlock(block, Q))

    S_RLE = RLE_encode(S)

    return S_RLE


def DecompressBlock(block, Q):
    block_izigzag = zigzag(block)
    block_dequantized = dequantize(block_izigzag, Q)
    block_idct = idct2(block_dequantized)

    return block_idct


def DecompressLayer(S_RLE, Q, Ratio, shape):
    L = np.zeros((shape[0], shape[1]))

    s_rle_size = S_RLE.size
    S = RLE_decode(S_RLE)
    s_size = S.size

    size_compression = np.round(s_size / s_rle_size, 2)
    print(f"Compression ratio: {size_compression}")

    i = 0
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            vector = S[i : i + 64]
            L[w : (w + 8), k : (k + 8)] = DecompressBlock(vector, Q)
            i += 64

    return L


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

    JPEG.Y = CompressLayer(JPEG.Y, JPEG.QY)
    JPEG.Cr = CompressLayer(JPEG.Cr, JPEG.QC)
    JPEG.Cb = CompressLayer(JPEG.Cb, JPEG.QC)

    return JPEG


def decompress_image(JPEG):
    print("Y")
    Y = DecompressLayer(JPEG.Y, JPEG.QY, JPEG.ChromaRatio, JPEG.shape)

    if JPEG.ChromaRatio == "4:2:2":
        shape = (JPEG.shape[0], JPEG.shape[1] // 2)
    else:
        shape = JPEG.shape

    print("Cr")
    Cr = DecompressLayer(JPEG.Cr, JPEG.QC, JPEG.ChromaRatio, shape)
    print("Cb")
    Cb = DecompressLayer(JPEG.Cb, JPEG.QC, JPEG.ChromaRatio, shape)

    Cr = chromaResample(Cr, JPEG.ChromaRatio)
    Cb = chromaResample(Cb, JPEG.ChromaRatio)

    decompressed_ycrcb = np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)
    decompressed_rgb = cv2.cvtColor(decompressed_ycrcb, cv2.COLOR_YCrCb2RGB)

    return decompressed_rgb


def generate_plot(PRZED_RGB, PO_RGB, suptitle=""):
    fig, axs = plt.subplots(4, 2, sharey=True)
    fig.suptitle(suptitle)
    fig.set_size_inches(6, 7)
    fig.tight_layout()

    for ax in axs.flat:
        ax.axis("off")

    axs[0, 0].set_title("PRZED")
    axs[0, 0].imshow(PRZED_RGB)
    PRZED_YCrCb = cv2.cvtColor(PRZED_RGB, cv2.COLOR_RGB2YCrCb)
    axs[1, 0].imshow(PRZED_YCrCb[:, :, 0], cmap=plt.cm.gray)
    axs[1, 0].set_title("Y")
    axs[2, 0].imshow(PRZED_YCrCb[:, :, 1], cmap=plt.cm.gray)
    axs[2, 0].set_title("Cr")
    axs[3, 0].imshow(PRZED_YCrCb[:, :, 2], cmap=plt.cm.gray)
    axs[3, 0].set_title("Cb")

    axs[0, 1].set_title("PO")
    axs[0, 1].imshow(PO_RGB)
    PO_YCrCb = cv2.cvtColor(PO_RGB, cv2.COLOR_RGB2YCrCb)
    axs[1, 1].imshow(PO_YCrCb[:, :, 0], cmap=plt.cm.gray)
    axs[1, 1].set_title("Y")
    axs[2, 1].imshow(PO_YCrCb[:, :, 1], cmap=plt.cm.gray)
    axs[2, 1].set_title("Cr")
    axs[3, 1].imshow(PO_YCrCb[:, :, 2], cmap=plt.cm.gray)
    axs[3, 1].set_title("Cb")

    return fig


def test():
    img = read_image("IMG/4.jpg")
    img = clip_image(img, 500, 628, 400, 528)

    plt.imshow(img)
    plt.axis("off")
    plt.show()


def main():
    img_name = "2"
    img = read_image(f"IMG/{img_name}.jpg")

    TITLES = ["4:4:4 QY, QC", "4:4:4 QN", "4:2:2 QY, QC", "4:2:2 QN"]

    fragments = [
        [0, 128, 0, 128],
        [120, 248, 280, 408],
        [150, 278, 330, 458],
    ]

    for i, fragment in enumerate(fragments):
        print(f"Fragment {i+1}")
        img_c = clip_image(img, *fragment)

        compressed1 = compress_image(img_c, QY, QC, "4:4:4")
        decompressed1 = decompress_image(compressed1)

        compressed2 = compress_image(img_c, QN, QN, "4:4:4")
        decompressed2 = decompress_image(compressed2)

        compressed3 = compress_image(img_c, QY, QC, "4:2:2")
        decompressed3 = decompress_image(compressed3)

        compressed4 = compress_image(img_c, QN, QN, "4:2:2")
        decompressed4 = decompress_image(compressed4)

        decompressed = [decompressed1, decompressed2, decompressed3, decompressed4]

        for j, d in enumerate(decompressed):
            title = TITLES[j]
            fig = generate_plot(img_c, d, title)
            fig.savefig(f"OUTPUT/{img_name}_{j+1}_fragment_{i+1}.png")


if __name__ == "__main__":
    main()
    # test()
