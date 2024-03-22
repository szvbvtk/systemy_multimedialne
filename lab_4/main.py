import numpy as np
import matplotlib.pyplot as plt
import cv2


def imgToFloat(img):
    img_out = img.copy()
    if np.issubdtype(img_out.dtype, np.floating):
        return img_out
    elif np.issubdtype(img_out.dtype, np.unsignedinteger):
        return img_out.astype(np.float32) / 255

    raise ValueError("Unsupported image type")


def read_image(image_path, color_space=cv2.COLOR_BGR2RGB):
    """
    Read image from file
    :param image_path: path to the image
    :param convertToUint8: convert image to uint8
    :param color_space: color space
    :return: image as numpy array
    """

    image = cv2.imread(image_path)

    if color_space is not None:
        image = cv2.cvtColor(image, color_space)

    return image


def colorFit(pixel, pallet):
    idx = np.argmin(np.linalg.norm(pallet - pixel, axis=1))
    return pallet[idx]


# paleta = np.linspace(0,1,3).reshape(3,1)
# print(colorFit(0.43,paleta)) # 0.5
# print(colorFit(0.66,paleta))
# print(colorFit(0.8,paleta))



pallet1bit = np.linspace(0, 1, 2).reshape(2, 1)
pallet2bit = np.linspace(0, 1, 4).reshape(4, 1)
pallet4bit = np.linspace(0, 1, 16).reshape(16, 1)



pallet8 = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            0.0,
        ],
        [
            1.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            1.0,
            0.0,
        ],
        [
            1.0,
            1.0,
            1.0,
        ],
    ]
)


pallet16 = np.array(
    [
        [
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            1.0,
            1.0,
        ],
        [
            0.0,
            0.0,
            1.0,
        ],
        [
            1.0,
            0.0,
            1.0,
        ],
        [
            0.0,
            0.5,
            0.0,
        ],
        [
            0.5,
            0.5,
            0.5,
        ],
        [
            0.0,
            1.0,
            0.0,
        ],
        [
            0.5,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.5,
        ],
        [
            0.5,
            0.5,
            0.0,
        ],
        [
            0.5,
            0.0,
            0.5,
        ],
        [
            1.0,
            0.0,
            0.0,
        ],
        [
            0.75,
            0.75,
            0.75,
        ],
        [
            0.0,
            0.5,
            0.5,
        ],
        [
            1.0,
            1.0,
            1.0,
        ],
        [
            1.0,
            1.0,
            0.0,
        ],
    ]
)

# print(colorFit(np.array([0.25, 0.25, 0.5]), pallet8))
# print(colorFit(np.array([0.25, 0.25, 0.5]), pallet16))


def kwant_colorFit(img, pallet):
    img_out = imgToFloat(img)
    height, width = img_out.shape[:2]
    for i in range(height):
        for j in range(width):
            img_out[i, j] = colorFit(img_out[i, j], pallet)

    return img_out


rng = np.random.default_rng(seed=4287873748)


def random_dithering(img, rnd_generator=rng):
    img = imgToFloat(img)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    height, width = img.shape[:2]

    random_matrix = rnd_generator.random((height, width))
    img = img >= random_matrix
    img = img * 1

    return img


def ordered_dithering(img, pallet):
    img = imgToFloat(img)

    height, width = img.shape[:2]

    # M2
    M = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
    n = 2
    M_pre = (M + 1) / pow((2 * n), 2) - 0.5

    r = 1
    for i in range(height):
        for j in range(width):
            C = img[i, j]
            img[i, j] = colorFit(C + M_pre[i % (2 * n), j % (2 * n)], pallet)

    return img


def floyd_steinberg_dithering(img, pallet):
    img = imgToFloat(img)

    height, width = img.shape[:2]

    for i in range(height):
        for j in range(width):
            old_pixel = img[i, j].copy()
            new_pixel = colorFit(old_pixel, pallet)
            img[i, j] = new_pixel
            quant_error = old_pixel - new_pixel

            if j < width - 1:
                img[i, j + 1] += quant_error * (7 / 16)
            if i < height - 1 and j > 0:
                img[i + 1, j - 1] += quant_error * (3 / 16)
            if i < height - 1:
                img[i + 1, j] += quant_error * (5 / 16)
            if i < height - 1 and j < width - 1:
                img[i + 1, j + 1] += quant_error * (1 / 16)

    return img


nr = 5

def main3():
    img = read_image(f"IMG_GS/GS_000{nr}.png", color_space=cv2.COLOR_BGR2GRAY)

    pallet = pallet2bit

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(img, cmap="gray")
    axs[1, 0].imshow(kwant_colorFit(img, pallet), cmap="gray")
    axs[0, 1].imshow(ordered_dithering(img, pallet), cmap="gray")
    axs[1, 1].imshow(floyd_steinberg_dithering(img, pallet), cmap="gray")
    # axs[1, 1].imshow(random_dithering(img), cmap="gray")

    axs[0, 0].set_title("Oryginał")
    axs[1, 0].set_title("Kwantyzacja")
    axs[0, 1].set_title("Dithering zorganizowany")
    # axs[1, 1].set_title("Dithering losowy")
    axs[1, 1].set_title("Dithering Floyda-Steinberga")

    for ax in axs.flat:
        ax.axis("off")

    fig.suptitle("4 bity")

    plt.tight_layout()

    plt.savefig(f"OUTPUT/4bits_GS_000{nr}.jpg")

    plt.show()

def main1():
    # img = read_image("IMG_SMALL/SMALL_0002.png", color_space=cv2.COLOR_BGR2GRAY)
    img = read_image(f"IMG_SMALL/SMALL_000{nr}.jpg")

    pallet = pallet8
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 1].imshow(kwant_colorFit(img, pallet), cmap="gray")
    axs[0, 2].imshow(ordered_dithering(img, pallet), cmap="gray")
    # axs[1, 0].imshow(random_dithering(img), cmap="gray")
    axs[1, 1].imshow(floyd_steinberg_dithering(img, pallet), cmap="gray")

    axs[0, 0].set_title("Original")
    axs[0, 1].set_title("Kwantyzacja")
    axs[0, 2].set_title("Dithering zorganizowany")
    # axs[1, 0].set_title("Dithering losowy")
    axs[1, 1].set_title("Dithering Floyda-Steinberga")

    for ax in axs.flat:
        ax.axis("off")

    fig.suptitle("Paleta 8 kolorów")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"OUTPUT/8colors_SMALL_000{nr}.jpg")


def main2():
    # img = read_image("IMG_SMALL/SMALL_0002.png", color_space=cv2.COLOR_BGR2GRAY)
    img = read_image(f"IMG_SMALL/SMALL_000{nr}.jpg")

    pallet = pallet16
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 1].imshow(kwant_colorFit(img, pallet), cmap="gray")
    axs[0, 2].imshow(ordered_dithering(img, pallet), cmap="gray")
    # axs[1, 0].imshow(random_dithering(img), cmap="gray")
    axs[1, 1].imshow(floyd_steinberg_dithering(img, pallet), cmap="gray")

    axs[0, 0].set_title("Original")
    axs[0, 1].set_title("Kwantyzacja")
    axs[0, 2].set_title("Dithering zorganizowany")
    # axs[1, 0].set_title("Dithering losowy")
    axs[1, 1].set_title("Dithering Floyda-Steinberga")

    for ax in axs.flat:
        ax.axis("off")

    fig.suptitle("Paleta 16 kolorów")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"OUTPUT/16colors_SMALL_000{nr}.jpg")


if __name__ == "__main__":
    # print(np.unique(floyd_steinberg_dithering(img.copy(),np.linspace(0,1,2).reshape(2,1))).size)
    main1()
    main2()
    # main3()
