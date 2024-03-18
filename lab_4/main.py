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

pallet1 = np.linspace(0, 1, 2).reshape(2, 1)
pallet2 = np.linspace(0, 1, 4).reshape(4, 1)
pallet4 = np.linspace(0, 1, 8).reshape(8, 1)

pallet_4bit = np.linspace(0, 1, 16).reshape(16, 1)

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


rng = np.random.default_rng(seed=423748)


def random_dithering(img, rnd_generator=rng):
    img = imgToFloat(img)

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
    n = M.shape[0]
    # n = 3
    M_pre = (M + 1) / pow((2 * n), 2) - 0.5

    r = 1
    for i in range(height):
        for j in range(width):
            C = img[i, j]
            img[i, j] = colorFit(C + M_pre[i % (2 * n), j % (2 * n)], pallet)

    return img


# pallet8 i 16 do kolorowych, reszta grayscale
def main():
    # img = read_image("IMG_GS/GS_0002.png", color_space=Gra)
    img = read_image("IMG_GS/GS_0002.png", color_space=cv2.COLOR_BGR2GRAY)
    # img = kwant_colorFit(img, pallet8)
    # plt.imshow(img, cmap="gray")
    # plt.imshow(img, cmap=plt.cm.gray)
    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    print(img.shape)

    # img = random_dithering(img)
    img = ordered_dithering(img, pallet8)
    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
