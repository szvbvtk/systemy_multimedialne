import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import cv2
from enum import Enum


def imgToUInt8(image):
    """
    Convert image to uint8
    :param img: input image
    :return: new image
    """

    if image.dtype == np.uint8:
        return image
    elif np.issubdtype(image.dtype, np.floating):
        return (image * 255).astype(np.uint8)

    raise ValueError("Unsupported image type")


def read_image(image_path, convertToUint8=True, color_space=cv2.COLOR_BGR2RGB):
    """
    Read image from file
    :param image_path: path to the image
    :param convertToUint8: convert image to uint8
    :param color_space: color space
    :return: image as numpy array
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, color_space)

    if convertToUint8:
        image = imgToUInt8(image)

    return image


def nearest_neighbor(image, scale):
    """
    Nearest neighbor interpolation
    :param image: input image
    :param scale: scale factor
    :return: new image
    """

    height, width = image.shape[0], image.shape[1]
    new_height = ceil(height * scale)
    new_width = ceil(width * scale)

    # grayscale or color image
    if len(image.shape) == 2:
        new_image = np.empty((new_height, new_width), dtype=np.uint8)
    elif len(image.shape) == 3:
        channels = image.shape[2]
        new_image = np.empty((new_height, new_width, channels), dtype=np.uint8)
    else:
        raise ValueError("Image array must be 2D or 3D")

    # -1 to avoid out of bounds, becouse we start from 0
    rows = np.ceil(np.linspace(0, height - 1, new_height)).astype(np.int64)
    cols = np.ceil(np.linspace(0, width - 1, new_width)).astype(np.int64)


    for i in range(new_height):
        for j in range(new_width):
            # replacing the pixels with the nearest one
            new_image[i, j] = image[rows[i], cols[j]]

    return new_image


def bilinear_interpolation(image, scale):
    """
    Bilinear interpolation
    :param image: input image
    :param scale: scale factor
    :return: new image
    """

    height, width = image.shape[0], image.shape[1]
    new_height = ceil(height * scale)
    new_width = ceil(width * scale)

    # grayscale or color image
    if len(image.shape) == 2:
        new_image = np.empty((new_height, new_width), dtype=np.uint8)
    elif len(image.shape) == 3:
        channels = image.shape[2]
        new_image = np.empty((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        raise ValueError("Image array must be 2D or 3D")

    new_image = np.empty((new_height, new_width, channels), dtype=np.uint8)

    rows = np.linspace(0, height - 1, new_height)
    cols = np.linspace(0, width - 1, new_width)

    for i in range(new_height):
        for j in range(new_width):
            xx, yy = rows[i], cols[j]

            x1, y1 = floor(xx), floor(yy)
            x2, y2 = ceil(xx), ceil(yy)

            x, y = xx - floor(xx), yy - floor(yy)

            # 4 nearest pixels
            q11 = image[x1, y1]
            q12 = image[x1, y2]
            q21 = image[x2, y1]
            q22 = image[x2, y2]

            new_image[i, j] = (
                q11 * (1 - x) * (1 - y)
                + q21 * x * (1 - y)
                + q12 * (1 - x) * y
                + q22 * x * y
            )

    return new_image


class ReduceMethod(Enum):
    MEAN = 1
    MEDIAN = 2
    WEIGHTED_MEAN = 3


def reduce_image(image, scale, method=ReduceMethod.MEAN):
    height, width = image.shape[0], image.shape[1]

    new_height = ceil(height / scale)
    new_width = ceil(width / scale)

    if not isinstance(method, ReduceMethod):
        raise ValueError("Method must be of type Method")

    # grayscale or color image
    if len(image.shape) == 2:
        new_image = np.empty((new_height, new_width), dtype=np.uint8)
    elif len(image.shape) == 3:
        channels = image.shape[2]
        new_image = np.empty((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        raise ValueError("Image array must be 2D or 3D")

    rows = np.linspace(0, height - 1, new_height).astype(np.uint8)
    cols = np.linspace(0, width - 1, new_width).astype(np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            ix = np.round(rows[i] + np.arange(-3, 4))
            iy = np.round(cols[j] + np.arange(-3, 4))

            ix = ix.clip(0, height - 1).astype(np.uint8)
            iy = iy.clip(0, width - 1).astype(np.uint8)

            fragment = image[ix, iy]
            # these two should do the same, must be tested
            # new_image[i, j] = np.mean(fragment, axis=(0, 1))
            # for k in range(channels):
            #     new_image[i, j, k] = np.mean(fragment[:, :, k])

            if method == ReduceMethod.MEAN:
                new_image[i, j] = np.mean(fragment, axis=(0, 1))
            elif method == ReduceMethod.MEDIAN:
                new_image[i, j] = np.median(fragment, axis=(0, 1))
            elif method == ReduceMethod.WEIGHTED_MEAN:
                # weights = np.array([5, 10, 15, 20, 15, 10, 5])
                # ix_weighted = np.dot(fragment, weights) / np.sum(weights)
                pass


if __name__ == "__main__":
    # image = cv2.imread("IMG_SMALL/SMALL_0003.png")
    image = read_image("IMG_SMALL/SMALL_0003.png")
    print(image.dtype)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image= np.zeros((3,3,3),dtype=np.uint8)
    # image[1,1,:]=255

    new_image = nearest_neighbor(image, 4)
    plt.imshow(new_image)
    plt.show()
    # plt.imsave("bilinear.png", new_image)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(image)
    # axs[0].set_title("Original Image")
    # axs[1].imshow(new_image)
    # axs[1].set_title("Scaled Image")
    # plt.show()

    # image = read_image("IMG_BIG/BIG_0003.jpg")
    # image = read_image("IMG_SMALL/SMALL_0003.png")
    # new_image = reduce_image(image, 2, ReduceMethod.MEAN)
    # new_image = nearest_neighbor(image, 2)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(image)
    # axs[0].set_title("Original Image")
    # axs[1].imshow(new_image)
    # axs[1].set_title("Scaled Image")
    # plt.show()
