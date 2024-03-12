import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import cv2
from enum import Enum
import matplotlib

matplotlib.use("TkAgg")


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
    rows = np.ceil(np.linspace(0, height - 1, new_height)).astype(np.int32)
    cols = np.ceil(np.linspace(0, width - 1, new_width)).astype(np.int32)

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


# class Method(Enum):
#     MEAN = 1
#     MEDIAN = 2
#     WEIGHTED_MEAN = 3
#     NEAREST_NEIGHBOR = 4
#     BILINEAR_INTERPOLATION = 5


class ScaleUpMethod(Enum):
    NEAREST_NEIGHBOR = 1
    BILINEAR_INTERPOLATION = 2


class ScaleDownMethod(Enum):
    MEAN = 1
    MEDIAN = 2
    WEIGHTED_MEAN = 3


def scale_up(image, scale, method=ScaleUpMethod.NEAREST_NEIGHBOR):
    if not isinstance(method, ScaleUpMethod):
        raise ValueError("Method must be of type Method")

    if method not in (
        ScaleUpMethod.NEAREST_NEIGHBOR,
        ScaleUpMethod.BILINEAR_INTERPOLATION,
    ):
        print(method)
        raise ValueError("Method must be NEAREST_NEIGHBOR or BILINEAR_INTERPOLATION")

    if method == ScaleUpMethod.NEAREST_NEIGHBOR:
        new_image = nearest_neighbor(image, scale)
    elif method == ScaleUpMethod.BILINEAR_INTERPOLATION:
        new_image = bilinear_interpolation(image, scale)

    return new_image


def scale_down(image, scale, method=ScaleDownMethod.MEAN):
    """
    Scale down image
    :param image: input image
    :param scale: scale factor
    :param method: method, MEAN, MEDIAN, WEIGHTED_MEAN
    :return: new image
    """
    height, width = image.shape[0], image.shape[1]

    new_height = ceil(height / scale)
    new_width = ceil(width / scale)

    if not isinstance(method, ScaleDownMethod):
        raise ValueError("Method must be of type Method")

    if method not in (
        ScaleDownMethod.MEAN,
        ScaleDownMethod.MEDIAN,
        ScaleDownMethod.WEIGHTED_MEAN,
    ):
        raise ValueError("Method must be MEAN, MEDIAN or WEIGHTED_MEAN")

    # grayscale or color image
    if len(image.shape) == 2:
        new_image = np.empty((new_height, new_width), dtype=np.uint8)
    elif len(image.shape) == 3:
        channels = image.shape[2]
        new_image = np.empty((new_height, new_width, image.shape[2]), dtype=np.uint8)
    else:
        raise ValueError("Image array must be 2D or 3D")

    rows = np.linspace(0, height - 1, new_height).astype(np.int32)
    cols = np.linspace(0, width - 1, new_width).astype(np.int32)

    for i in range(new_height):
        for j in range(new_width):
            ix = np.round(rows[i] + np.arange(-3, 4))
            iy = np.round(cols[j] + np.arange(-3, 4))

            ix = ix.clip(0, height - 1).astype(np.int32)
            iy = iy.clip(0, width - 1).astype(np.int32)

            fragment = image[ix, iy]
            # these two should do the same, must be tested
            # new_image[i, j] = np.mean(fragment, axis=(0, 1))
            # for k in range(channels):
            #     new_image[i, j, k] = np.mean(fragment[:, :, k])

            if method == ScaleDownMethod.MEAN:
                new_image[i, j] = np.mean(fragment, axis=(0, 1))
            elif method == ScaleDownMethod.MEDIAN:
                new_image[i, j] = np.median(fragment, axis=(0, 1))
            elif method == ScaleDownMethod.WEIGHTED_MEAN:
                weights = np.array([5, 10, 15, 20, 15, 10, 5])

                # ix = (np.sum(np.multiply(ix, weights)) / np.sum(weights)).astype(
                #     np.int32
                # )
                # iy = (np.sum(np.multiply(iy, weights)) / np.sum(weights)).astype(
                #     np.int32
                # )

                # check whether this gives the same result as the above
                ix = np.average(ix, weights=weights).astype(np.int32)
                iy = np.average(iy, weights=weights).astype(np.int32)

                new_image[i, j] = image[ix, iy]

    return new_image


def scale_image(image, scale, method=None):
    """
    Scale image
    :param image: input image
    :param scale: scale factor, < 1 for downscaling, > 1 for upscaling
    :param method: method
    :return: new image
    """

    if scale < 1:
        scale = 1 / scale
        new_image = scale_down(image, scale, method=ScaleDownMethod.WEIGHTED_MEAN)
    else:
        new_image = scale_up(image, scale, method, ScaleUpMethod.NEAREST_NEIGHBOR)

    return new_image


def plot_images(images, titles, figsize=(15, 10)):
    """
    Plot images
    :param images: list of images
    :param titles: list of titles
    :param figsize: figure size
    :return: None
    """

    if len(images) != len(titles):
        raise ValueError("Number of images is not equal to number of titles")

    fig, axs = plt.subplots(1, len(images), figsize=figsize)

    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].set_title(titles[i])
        # axs[i].axis("off")

    return fig


def show_figure(fig):
    """
    Show figure
    :param fig: figure
    :return: None
    """

    # don not know why it is automatically closed after the first plot is shown (using fig.show, plt.show works perfectly fine)
    fig.show()
    plt.waitforbuttonpress()


def save_figure(fig, path, dpi, format, name):
    # TO DO
    pass


if __name__ == "__main__":
    # image = cv2.imread("IMG_SMALL/SMALL_0003.png")
    image = read_image("IMG_SMALL/SMALL_0002.png")

    # new_image = scale_down(image, 0.3, Method.WEIGHTED_MEAN)
    # new_image = scale_image(image, 2, Method.BILINEAR_INTERPOLATION)
    new_image = scale_image(image, 0.5, ScaleDownMethod.WEIGHTED_MEAN)

    fig = plot_images([image, new_image], ["Original", "Reduced"])
    show_figure(fig)

    # do refaktoryzacji, w szczegolnosci scale up (tak aby przypominało scale down - brak osobnych funkcji tylko jedna)
    # sprawdzenie czy zakomentowany kod daje te same wyniki co odkomentowany
