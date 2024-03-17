import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import cv2
from enum import Enum
import matplotlib
import os

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
        new_image = np.empty((new_height, new_width, channels), dtype=np.uint8)
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

            fragment = image[ix, iy, :]

            if method != ScaleDownMethod.WEIGHTED_MEAN:
                # new_image[i, j] = np.mean(fragment, axis=(0, 1))
                if len(image.shape) < 3:
                    new_image[i, j] = np.mean(fragment)
                elif len(image.shape) == 3:
                    for channel in range(channels):
                        if method == ScaleDownMethod.MEAN:
                            new_image[i, j, channel] = np.mean(image[ix, iy, channel])
                        elif method == ScaleDownMethod.MEDIAN:
                            new_image[i, j, channel] = np.median(image[ix, iy, channel])

            elif method == ScaleDownMethod.WEIGHTED_MEAN:
                weights = np.array([7, 9, 12, 15, 11, 4, 18])

                # ix1 = (np.sum(np.multiply(ix, weights)) / np.sum(weights)).astype(
                #     np.int32
                # )
                # iy = (np.sum(np.multiply(iy, weights)) / np.sum(weights)).astype(
                #     np.int32
                # )

                ix = np.average(ix, weights=weights).astype(np.int32)
                iy = np.average(iy, weights=weights).astype(np.int32)

                # print(np.equal(ix, ix1))

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

    image_copy = image.copy()

    if scale < 1:
        scale = 1 / scale

        if method is None:
            method = ScaleDownMethod.MEAN

        new_image = scale_down(image_copy, scale, method=method)
    else:
        if method is None:
            method = ScaleUpMethod.NEAREST_NEIGHBOR

        new_image = scale_up(image_copy, scale, method=method)

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
        axs[i].axis("on")

    plt.tight_layout()

    return fig


def show_figure(fig):
    """
    Show figure
    :param fig: figure
    :return: None
    """

    # do not know why it is automatically closed after the first plot is shown (using fig.show, plt.show works perfectly fine)
    fig.show()
    plt.waitforbuttonpress()


def save_figure(fig, path, dpi, format, name):
    """
    Save figure
    :param fig: figure
    :param path: path
    :param dpi: dpi
    :param format: format
    :param name: name
    :return: None
    """

    i = 1
    while os.path.exists(f"{path}/{name}_{i}.{format}"):
        i += 1

    fig.savefig(f"{path}/{name}_{i}.{format}", dpi=dpi, format=format)


def slice_image(image, x, y, height, width):
    """
    Slice image
    :param image: input image
    :param x: x position
    :param y: y position
    :param height: height
    :param width: width
    """

    y_stop = min(y + height, image.shape[0])
    x_stop = min(x + width, image.shape[1])

    if y_stop - y != height or x_stop - x != width:
        print(
            f"Couldn't slice the image, because the fragment is out of bounds. New fragment size: {y_stop - y}x{x_stop - x}"
        )

    return image[y:y_stop, x:x_stop]


def main_scale_up():
    fragment = None

    dir = "IMG_SMALL"
    filename = "SMALL_0003.png"

    format = filename.split(".")[-1]

    image = read_image(
        f"{dir}/{filename}", convertToUint8=True, color_space=cv2.COLOR_BGR2RGB
    )

    fragment = slice_image(image, 45, 235, 40, 40)

    if fragment is None:
        fragment = image

    scale = 5

    fragment_nn = scale_image(fragment, scale, ScaleUpMethod.NEAREST_NEIGHBOR)
    fragment_bi = scale_image(fragment, scale, ScaleUpMethod.BILINEAR_INTERPOLATION)

    fig = plot_images(
        [fragment, fragment_nn, fragment_bi],
        ["Oryginalne", "Nearest neighbor", "Bilinear interpolation"],
    )
    show_figure(fig)
    save_figure(fig, "./OUTPUT", 600, format, f"{filename}_upscaling_{scale}")


def main_scale_down():
    fragment = None

    dir = "IMG_BIG"
    filename = "BIG_0003.jpg"

    format = filename.split(".")[-1]

    image = read_image(
        f"{dir}/{filename}", convertToUint8=True, color_space=cv2.COLOR_BGR2RGB
    )

    fragment = slice_image(image, 300, 4200, 300, 300)

    if fragment is None:
        fragment = image

    scale = 0.8

    fragment_mean = scale_image(fragment, scale, ScaleDownMethod.MEAN)
    fragment_median = scale_image(fragment, scale, ScaleDownMethod.MEDIAN)
    fragment_weighted_mean = scale_image(fragment, scale, ScaleDownMethod.WEIGHTED_MEAN)

    fig = plot_images(
        [fragment, fragment_mean, fragment_median, fragment_weighted_mean],
        ["Oryginalne", "Mean", "Median", "Weighted mean"],
    )

    # print(fragment.shape[0], fragment_mean.shape[0], fragment_median.shape[0], fragment_weighted_mean.shape[0])

    show_figure(fig)
    save_figure(fig, "./OUTPUT", 600, format, f"{filename}_downscaling_{scale}")


def main_canny_test():
    fragment = None

    dir = "IMG_BIG"
    filename = "BIG_0003.jpg"

    format = filename.split(".")[-1]

    image = read_image(
        f"{dir}/{filename}", convertToUint8=True, color_space=cv2.COLOR_BGR2RGB
    )

    # fragment = slice_image(image, 0, 2300, 1200, 1200)

    if fragment is None:
        fragment = image

    scale = 3

    # fragment_mean = scale_image(fragment, scale, ScaleDownMethod.MEAN)
    # fragment_median = scale_image(fragment, scale, ScaleDownMethod.MEDIAN)
    # fragment_weighted_mean = scale_image(fragment, scale, ScaleDownMethod.WEIGHTED_MEAN)

    # fragment = cv2.Canny(fragment, 50, 150)
    # fragment_mean = cv2.Canny(fragment_mean, 50, 150)
    # fragment_median = cv2.Canny(fragment_median, 50, 150)
    # fragment_weighted_mean = cv2.Canny(fragment_weighted_mean, 50, 150)

    fragment_nn = scale_image(fragment, 3, ScaleUpMethod.NEAREST_NEIGHBOR)
    fragment_bi = scale_image(fragment, 3, ScaleUpMethod.BILINEAR_INTERPOLATION)

    fragment = cv2.Canny(fragment, 50, 150)
    fragment_nn = cv2.Canny(fragment_nn, 50, 150)
    fragment_bi = cv2.Canny(fragment_bi, 50, 150)

    fig = plot_images(
        [fragment, fragment_nn, fragment_bi],
        ["Oryginalne", "Nearest neighbor", "Bilinear interpolation"],
    )

    # fig = plot_images(
    #     [fragment, fragment_mean, fragment_median, fragment_weighted_mean],
    #     ["Oryginalne", "Mean", "Median", "Weighted mean"],
    # )

    # print(fragment.shape[0], fragment_mean.shape[0], fragment_median.shape[0], fragment_weighted_mean.shape[0])

    show_figure(fig)
    save_figure(fig, "./OUTPUT", 600, format, f"{filename}_canny_{scale}")


if __name__ == "__main__":
    # main_scale_up()
    # main_scale_down()
    main_canny_test()

    # dir = "IMG_BIG"
    # filename = "BIG_0003.jpg"

    # image = read_image(
    #     f"{dir}/{filename}", convertToUint8=True, color_space=cv2.COLOR_BGR2RGB
    # )

    # image = image[2300:3500, 0:1200]

    # plt.imshow(image)
    # plt.show()

    # print( 1/ 0.8)
    # fragment 300 px -> 240 px fragment_mean
