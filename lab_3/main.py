import matplotlib.pyplot as plt
import numpy as np
import cv2


def scale_nearest_neighbor(image, scale):
    """
    Nearest neighbor interpolation
    :param image: input image
    :param scale: scale factor
    :return: new image
    """

    height, width, channels = image.shape
    new_height = int(height * scale)
    new_width = int(width * scale)

    # -1 to avoid out of bounds, becouse we start from 0
    rows = np.ceil(np.linspace(0, height - 1, new_height)).astype(np.uint8)
    cols = np.ceil(np.linspace(0, width - 1, new_width)).astype(np.uint8)

    new_image = np.empty((new_height, new_width, channels), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # replacing the pixels with the nearest one
            new_image[i, j] = image[rows[i], cols[j]]

    return new_image


# działa tylko jak się dopisze do obu pętli -1, trzeba sprawdzić czy tak powinno być (wiem czemu nie działa, nie wiem czy jest inn metoda)
def scale_bilinear(image, scale):
    """
    Bilinear interpolation
    :param image: input image
    :param scale: scale factor
    :return: new image
    """

    height, width, channels = image.shape
    new_height = int(height * scale)
    new_width = int(width * scale)

    new_image = np.empty((new_height, new_width, channels), dtype=np.uint8)

    rows = np.linspace(0, height - 1, new_height)
    cols = np.linspace(0, width - 1, new_width)

    for i in range(new_height - 1):
        for j in range(new_width - 1):
            x = rows[i] - int(rows[i])
            y = cols[j] - int(cols[j])

            # 4 nearest pixels
            pixel1 = image[int(rows[i]), int(cols[j])]
            pixel2 = image[int(rows[i]), int(cols[j] + 1)]
            pixel3 = image[int(rows[i] + 1), int(cols[j])]
            pixel4 = image[int(rows[i] + 1), int(cols[j] + 1)]

            # bilinear interpolation
            new_image[i, j] = (
                (1 - x) * (1 - y) * pixel1
                + x * (1 - y) * pixel3
                + (1 - x) * y * pixel2
                + x * y * pixel4
            )

    return new_image


if __name__ == "__main__":
    # image = cv2.imread("IMG_SMALL/SMALL_0002.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image= np.zeros((3,3,3),dtype=np.uint8)
    image[1,1,:]=255

    scale = 2
    new_image = scale_nearest_neighbor(image, scale)
    new_image = image
    # new_image = scale_bilinear(image, scale)

    print(new_image.shape)
    # print(new_image2.shape)

    plt.imshow(new_image)
    plt.show()

    # print(image.shape, new_image.shape)
