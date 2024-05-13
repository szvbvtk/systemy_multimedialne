# https://pixabay.com/pl/photos/building-kopu%C5%82a-dzwonnica-6351976/

import cv2
import numpy as np
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


np.random.seed(42)


# Zniekształcenia obrazu
def read_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def jpeg_compress(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimage = cv2.imencode(".jpg", image, encode_param)
    decimage = cv2.imdecode(encimage, 1)

    return decimage


def blur_image(image, kernel_size):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image


def noise_image(image, alpha):
    noise_range = (-25, 25)
    rand = (noise_range[1] - noise_range[0]) * np.random.random(
        (image.shape)
    ) + noise_range[0]

    noisy_image = (image + alpha * rand).clip(0, 255).astype(np.uint8)

    return noisy_image


def noise_SnP(image, S=255, P=0, rnd=(333, 9999)):
    r, c = image.shape
    number_of_pixels = np.random.randint(rnd[0], rnd[1])

    for i in range(number_of_pixels):
        y = np.random.randint(0, r - 1)
        x = np.random.randint(0, c - 1)
        image[y][x] = S

    number_of_pixels = np.random.randint(rnd[0], rnd[1])

    for i in range(number_of_pixels):
        y = np.random.randint(0, r - 1)
        x = np.random.randint(0, c - 1)
        image[y][x] = P

    return image


# Obiektywne miary jakości
def MSE(source_image, target_image):
    return np.mean((source_image - target_image) ** 2)


def NMSE(source_image, target_image):
    return np.mean((source_image - target_image) ** 2) / np.mean(target_image**2)


def PSNR(source_image, target_image):
    # sprawdzic czy sygnal jest w uint8 - 255
    return 10 * np.log10(255**2 / MSE(source_image, target_image))


def IF(source_image, target_image):
    # ou can use the numpy. multiply() function to perform element-wise multiplication with two-dimensional arrays. For instance, numpy. multiply() performs element-wise multiplication on the two 2D arrays arr and arr1 , resulting in the output array [[ 4 12 30 32], [ 8 15 15 14]] .
    #    sprawdzic czy to dziala jak powinno
    return 1 - (np.sum((source_image - target_image) ** 2)) / np.sum(
        np.multiply(source_image, target_image)
    )


def SSIM(source_image, target_image):
    return ssim(source_image, target_image, channel_axis=2)


def plot_images(image_1, image_2, title1, title2):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    axs[0].imshow(image_1)
    axs[0].set_title(title1)
    axs[1].imshow(image_2)
    axs[1].set_title(title2)

    plt.show()


if __name__ == "__main__":
    image_path = Path("./img/1.jpg")

    image = read_image(image_path)
    # new_image = jpeg_compress(image, 5)
    # new_image = blur_image(image, 3)
    new_image = noise_image(image, 0.001)

    ssim = SSIM(image, new_image)
    print(ssim)

    plot_images(image, new_image, "1", "2")
