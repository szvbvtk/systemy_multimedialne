# https://pixabay.com/pl/photos/building-kopu%C5%82a-dzwonnica-6351976/

import cv2
import numpy as np
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


np.random.seed(42)


# Zniekształcenia obrazu
def read_image(image_path):
    image = cv2.imread(str(image_path))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def jpeg_compress(image, alpha):
    encode_param = [int(cv2.IMWRITE_JPEG_alpha), alpha]
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


def plot_images(image_1, image_2, title1, title2):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    axs[0].imshow(image_1)
    axs[0].set_title(title1)
    axs[1].imshow(image_2)
    axs[1].set_title(title2)

    plt.show()


image_path = Path("./img/1.jpg")
output_dir = Path("./output")


def main_test():
    image_path = Path("./img/1.jpg")

    image = read_image(image_path)
    # new_image = jpeg_compress(image, 5)
    # new_image = blur_image(image, 5)
    new_image = noise_image(image, 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    plot_images(image, new_image, "1", "2")


def main_generate():

    SAVE = True

    image = read_image(image_path)
    method = "noise"

    # qualities = np.linspace(8, 71, 15).astype(np.uint8)
    # qualities = np.array([5, 10, 15, 20, 25, 30, 35, 40 , 45, 50, 55, 60, 65, 75, 80])
    # modified_images = [jpeg_compress(image, alpha) for alpha in qualities]
    A = np.linspace(0.1, 5.5, 15)
    modified_images = [noise_image(image, alpha) for alpha in A]

    if SAVE:
        [p.unlink() for p in output_dir.glob("*.jpg") if p.is_file()]
        [
            cv2.imwrite(
                str(
                    output_dir / f"{i+1}_{method}_{round(alpha, 2)}.jpg",
                ),
                modified_image,
            )
            for i, (alpha, modified_image) in enumerate(
                zip(A, modified_images)
            )
        ]
    else:
        [
            cv2.imshow(f"alpha: {alpha}%", modified_image)
            for alpha, modified_image in zip(A, modified_images)
        ]
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # main_test()
    main_generate()
