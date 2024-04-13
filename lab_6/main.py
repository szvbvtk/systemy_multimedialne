import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from tabulate import tabulate


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        size = obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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

    output = np.empty(np.prod(shape), dtype=int)
    j = 0
    print("RLE decoding...")
    for i in tqdm(range(0, len(data), 2)):
        output[j : j + data[i + 1]] = data[i]
        j += data[i + 1]

    output = np.reshape(output, shape)

    return output


def find_repeating_elements_sequence(arr, start):
    counter = 1
    for i in range(start, len(arr) - 1):
        if arr[i] == arr[i + 1]:
            counter += 1
        else:
            break

    return counter


def find_different_elements_sequence(arr, start):
    counter = 1
    for i in range(start, len(arr) - 1):
        if arr[i] != arr[i + 1]:
            counter += 1
        else:
            break

    return counter


def ByteRun_encode(img):
    shape = np.array([len(img.shape)])
    shape = np.concatenate([shape, img.shape])

    img = img.flatten()

    output = np.empty(np.prod(img.shape) * 2, dtype=int)

    i = 0
    j = 0
    print("ByteRun encoding...")
    with tqdm(total=len(img)) as pbar:
        while i < len(img) - 1:
            repeating = True
            if img[i] == img[i + 1]:
                count = find_repeating_elements_sequence(img, i)
            else:
                count = find_different_elements_sequence(img, i)
                repeating = False

            if repeating:
                output[j] = count
                output[j + 1] = img[i]
                j += 2
            else:
                output[j] = -count
                output[j + 1 : j + 1 + count] = img[i : i + count]

                j += count + 1

            i += count
            pbar.update(count)

    output = output[:j]
    output = np.concatenate([shape, output])

    return output


def ByteRun_decode(data):
    if data[0] == 2:
        shape = data[1:3]
        data = data[3:]
    elif data[0] == 3:
        shape = data[1:4]
        data = data[4:]
    else:
        raise ValueError("Invalid data")

    output = np.empty(np.prod(shape), dtype=int)

    i = 0
    j = 0
    print("ByteRun decoding...")
    with tqdm(total=len(data)) as pbar:
        while i < len(data):
            i_prev = i
            count = data[i]
            repeating = count > 0
            count = abs(count)

            i += 1
            if repeating:
                output[j : j + count] = data[i]
                j += count
                i += 1
            else:
                output[j : j + count] = data[i : i + count]
                j += count
                i += count

            pbar.update(i - i_prev)

        output = np.reshape(output, shape)

    return output


def imgToUInt8(image):

    if image.dtype == np.uint8:
        return image
    elif np.issubdtype(image.dtype, np.floating):
        return (image * 255).astype(np.uint8)

    raise ValueError("Unsupported image type")


def compare_images(img1, img2):
    if img1.shape != img2.shape:
        return False

    return np.all(img1 == img2)


def read_image(path):
    img = plt.imread(path)
    img = imgToUInt8(img)
    img = img.astype(int)
    return img


def comparison_plot(images, titles, suptitle):
    number_of_images = len(images)
    fig, axs = plt.subplots(1, number_of_images)
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.suptitle(suptitle)

    for i in range(number_of_images):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
        # axs[i].axis("off")

    return fig


def CR_PR(org_size, new_size):
    return round(org_size / new_size, 4), f"{round(new_size / org_size * 100, 2)}%"


def main():
    dir = "img/"
    filenames = ("aslan.png", "document.png", "technical_drawing.png")

    # CR = compression ratio, PR = compression percentage
    lst = [
        (
            "filename",
            "img size",
            "RLE size",
            "ByteRun size",
            "RLE CR",
            "ByteRun CR",
            "RLE PR",
            "ByteRun PR",
            "RLE==org",
            "ByteRun==org",
        )
    ]

    for i, filename in enumerate(filenames):
        img = read_image(f"{dir}{filename}")
        filename = filename.rsplit(".", 1)[0]
        print(f"Image {i + 1}: {filename}")

        data_RLE = RLE_encode(img)
        new_img_RLE = RLE_decode(data_RLE)

        data_BR = ByteRun_encode(img)
        new_img_BR = ByteRun_decode(data_BR)

        img_size = get_size(img)
        img_size_RLE = get_size(data_RLE)
        img_size_BR = get_size(data_BR)

        RLE_cmp = compare_images(img, new_img_RLE)
        BR_cmp = compare_images(img, new_img_BR)

        CR_RLE, PR_RLE = CR_PR(img_size, img_size_RLE)
        CR_BR, PR_BR = CR_PR(img_size, img_size_BR)

        fig = comparison_plot(
            [img, new_img_RLE, new_img_BR],
            ["Original", "RLE", "ByteRun"],
            suptitle=filename,
        )
        fig.savefig(f"results/{filename}.png", dpi=600)

        lst.append(
            (
                filename,
                img_size,
                img_size_RLE,
                img_size_BR,
                CR_RLE,
                CR_BR,
                PR_RLE,
                PR_BR,
                RLE_cmp,
                BR_cmp,
            )
        )

    table = tabulate(lst, headers="firstrow", tablefmt="pretty", showindex=False)
    with open("results/results.txt", "w") as file:
        file.write(table)


if __name__ == "__main__":
    main()
