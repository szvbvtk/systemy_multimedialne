import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def IF(source_image, target_image):
    return 1 - (np.sum((source_image - target_image) ** 2)) / np.sum(
        np.multiply(source_image, target_image)
    )


def MSE(source_image, target_image):
    return np.mean((source_image - target_image) ** 2)


def watermark(img, mask, alpha=0.25):
    assert (img.shape[0] == mask.shape[0]) and (
        img.shape[1] == mask.shape[1]
    ), "Wrong size"
    if len(img.shape) < 3:
        flag = True
        t_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    else:
        flag = False
        t_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    if mask.dtype == bool:
        t_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
    elif mask.dtype == np.uint8:
        if len(mask.shape) < 3:
            t_mask = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        else:
            t_mask = cv2.cvtColor((mask).astype(np.uint8), cv2.COLOR_RGB2RGBA)
    else:
        if len(mask.shape) < 3:
            t_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
        else:
            t_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_RGB2RGBA)
    t_out = cv2.addWeighted(t_img, 1, t_mask, alpha, 0)
    if flag:
        out = cv2.cvtColor(t_out, cv2.COLOR_RGBA2GRAY)
    else:
        out = cv2.cvtColor(t_out, cv2.COLOR_RGBA2RGB)
    return out


def put_data(img, data, binary_mask=np.uint8(1)):
    assert img.dtype == np.uint8, "img wrong data type"
    assert binary_mask.dtype == np.uint8, "binary_mask wrong data type"
    un_binary_mask = np.unpackbits(binary_mask)
    if data.dtype != bool:
        unpacked_data = np.unpackbits(data)
    else:
        unpacked_data = data
    dataspace = img.shape[0] * img.shape[1] * np.sum(un_binary_mask)
    assert dataspace >= unpacked_data.size, "too much data"
    if dataspace == unpacked_data.size:
        prepered_data = unpacked_data.reshape(
            img.shape[0], img.shape[1], np.sum(un_binary_mask)
        ).astype(np.uint8)
    else:
        prepered_data = np.resize(
            unpacked_data, (img.shape[0], img.shape[1], np.sum(un_binary_mask))
        ).astype(np.uint8)
    mask = np.full((img.shape[0], img.shape[1]), binary_mask)
    img = np.bitwise_and(img, np.invert(mask))
    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):
        if b:
            temp = prepered_data[:, :, bv]
            temp = np.left_shift(temp, i)
            img = np.bitwise_or(img, temp)
            bv += 1
    return img


def pop_data(img, binary_mask=np.uint8(1), out_shape=None):
    un_binary_mask = np.unpackbits(binary_mask)
    data = np.zeros((img.shape[0], img.shape[1], np.sum(un_binary_mask))).astype(
        np.uint8
    )
    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):
        if b:
            mask = np.full((img.shape[0], img.shape[1]), 2**i)
            temp = np.bitwise_and(img, mask)
            data[:, :, bv] = temp[:, :].astype(np.uint8)
            bv += 1
    if out_shape != None:
        tmp = np.packbits(data.flatten())
        tmp = tmp[: np.prod(out_shape)]
        data = tmp.reshape(out_shape)
    return data


def read_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def main_watermark():
    img_dir = Path("./img")
    img1_path = img_dir / "5.jpg"
    mask_path = img_dir / "watermark2.jpg"

    img1 = read_image(img1_path)
    mask = read_image(mask_path)

    img1 = cv2.resize(img1, (mask.shape[1], mask.shape[0]))

    # watermarked = watermark(img1, mask, alpha=0.5)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(img1)
    # axs[0].set_title("Original image")
    # axs[0].axis("off")
    # axs[1].imshow(mask)
    # axs[1].set_title("Mask")
    # axs[1].axis("off")
    # axs[2].imshow(watermarked)
    # axs[2].set_title("Watermarked image")
    # axs[2].axis("off")
    # plt.show()

    # mse = round(MSE(img1, watermarked), 2)
    # if_ = round(IF(img1, watermarked), 2)

    # print(f"MSE: {mse}")
    # print(f"IF: {if_}")

    MSE_ = []
    IF_ = []

    for i in range(5, 95, 10):
        watermarked = watermark(img1, mask, alpha=i / 100)
        mse = round(MSE(img1, watermarked), 2)
        if_ = round(IF(img1, watermarked), 2)
        MSE_.append(mse)
        IF_.append(if_)

        plt.imshow(watermarked)
        plt.title(f"Watermarked image with alpha = {i}%")
        plt.axis("off")
        plt.savefig(f"output_watermark/{i}.png")
        plt.close()

    plt.plot(range(5, 85, 10), MSE_, label="MSE")
    plt.xlabel("Alpha [%]")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output_watermark/metric1.png")
    plt.close()
    plt.plot(range(5, 85, 10), IF_, label="IF")
    plt.xlabel("Alpha [%]")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output_watermark/metric2.png")


def main():
    img_dir = Path("./img")
    img1_path = img_dir / "1.jpg"
    img2_path = img_dir / "watermark2.jpg"

    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    print(min(img1.flatten()), max(img1.flatten()))
    print(min(img2.flatten()), max(img2.flatten()))

    img2_shape = img2.shape
    img2 = img2.flatten()
    l = len(img2) // 3
    img2_1 = img2[:l]
    img2_2 = img2[l : 2 * l]
    img2_3 = img2[2 * l :]

    binary_mask1 = 1
    binary_mask2 = 1
    binary_mask3 = 1

    img_r = put_data(img1[:, :, 0], img2_1, np.uint8(binary_mask1))
    img_g = put_data(img1[:, :, 1], img2_2, np.uint8(binary_mask2))
    img_b = put_data(img1[:, :, 2], img2_3, np.uint8(binary_mask3))

    r = pop_data(img_r, np.uint8(binary_mask1), out_shape=img2_1.shape)
    g = pop_data(img_g, np.uint8(binary_mask2), out_shape=img2_2.shape)
    b = pop_data(img_b, np.uint8(binary_mask3), out_shape=img2_3.shape)

    vector = np.concatenate((r.flatten(), g.flatten(), b.flatten()))

    new_img1 = np.empty_like(img1)
    new_img1[:, :, 0] = img_r
    new_img1[:, :, 1] = img_g
    new_img1[:, :, 2] = img_b

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(img1)
    axs[0].set_title("Original image")
    axs[0].axis("off")
    axs[1].imshow(new_img1)
    axs[1].set_title("with hidden data")
    axs[1].axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"output_steg/1_{binary_mask1}_{binary_mask2}_{binary_mask3}.png")

    img2_reconstructed = vector.reshape(img2_shape)
    plt.imshow(img2_reconstructed)
    plt.show()

    # Miary jakosci
    mse_r = round(MSE(img1[:, :, 0], img_r), 2)
    mse_g = round(MSE(img1[:, :, 1], img_g), 2)
    mse_b = round(MSE(img1[:, :, 2], img_b), 2)

    IF_r = round(IF(img1[:, :, 0], img_r), 2)
    IF_g = round(IF(img1[:, :, 1], img_g), 2)
    IF_b = round(IF(img1[:, :, 2], img_b), 2)

    print(f"MSE R: {mse_r}")
    print(f"MSE G: {mse_g}")
    print(f"MSE B: {mse_b}")
    print(f"IF R: {IF_r}")
    print(f"IF G: {IF_g}")
    print(f"IF B: {IF_b}")


if __name__ == "__main__":
    main()
    # main_watermark()
    # test()

    # text = "Hello, World!"

    # # Konwersja łańcucha tekstowego na tablicę bajtów
    # byte_array = np.frombuffer(text.encode(), dtype=np.uint8)
    # print(byte_array)
    # print([ord(c) for c in text])
