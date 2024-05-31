import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def SSIM(source_image, target_image):
    return ssim(source_image, target_image, channel_axis=2)


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


def main():
    img_dir = Path("./img")
    img_path = img_dir / "1.png"
    img2_path = img_dir / "2.jpg"
    watermark_path = img_dir / "2.jpg"

    # img = read_image(img_path)
    # # mask = read_image(img2_path)
    # # cv2.resize(mask, (img.shape[1], img.shape[0]))

    # img2 = read_image(watermark_path)

    # # print(img.shape, mask.shape)
    # # print(img.dtype, mask.dtype)
    # # watermarked_img = watermark(img, mask, 0.4)

    # # mask = np.random.randint(0, 2, (img.shape[0], img.shape[1]), dtype=bool)
    # # alpha = 0.25
    # # watermarked_img = watermark(img, mask, alpha)

    # # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # # ax[0].imshow(img)
    # # ax[0].set_title("Original image")
    # # ax[0].axis("off")
    # # ax[1].imshow(mask, cmap="gray")
    # # ax[1].set_title("Mask")
    # # ax[1].axis("off")
    # # ax[2].imshow(watermarked_img)
    # # ax[2].set_title("Watermarked image")
    # # ax[2].axis("off")
    # # plt.show()

    # img2 = img2[200 : img.shape[0] // 2, 100 : img.shape[1] // 2]
    # shape = img2.shape
    # img2 = img2.flatten()
    # img2 = img2.reshape(shape)
    # plt.imshow(img2)
    # plt.show()
    # # print(img2.flatten().shape)

    img = read_image(img_path)
    img2 = read_image(img2_path)
    img2 = img2[200 : img.shape[0] // 2, 100 : img.shape[1] // 2]
    shape = img2.shape
    img2 = img2.flatten()
    l = len(img2) // 3
    img2_1 = img2[:l]
    img2_2 = img2[l : 2 * l]
    img2_3 = img2[2 * l :]

    img_r = put_data(img[:, :, 0], img2_1, np.uint8(12))
    img_g = put_data(img[:, :, 1], img2_2, np.uint8(12))
    img_b = put_data(img[:, :, 2], img2_3, np.uint8(12))

    r = pop_data(img_r, np.uint8(12), out_shape=img2_1.shape)
    g = pop_data(img_g, np.uint8(12), out_shape=img2_2.shape)
    b = pop_data(img_b, np.uint8(12), out_shape=img2_3.shape)

    vector = np.concatenate((r.flatten(), g.flatten(), b.flatten()))

    new_img2 = vector.reshape(shape)
    plt.imshow(new_img2)
    plt.show()

    new_img1 = np.empty_like(img)
    new_img1[:, :, 0] = img_r
    new_img1[:, :, 1] = img_g
    new_img1[:, :, 2] = img_b

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original image")
    axs[0].axis("off")
    axs[1].imshow(new_img1)
    axs[1].set_title("with hidden data")
    axs[1].axis("off")
    plt.show()
    

    # mask = np.array(
    #     [
    #         ord(c)
    #         for c in "Hello, World!Hello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, WorldHello, World"
    #     ],
    #     dtype=np.uint8,
    # )
    # # mask = np.unpackbits(mask)
    # print(mask)
    # img_r = put_data(img[:, :, 0], mask, np.uint8(1))
    # r = pop_data(img_r, np.uint8(1), out_shape=(1, len(mask)))
    # print(r)
    # # print(r)
    # # print(mask[:200, :200, 0] - r)
    # # # print(r[:10])
    # img[:, :, 0] = img_r
    # plt.imshow(img)
    # plt.show()


if __name__ == "__main__":
    main()
    # test()

    # text = "Hello, World!"

    # # Konwersja łańcucha tekstowego na tablicę bajtów
    # byte_array = np.frombuffer(text.encode(), dtype=np.uint8)
    # print(byte_array)
    # print([ord(c) for c in text])
