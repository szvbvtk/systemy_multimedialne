import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as ssim
import cv2
import csv

output_dir = Path("./output")


def read_image(image_path):
    image = cv2.imread(str(image_path))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# Obiektywne miary jakości
def MSE(source_image, target_image):
    return np.mean((source_image - target_image) ** 2)


def NMSE(source_image, target_image):
    return MSE(source_image, target_image) / MSE(
        target_image, np.zeros_like(target_image)
    )


def PSNR(source_image, target_image, max_value=255):
    return 10 * np.log10(max_value**2 / MSE(source_image, target_image))


def IF(source_image, target_image):
    # ou can use the numpy. multiply() function to perform element-wise multiplication with two-dimensional arrays. For instance, numpy. multiply() performs element-wise multiplication on the two 2D arrays arr and arr1 , resulting in the output array [[ 4 12 30 32], [ 8 15 15 14]] .
    #    sprawdzic czy to dziala jak powinno
    return 1 - (np.sum((source_image - target_image) ** 2)) / np.sum(
        np.multiply(source_image, target_image)
    )


def SSIM(source_image, target_image):
    return ssim(source_image, target_image, channel_axis=2)


def measure_quality(source_image, target_image):
    mse = MSE(source_image, target_image)
    nmse = NMSE(source_image, target_image)
    psnr = PSNR(source_image, target_image)
    if_ = IF(source_image, target_image)
    ssim = SSIM(source_image, target_image)

    return [mse, nmse, psnr, if_, ssim]


def calculate_norms():
    image_path = Path("./img/1.jpg")

    image = read_image(image_path)

    stats = []
    paths = sorted(output_dir.glob("*.jpg"), key=lambda x: int(x.stem.split("_")[0]))
    for i, path in enumerate(paths):
        modified_image = read_image(path)
        quality = f"{path.stem.split('_')[-1]}%"
        stats.append([f"image_{i+1}"] + measure_quality(image, modified_image))

    df = pd.DataFrame(stats, columns=["image_id", "MSE", "NMSE", "PSNR", "IF", "SSIM"])
    df = df.set_index("image_id")

    df.to_csv(output_dir / "norms.csv")


def generate_pairs(norm, MOS):
    All = []
    MeanPerPerson = []

    for i in range(MOS.shape[0]):
        for j in range(MOS.shape[1]):
            All.append([norm[i], MOS[i][j]])

    print(All)


# def create_random_answers():
#     import random

#     list = []
#     for i in range(30):
#         inner_list = []
#         for j in range(15):
#             inner_list.append(random.randint(1, 5))

#         list.append(inner_list)

#     columns = [f"image_{i+1}" for i in range(15)]
#     id = [f"person_{i}_0" for i in range(10)]
#     id += [f"person_{i}_1" for i in range(10)]
#     id += [f"person_{i}_2" for i in range(10)]

#     df = pd.DataFrame(list, columns=columns)
#     df.index = id
#     df.index.name = "person"

#     df.to_csv(output_dir / "_results.csv")


def create_random_answers():
    import random

    persons = [f"person_{i}" for i in range(10)]
    persons += [f"person_{i}" for i in range(10)]
    persons += [f"person_{i}" for i in range(10)]

    list = []
    for i in range(30):
        inner_list = []
        inner_list.append(persons[i])
        for j in range(15):
            inner_list.append(random.randint(1, 5))

        list.append(inner_list)

    columns = ["person"] + [f"image_{i+1}" for i in range(15)]

    df = pd.DataFrame(list, columns=columns)
    df.index.name = "id"

    df.to_csv(output_dir / "_results.csv")


def test():
    answers = pd.read_csv(output_dir / "_results.csv", index_col="id").iloc[:, 0:5]
    # 3
    # answers = answers.groupby("person").mean()
    # answers = answers.transpose()
    # print(answers)

    answers = answers.transpose()
    answers.columns = answers.iloc[0]
    answers = answers.drop(["person"])
    answers.index.name = "image_id"
    # print(answers.index)
    # answers = answers.drop(["person"])
    # answers = answers.
    # print(answers.columns)
    # answers.index.name = "image_id"
    answers = answers.mean(axis=1)
    print(answers)


def main():
    norms = pd.read_csv(output_dir / "norms.csv", index_col="image_id")
    # sns.pairplot(norms)
    # plt.show()
    answers = pd.read_csv(output_dir / "_results.csv", index_col="person")
    answers = answers.transpose()
    answers.index.name = "image_id"
    # print(answers.values)

    data = pd.concat([norms, answers], axis=1)
    print(answers.columns.tolist())

    # generate_pairs(norms.iloc[:, 1].values, answers.values)


if __name__ == "__main__":
    # calculate_norms()
    # create_random_answers()

    # main()
    test()
