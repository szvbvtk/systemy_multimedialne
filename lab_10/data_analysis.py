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
    paths = sorted(output_dir.glob("*.png"), key=lambda x: int(x.stem.split("_")[0]))
    for i, path in enumerate(paths):
        modified_image = read_image(path)
        quality = f"{path.stem.split('_')[-1]}%"
        stats.append([f"image_{i+1}"] + measure_quality(image, modified_image))

    df = pd.DataFrame(stats, columns=["image_id", "MSE", "NMSE", "PSNR", "IF", "SSIM"])
    df = df.set_index("image_id")
    df = df.round(decimals=2)

    df.to_csv(output_dir / "norms.csv")


def create_random_answers():
    import random

    list = []
    for i in range(30):
        inner_list = []
        for j in range(15):
            inner_list.append(random.randint(1, 5))

        list.append(inner_list)

    columns = [f"image_{i+1}" for i in range(15)]
    id = [f"person_{i}_0" for i in range(10)]
    id += [f"person_{i}_1" for i in range(10)]
    id += [f"person_{i}_2" for i in range(10)]

    df = pd.DataFrame(list, columns=columns)
    df.index = id
    df.index.name = "person"

    df.to_csv(output_dir / "_results.csv")


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
    answers = pd.read_csv(output_dir / "answers.csv", index_col="id").iloc[:, 0:5]
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


def generate_pairs(norm, answers):
    answers_per_image = answers.mean(axis=1).round(decimals=2).values  # mean
    answers_per_person = (
        answers.groupby(answers.columns, axis=1).mean().round(decimals=2).values
    )  # mean

    norm = norm.tolist()
    answers = answers.values

    All = []
    MeanPerPerson = []
    MeanPerImage = []

    for i in range(answers.shape[0]):
        for j in range(answers.shape[1]):
            All.append([norm[i], answers[i, j]])

    for i in range(answers_per_person.shape[0]):
        for j in range(answers_per_person.shape[1]):
            MeanPerPerson.append([norm[i], answers_per_person[i, j]])

    for i in range(answers_per_image.shape[0]):
        MeanPerImage.append([norm[i], answers_per_image[i]])

    return All, MeanPerPerson, MeanPerImage


def draw_plot(All, MeanPerPerson, MeanPerImage):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    All = np.array(All)
    MeanPerPerson = np.array(MeanPerPerson)
    MeanPerImage = np.array(MeanPerImage)

    sns.scatterplot(x=All[:, 0], y=All[:, 1], ax=axs[0])
    sns.scatterplot(x=MeanPerPerson[:, 0], y=MeanPerPerson[:, 1], ax=axs[1])
    sns.scatterplot(x=MeanPerImage[:, 0], y=MeanPerImage[:, 1], ax=axs[2])

    plt.show()


def main():

    norms = pd.read_csv(output_dir / "norms.csv", index_col="image_id")
    answers = pd.read_csv(output_dir / "answers.csv", index_col="Nazwa użytkownika")

    fct = pd.factorize(answers.index)
    answers.index = fct[0]

    answers = answers.drop(columns=["Sygnatura czasowa"])
    num_columns = len(answers.columns)
    answers = answers.transpose()
    answers.index.name = "image_id"
    answers.index = np.arange(1, num_columns + 1)
    # print(answers)

    answers_mean_per_image = answers.mean(axis=1).round(decimals=2)
    answers_mean_per_person = (
        answers.groupby(answers.columns, axis=1).mean().round(decimals=2)
    )
    print(answers_mean_per_image)
    # print(answers_mean_per_person)
    # print(norms.index.tolist())
    # print(answers.values)
    # print()
    # answers_array = answers.values
    # print(answers_array)

    All, MeanPerPerson, MeanPerImage = generate_pairs(norms.index, answers)
    draw_plot(All, MeanPerPerson, MeanPerImage)


if __name__ == "__main__":
    calculate_norms()
    # create_random_answers()

    main()
    # test()
