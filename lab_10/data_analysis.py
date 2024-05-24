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
        answers.groupby(answers.columns, axis=1).mean().round(decimals=2)
    )  # mean

    columns_A = answers.columns
    columns_P = answers_per_person.columns

    answers = answers.values
    answers_per_person = answers_per_person.values

    All = []
    MeanPerPerson = []
    MeanPerImage = []

    for i in range(answers.shape[0]):
        for j in range(answers.shape[1]):
            All.append([norm[i], answers[i, j], columns_A[j]])

    for i in range(answers_per_person.shape[0]):
        for j in range(answers_per_person.shape[1]):
            MeanPerPerson.append([norm[i], answers_per_person[i, j], columns_P[j]])

    for i in range(answers_per_image.shape[0]):
        MeanPerImage.append([norm[i], answers_per_image[i]])

    return (
        np.array(All, dtype=np.float32),
        np.array(MeanPerPerson, dtype=np.float32),
        np.array(MeanPerImage, dtype=np.float32),
    )


def draw_plot(All, MeanPerPerson, MeanPerImage, predictions, X):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    persons_A = np.array(All[:, 2], dtype=np.uint8)
    persons_P = np.array(MeanPerPerson[:, 2], dtype=np.uint8)

    All = np.array(All)
    MeanPerPerson = np.array(MeanPerPerson)
    MeanPerImage = np.array(MeanPerImage)

    sns.scatterplot(x=All[:, 0], y=All[:, 1], ax=axs[0], style=persons_A)
    axs[0].plot(X[0], predictions[0], color="magenta")
    sns.scatterplot(x=MeanPerPerson[:, 0], y=MeanPerPerson[:, 1], ax=axs[1], style=persons_P)
    axs[1].plot(X[1], predictions[1], color="magenta")
    sns.scatterplot(x=MeanPerImage[:, 0], y=MeanPerImage[:, 1], ax=axs[2])
    axs[2].plot(X[2], predictions[2], color="magenta")

    axs[0].set_title("Wszystkie oceny")
    axs[0].set_xlabel("Miara jakości")
    axs[0].set_ylabel("MOS")
    axs[0].set_yticks(np.arange(1, 6))
    axs[1].set_yticks(np.arange(1, 6))
    axs[2].set_yticks(np.arange(1, 6))

    axs[1].set_title("Zagregowane dla użytkownika")
    axs[1].set_xlabel("Miara jakości")
    axs[1].set_ylabel("MOS")

    axs[2].set_title("Zagregowane")
    axs[2].set_xlabel("Miara jakości")
    axs[2].set_ylabel("MOS")

    # plt.show()

    return fig


def draw_heatmap(corr_matrix):
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="PiYG", fmt=".2f")
    plt.title("Correlation Matrix")
    # plt.show()
    # plt.savefig(output_dir / "plots/heatmap.png")
    return fig



def main():
    # transform data
    norms = pd.read_csv(output_dir / "norms.csv", index_col="image_id")
    norms.index = np.arange(1, len(norms.index) + 1)
    answers = pd.read_csv(output_dir / "answers.csv", index_col="Nazwa użytkownika")

    fct = pd.factorize(answers.index)
    answers.index = fct[0]

    answers = answers.drop(columns=["Sygnatura czasowa"])
    num_columns = len(answers.columns)
    answers = answers.transpose()
    answers.index.name = "image_id"
    answers.index = np.arange(1, num_columns + 1)

    # --------------------------------------------
    answers_mean_per_image = answers.mean(axis=1).round(decimals=2)

    for i in range(norms.shape[1]):
        norm = norms.iloc[:, i]
        norm = norm.tolist()
        All, MeanPerPerson, MeanPerImage = generate_pairs(norm, answers)

    norm = norms.index
    # norm = norms.iloc[:, 0]
    norm = norm.tolist()
    All, MeanPerPerson, MeanPerImage = generate_pairs(norm, answers)
    # number_of_persons = len(answers.columns)

    model_A = LinearRegression()
    model_P = LinearRegression()
    model_I = LinearRegression()

    model_A.fit(All[:, 0].reshape(-1, 1), All[:, 1])
    model_P.fit(MeanPerPerson[:, 0].reshape(-1, 1), MeanPerPerson[:, 1])
    model_I.fit(MeanPerImage[:, 0].reshape(-1, 1), MeanPerImage[:, 1])
    x_A = np.linspace(All[:, 0].min(), All[:, 0].max(), 100).reshape(-1, 1)
    x_P = np.linspace(
        MeanPerPerson[:, 0].min(), MeanPerPerson[:, 0].max(), 100
    ).reshape(-1, 1)
    x_I = np.linspace(MeanPerImage[:, 0].min(), MeanPerImage[:, 0].max(), 100).reshape(
        -1, 1
    )

    pred_A = model_A.predict(x_A.reshape(-1, 1))
    pred_P = model_P.predict(x_P.reshape(-1, 1))
    pred_I = model_I.predict(x_I.reshape(-1, 1))

    predictions = [pred_A, pred_P, pred_I]
    X = [x_A, x_P, x_I]

    p = pd.concat([answers_mean_per_image, norms], axis=1)
    p.columns = ["MOS", "MSE", "NMSE", "PSNR", "IF", "SSIM"]
    p.index.name = "image_id"

    corr_matrix = p.corr()

    hm = draw_heatmap(corr_matrix)
    fig = draw_plot(All, MeanPerPerson, MeanPerImage, predictions, X)


if __name__ == "__main__":
    # calculate_norms()
    # create_random_answers()

    main()
