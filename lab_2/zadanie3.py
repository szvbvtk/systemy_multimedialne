import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# można zostawić tylko dla jednego obrazka do sprawo, nie trzeba wszystkich
img = cv2.imread("IMG_INTRO/B02.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(img.shape)
fragment = img[200:400, 200:400].copy()

format = "jpg"
input_dir = "IMG_INTRO/"
output_dir = "zadanie_3_img"
df = pd.DataFrame(
    data={
        "Filename": ["B02.jpg", "B01.png"],
        "Grayscale": [False, False],
        "Fragments": [[[200, 200, 400, 400], [20, 20, 60, 60]], [[200, 200, 400, 400], [20, 20, 60, 60]]],
    }
)
print(df)

for index, row in df.iterrows():
    img = cv2.imread(input_dir + row["Filename"])

    if row["Grayscale"]:
        pass
    else:
        if row["Fragments"] is not None:
            for f in row["Fragments"]:
                fragment = img[f[0] : f[2], f[1] : f[3]].copy()

                # operacje
                O = fragment.copy()
                R1 = fragment.copy()
                R2 = fragment.copy()
                G1 = fragment.copy()
                G2 = fragment.copy()
                B1 = fragment.copy()
                B2 = fragment.copy()

                R = fragment[:, :, 0].copy()
                G = fragment[:, :, 1].copy()
                B = fragment[:, :, 2].copy()

                Y1 = 0.299 * R + 0.587 * G + 0.114 * B
                Y2 = 0.2126 * R + 0.7152 * G + 0.0722 * B

                R1 = R1[:, :, 0]
                G1 = G1[:, :, 1]
                B1 = B1[:, :, 2]

                R2[:, :, 1:3] = 0
                G2[:, :, [0, 2]] = 0
                B2[:, :, 0:2] = 0

                fig, axs = plt.subplots(3, 3, figsize=(15, 15))

                axs[0, 0].imshow(O)
                axs[0, 1].imshow(Y1, cmap="gray")
                axs[0, 2].imshow(Y2, cmap="gray")

                axs[1, 0].imshow(R1, cmap="gray")
                axs[1, 1].imshow(G1, cmap="gray")
                axs[1, 2].imshow(B1, cmap="gray")

                axs[2, 0].imshow(R2)
                axs[2, 1].imshow(G2)
                axs[2, 2].imshow(B2)

                plt.savefig(
                    f"{output_dir}/{row['Filename']}_fragment_{f[0]}_{f[1]}_{f[2]}_{f[3]}.{format}"
                )
