import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("IMG_INTRO/B01.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

O = img.copy()
R1 = img.copy()
R2 = img.copy()
G1 = img.copy()
G2 = img.copy()
B1 = img.copy()
B2 = img.copy()

R = img[:, :, 0].copy()
G = img[:, :, 1].copy()
B = img[:, :, 2].copy()

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

plt.show()
