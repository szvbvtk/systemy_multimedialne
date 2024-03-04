import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an image
img1 = plt.imread("IMG_INTRO/B01.png")
# or
# img1 = cv2.imread("IMG_INTRO/A1.png")

# png - float32, jpg - uint8
print(img1.dtype)
print(img1.shape)
print(np.min(img1), np.max(img1))


