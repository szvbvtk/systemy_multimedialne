import numpy as np


def imgToUInt8(img):
    if np.issubdtype(img.dtype, np.unsignedinteger):
        return img
    elif np.issubdtype(img.dtype, np.floating):
        return (img * 255).astype(np.uint8)

    raise ValueError("Unsupported image type")


def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        return img
    elif np.issubdtype(img.dtype, np.unsignedinteger):
        return img.astype(np.float32) / 255
        # return img / float(255)

    raise ValueError("Unsupported image type")
