import numpy as np
import sys
import matplotlib.pyplot as plt


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


def coder_RLE(img):
    shape = np.array([len(img.shape)])
    shape = np.concatenate([shape, img.shape])

    img = img.flatten()

    output = np.empty(np.prod(img.shape) * 2, dtype=int)
    j = 0
    count = 1

    for i in range(1, len(img)):
        if img[i] == img[i - 1]:
            count += 1
        else:
            output[[j, j + 1]] = img[i - 1], count
            j += 2
            count = 1

    output[[j, j + 1]] = img[-1], count
    output = output[: j + 2]
    # print("Size of original image: ", get_size(img))
    # print("Size of RLE image: ", get_size(output))
    # print("Compression ratio: ", get_size(img)/get_size(output))
    # print("Compression percentage: ", (get_size(img)-get_size(output))/get_size(img)*100)
    # print(shape, shape.shape, output.shape, output.shape[0])
    # print(output)
    output = np.concatenate([shape, output])
    return output


def decoder_RLE(data):
    if data[0] == 2:
        shape = data[1:3]
        data = data[3:]
    elif data[0] == 3:
        shape = data[1:4]
        data = data[4:]
    else:
        raise ValueError("Invalid data")

    output = []
    output_arr = np.empty(len(data) // 2, dtype=int)
    j = 0
    for i in range(0, len(data), 2):
        output.extend([data[i]] * data[i + 1])
        output_arr[j:j + data[i + 1]] = data[i]
        j += data[i + 1]
        

    output = np.array(output)
    print(output.shape, output_arr.shape)
    output = np.reshape(output, shape)
    # output_arr = np.reshape(output_arr, shape)

    return output


def coder_ByteRun(img):
    pass

def decoder_ByteRun(data):
    pass


def main():
    # img = cv2.imread("img/img_01.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = plt.imread("img/img_01.jpg")
    img = img.astype(int)

    data = coder_RLE(img)
    new_img = decoder_RLE(data)

    if np.array_equal(img, new_img):
        print("img equals new_img")
    else:
        print("img does not equal new_img")

    # plt.imshow(new_img)
    # plt.show()


if __name__ == "__main__":
    main()
