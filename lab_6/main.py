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
    j += 2
    output = output[:j]
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

    output = np.empty(np.prod(shape), dtype=int)
    j = 0
    for i in range(0, len(data), 2):
        output[j : j + data[i + 1]] = data[i]
        j += data[i + 1]

    output = np.reshape(output, shape)

    return output


def find_repeating_elements_sequence(arr, start):
    counter = 1
    for i in range(start, len(arr) - 1):
        if arr[i] == arr[i + 1]:
            counter += 1
        else:
            break

    return counter


def find_different_elements_sequence(arr, start):
    counter = 1
    for i in range(start, len(arr) - 1):
        if arr[i] != arr[i + 1]:
            counter += 1
        else:
            break

    return counter


def coder_ByteRun(img):
    shape = np.array([len(img.shape)])
    shape = np.concatenate([shape, img.shape])

    img = img.flatten()

    output = np.empty(np.prod(img.shape) * 2, dtype=int)

    i = 0
    j = 0
    while i < len(img):
        repeating = True
        if img[i] == img[i + 1]:
            count = find_repeating_elements_sequence(img, i)
        else:
            count = find_different_elements_sequence(img, i)
            repeating = False

        if repeating:
            output[j] = count
            output[j + 1] = img[i]
            j += 2
        else:
            output[j] = -count
            output[j + 1 : j + 1 + count] = img[i : i + count]

            j += count + 1

        i += count

    output = output[:j]
    output = np.concatenate([shape, output])

    return output


def decoder_ByteRun(data):
    if data[0] == 2:
        shape = data[1:3]
        data = data[3:]
    elif data[0] == 3:
        shape = data[1:4]
        data = data[4:]
    else:
        raise ValueError("Invalid data")

    output = np.empty(np.prod(shape), dtype=int)

    i = 0
    j = 0
    while i < len(data):
        count = data[i]
        repeating = count > 0
        count = abs(count)

        i += 1
        if repeating:
            output[j : j + count] = data[i]
            j += count
            i += 1
        else:
            output[j : j + count] = data[i : i + count]
            j += count
            i += count

    output = np.reshape(output, shape)

    return output
            


def main():

    img = plt.imread("img/img_01.jpg")
    img = img.astype(int)

    data = coder_RLE(img)
    new_img = decoder_RLE(data)

    data2 = coder_ByteRun(img)
    new_img2 = decoder_ByteRun(data2)


    if np.array_equal(img, new_img):
        print("img equals new_img")
    else:
        print("img does not equal new_img")

    if np.array_equal(img, new_img2):
        print("img equals new_img2")
    else:
        print("img does not equal new_img2")


    plt.imshow(new_img2)
    plt.show()


if __name__ == "__main__":
    main()
