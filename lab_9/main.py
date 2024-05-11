import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = "./VIDEOS"  # katalog z plikami wideo
plik = "clip_5.mp4"  # nazwa pliku
ile = 450  # ile klatek odtworzyć? <0 - całość
key_frame_counter = 4  # co która klatka ma być kluczowa i nie podlegać kompresji
plot_frames = np.array([349, 439])  # automatycznie wyrysuj wykresy
# auto_pause_frames = np.array([25])  # automatycznie za pauzuj dla klatki
auto_pause_frames = np.array([])
# subsampling = "4:1:0"  # parametry dla chroma subsampling
# dzielnik = 1  # dzielnik przy zapisie różnicy
wyswietlaj_klatki = True  # czy program ma wyświetlać klatki
ROI = [
    [600, 700, 420, 520],
]  # wyświetlane fragmenty (można podać kilka )
useRle = False  # czy używać kompresji RLE

save = True
subsamplings = ["4:4:4", "4:2:2", "4:4:0", "4:2:0", "4:1:1", "4:1:0"]
dzielniki = [1, 0.5, 0.25, 0.125]
output = "OUTPUT_diff/" + plik.split(".")[0] + "/"

if not save:
    dzielniki = [1]
    subsamplings = ["4:4:4"]


##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################


def RLE_encode(img):
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

    output = np.concatenate([shape, output])
    return output


def RLE_decode(data):
    if data[0] == 2:
        shape = data[1:3]
        data = data[3:]
    elif data[0] == 3:
        shape = data[1:4]
        data = data[4:]
    elif data[0] == 1:
        shape = data[1:2]
        data = data[2:]
    else:
        raise ValueError("Invalid data")

    output = np.empty(np.prod(shape), dtype=int)
    j = 0

    for i in range(0, len(data), 2):
        output[j : j + data[i + 1]] = data[i]
        j += data[i + 1]

    output = np.reshape(output, shape)

    return output


class data:
    def init(self):
        self.Y = None
        self.Cb = None
        self.Cr = None


def Chroma_subsampling(L, subsampling):
    if subsampling == "4:4:4":
        return L
    elif subsampling == "4:2:2":
        return L[:, ::2]
    elif subsampling == "4:4:0":
        return L[::2, :]
    elif subsampling == "4:2:0":
        return L[::2, ::2]
    elif subsampling == "4:1:1":
        return L[:, ::4]
    elif subsampling == "4:1:0":
        return L[::2, ::4]
    else:
        raise ValueError("Invalid subsampling parameter")


def Chroma_resampling(L, subsampling):
    if subsampling == "4:4:4":
        return L
    elif subsampling == "4:2:2":
        return np.repeat(L, 2, axis=1)
    elif subsampling == "4:4:0":
        return np.repeat(L, 2, axis=0)
    elif subsampling == "4:2:0":
        return np.repeat(np.repeat(L, 2, axis=0), 2, axis=1)
    elif subsampling == "4:1:1":
        return np.repeat(L, 4, axis=1)
    elif subsampling == "4:1:0":
        return np.repeat(
            np.repeat(L, 2, axis=0), 4, axis=1
        )  # byc moze trzeba na odwrot
    else:
        raise ValueError("Invalid subsampling parameter")


def frame_image_to_class(frame, subsampling):
    Frame_class = data()
    Frame_class.Y = frame[:, :, 0].astype(int)
    Frame_class.Cb = Chroma_subsampling(frame[:, :, 2].astype(int), subsampling)
    Frame_class.Cr = Chroma_subsampling(frame[:, :, 1].astype(int), subsampling)
    return Frame_class


def frame_layers_to_image(Y, Cr, Cb, subsampling):
    Cb = Chroma_resampling(Cb, subsampling)
    Cr = Chroma_resampling(Cr, subsampling)
    return np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)


def compress_KeyFrame(Frame_class):
    KeyFrame = data()

    KeyFrame.Y = Frame_class.Y
    KeyFrame.Cb = Frame_class.Cb
    KeyFrame.Cr = Frame_class.Cr

    if useRle:
        KeyFrame.Y = RLE_encode(KeyFrame.Y)
        KeyFrame.Cb = RLE_encode(KeyFrame.Cb)
        KeyFrame.Cr = RLE_encode(KeyFrame.Cr)

    return KeyFrame


def decompress_KeyFrame(KeyFrame):
    if useRle:
        KeyFrame.Y = RLE_decode(KeyFrame.Y)
        KeyFrame.Cb = RLE_decode(KeyFrame.Cb)
        KeyFrame.Cr = RLE_decode(KeyFrame.Cr)

    Y = KeyFrame.Y
    Cb = KeyFrame.Cb
    Cr = KeyFrame.Cr

    frame_image = frame_layers_to_image(Y, Cr, Cb, subsampling)
    return frame_image


def compress_not_KeyFrame(
    Frame_class, KeyFrame, dzielnik, inne_paramerty_do_dopisania=None
):
    Compress_data = data()

    Compress_data.Y = ((Frame_class.Y - KeyFrame.Y) * dzielnik).astype(np.int32)
    Compress_data.Cb = ((Frame_class.Cb - KeyFrame.Cb) * dzielnik).astype(np.int32)
    Compress_data.Cr = ((Frame_class.Cr - KeyFrame.Cr) * dzielnik).astype(np.int32)

    if useRle:
        Compress_data.Y = RLE_encode(Compress_data.Y)
        Compress_data.Cb = RLE_encode(Compress_data.Cb)
        Compress_data.Cr = RLE_encode(Compress_data.Cr)

    return Compress_data


def decompress_not_KeyFrame(
    Compress_data, KeyFrame, dzielnik, inne_paramerty_do_dopisania=None
):

    if useRle:
        Compress_data.Y = RLE_decode(Compress_data.Y)
        Compress_data.Cb = RLE_decode(Compress_data.Cb)
        Compress_data.Cr = RLE_decode(Compress_data.Cr)

    Y = Compress_data.Y
    Cb = Compress_data.Cb
    Cr = Compress_data.Cr

    Y = (Y / dzielnik) + KeyFrame.Y
    Cb = (Cb / dzielnik) + KeyFrame.Cb
    Cr = (Cr / dzielnik) + KeyFrame.Cr

    return frame_layers_to_image(Y, Cr, Cb, subsampling)


def plotDifference(
    ReferenceFrame, DecompressedFrame, ROI, dzielnik, subsampling, klatka
):
    # bardzo słaby i sztuczny przykład wykorzystania tej opcji
    # przerobić żeby porównanie było dokonywane w RGB nie YCrCb i/lub zastąpić innym porównaniem
    # ROI - Region of Insert współrzędne fragmentu który chcemy przybliżyć i ocenić w formacie [w1,w2,k1,k2]

    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(16, 5)
    fig.suptitle(f"subsampling={subsampling}, divider={1/dzielnik}")

    ReferenceFrame = cv2.cvtColor(ReferenceFrame, cv2.COLOR_YCrCb2RGB)
    DecompressedFrame = cv2.cvtColor(DecompressedFrame, cv2.COLOR_YCrCb2RGB)

    axs[0].imshow(ReferenceFrame[ROI[0] : ROI[1], ROI[2] : ROI[3]])
    axs[2].imshow(DecompressedFrame[ROI[0] : ROI[1], ROI[2] : ROI[3]])
    diff = ReferenceFrame[ROI[0] : ROI[1], ROI[2] : ROI[3]].astype(
        float
    ) - DecompressedFrame[ROI[0] : ROI[1], ROI[2] : ROI[3]].astype(float)
    # print(np.min(diff), np.max(diff))

    # to dodalem zeby nie wyskakiwalo ostrzezenie, zobaczyc jaka jest roznica bez tego
    # diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    # axs[1].imshow(diff)

    axs[1].imshow(diff, vmin=np.min(diff), vmax=np.max(diff))

    if save:
        plt.savefig(
            output
            + f"F_{klatka}_S_{subsampling.replace(':', '_')}_D_{dzielnik}_ROI_{ROI[0]}_{ROI[1]}_{ROI[2]}_{ROI[3]}_.png"
        )


##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################
for dzielnik in dzielniki:
    for subsampling in subsamplings:
        cap = cv2.VideoCapture(kat + "\\" + plik)

        if ile < 0:
            ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow("Normal Frame")
        cv2.namedWindow("Decompressed Frame")

        compression_information = np.zeros((3, ile))

        for i in range(ile):
            ret, frame = cap.read()
            if wyswietlaj_klatki:
                cv2.imshow("Normal Frame", frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            Frame_class = frame_image_to_class(frame, subsampling)
            if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
                KeyFrame = compress_KeyFrame(Frame_class)
                cY = KeyFrame.Y
                cCb = KeyFrame.Cb
                cCr = KeyFrame.Cr
                Decompresed_Frame = decompress_KeyFrame(KeyFrame)
            else:  # kompresja
                Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame, dzielnik)
                cY = Compress_data.Y
                cCb = Compress_data.Cb
                cCr = Compress_data.Cr
                Decompresed_Frame = decompress_not_KeyFrame(
                    Compress_data, KeyFrame, dzielnik
                )

            compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[
                :, :, 0
            ].size
            compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[
                :, :, 0
            ].size
            compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[
                :, :, 0
            ].size
            if wyswietlaj_klatki:
                cv2.imshow(
                    "Decompressed Frame",
                    cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR),
                )

            if np.any(plot_frames == i):  # rysuj wykresy
                for r in ROI:
                    plotDifference(
                        frame, Decompresed_Frame, r, dzielnik, subsampling, i
                    )

            if np.any(auto_pause_frames == i):
                cv2.waitKey(-1)  # wait until any key is pressed

            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break
            elif k == ord("p"):
                cv2.waitKey(-1)  # wait until any key is pressed

plt.figure()
plt.plot(np.arange(0, ile), compression_information[0, :] * 100)
plt.plot(np.arange(0, ile), compression_information[1, :] * 100)
plt.plot(np.arange(0, ile), compression_information[2, :] * 100)
plt.title(
    "File:{}, subsampling={}, divider={}, KeyFrame={} ".format(
        plik, subsampling, dzielnik, key_frame_counter
    )
)

if not save:
    plt.show()
