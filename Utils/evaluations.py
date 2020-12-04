import numpy as np
import cv2, glob
import statistics as s
from matplotlib import pyplot as plt
import scipy.spatial.distance as sd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def dice_bin(src, tgt):
    denom = src.sum() + tgt.sum()
    score = 2 * (src*tgt).sum() / denom
    return score


def hausdorff(src, tgt):
    h = sd.directed_hausdorff(src, tgt)
    return h[0]

def ssim_distance(src, tgt):
    s = ssim(src, tgt)
    return s

def mse_distance(src, tgt):
    m = mse(src, tgt)
    return m

def load_bin(path):
    src = cv2.imread(path)
    src = src[:, :, 1]
    ret, src = cv2.threshold(src, 80, 255, cv2.THRESH_BINARY)
    src = src / 255
    return src


def load_test(path, ind):
    paths = glob.glob(path + str(ind) + "/*.jpg")
    #path_new = glob.glob("Morph_Filters_Op/" + str(ind) + "/*.jpg")
    srcs = []
    tgts = []

    for i in range(0, len(paths)):
        if (i%3 == 1):#Fixa
            src = load_bin(paths[i])
            srcs.append(src)
        if (i%3 == 2):#Reg
            tgt = load_bin(paths[i])
            tgts.append(tgt)

    return srcs, tgts

def grading():
    src = cv2.imread("../DataBases/Originais/S_mov/S01_2.jpg")
    #src = cv2.imread("../teste_dUnet_256_sel/0/01_y.tif")
    src = cv2.resize(src, (512,512), interpolation=cv2.INTER_CUBIC)
    src = src[:, :, 1]

    tgt = cv2.imread("../teste-ordem/1/01_z.tif")
    tgt = tgt[:, :, 1]

    s1 = src[0:64, 0:64]
    s2 = src[0:64, 64:128]
    s3 = src[0:64, 128:192]
    s4 = src[0:64, 192:256]

    s5 = src[64:128, 0:64]
    s6 = src[64:128, 64:128]
    s7 = src[64:128, 128:192]
    s8 = src[64:128, 192:256]

    s9 = src[128:192, 0:64]
    s10 = src[128:192, 64:128]
    s11 = src[128:192, 128:192]
    s12 = src[128:192, 192:256]

    s13 = src[192:256, 0:64]
    s14 = src[192:256, 64:128]
    s15 = src[192:256, 128:192]
    s16 = src[192:256, 192:256]

    t1 = tgt[0:64, 0:64]
    t2 = tgt[0:64, 64:128]
    t3 = tgt[0:64, 128:192]
    t4 = tgt[0:64, 192:256]

    t5 = tgt[64:128, 0:64]
    t6 = tgt[64:128, 64:128]
    t7 = tgt[64:128, 128:192]
    t8 = tgt[64:128, 192:256]

    t9 = tgt[128:192, 0:64]
    t10 = tgt[128:192, 64:128]
    t11 = tgt[128:192, 128:192]
    t12 = tgt[128:192, 192:256]

    t13 = tgt[192:256, 0:64]
    t14 = tgt[192:256, 64:128]
    t15 = tgt[192:256, 128:192]
    t16 = tgt[192:256, 192:256]

    nova1 = np.block([[s1, t2, s3, t4], [t5, s6, t7, s8], [s9, t10, s11, t12], [t13, s14, t15, s16]])
    nova2 = np.block([[t1, s2, t3, s4], [s5, t6, s7, t8], [t9, s10, t11, s12], [s13, t14, s15, t16]])
    plt.subplot(1,2,1),plt.imshow(nova1,'gray')
    plt.subplot(1,2,2),plt.imshow(nova2,'gray')
    plt.show()


def main():
    src, tgt = load_test("Morph_Filters_all/", 0)
    mean_data = []

    print("A")

    for i in range (len(src)):
        mean_data.append(dice_bin(src[i], tgt[i]))

    mean = s.mean(mean_data)
    print(mean)
    var = s.pstdev(mean_data, mu=mean)
    print(var)

    src, tgt = load_test("Morph_Filters_all/", 1)
    mean_data = []

    print("S")

    for i in range(len(src)):
        mean_data.append(dice_bin(src[i], tgt[i]))

    mean = s.mean(mean_data)
    print(mean)
    var = s.pstdev(mean_data, mu=mean)
    print(var)

    src, tgt = load_test("Morph_Filters_all/", 2)
    mean_data = []

    print("P")

    for i in range(len(src)):
        mean_data.append(dice_bin(src[i], tgt[i]))

    mean = s.mean(mean_data)
    print(mean)
    var = s.pstdev(mean_data, mu=mean)
    print(var)


if __name__ == "__main__":
    main()