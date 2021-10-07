import numpy as np
import cv2, glob, csv
import statistics as s
from matplotlib import pyplot as plt
import scipy.spatial.distance as sd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import wx
from PIL import Image


import tensorflow as tf

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


def load_bin(path, size):
    src = cv2.imread(path)
    src = cv2.resize(src, (size, size), cv2.INTER_CUBIC)
    src = src[:, :, 1]
    ret, src = cv2.threshold(src, 80, 255, cv2.THRESH_BINARY)
    #cv2.imshow("teste", src)
    #cv2.waitKey(0)
    src = src / 255

    return src


def load_test(paths_ref, paths_pro, size):
    srcs = []
    tgts = []

    for i in range(0, len(paths_ref)):
        src = load_bin(paths_ref[i], size)
        srcs.append(src)
        tgt = load_bin(paths_pro[i], size)
        tgts.append(tgt)

    return srcs, tgts


def mean_results(paths_ref, paths_tgt, func, size):
    src, tgt = load_test(paths_ref, paths_tgt, size)
    mean_data = []
    for i in range(len(src)):
        mean_data.append(func(src[i], tgt[i]))

    mean = s.mean(mean_data)
    print('{:.4f}'.format(mean))
    var = s.pstdev(mean_data, mu=mean)
    print('{:.4f}'.format(var))
    return mean, var


def main(size):
    paths_ref = glob.glob("../DataBases/Segmentadas/A_s_fix/*.jpg")
    paths_tgt = glob.glob("../DataBases/Teste/GFEMR_inv/A_s_inv/*.jpg")

    print("A - MSE")
    mean_results(paths_ref, paths_tgt, mse_distance, size)
    print("A - SSIM")
    mean_results(paths_ref, paths_tgt, ssim_distance, size)
    print("A - DICE BIN")
    mean_results(paths_ref, paths_tgt, dice_bin, size)

    paths_ref = glob.glob("../DataBases/Segmentadas/S_s_fix/*.jpg")
    paths_tgt = glob.glob("../DataBases/Teste/GFEMR_inv/S_s_inv/*.jpg")

    print("\n")
    print("S - MSE")
    mean_results(paths_ref, paths_tgt, mse_distance, size)
    print("S - SSIM")
    mean_results(paths_ref, paths_tgt, ssim_distance, size)
    print("S - DICE BIN")
    mean_results(paths_ref, paths_tgt, dice_bin, size)

    paths_ref = glob.glob("../DataBases/Teste/Ref/P_s_fix/*.jpg")
    paths_tgt = glob.glob("../DataBases/Teste/GFEMR_inv/P_s_inv/*.jpg")

    print("\n")
    print("P - MSE")
    mean_results(paths_ref, paths_tgt, mse_distance, size)
    print("P - SSIM")
    mean_results(paths_ref, paths_tgt, ssim_distance, size)
    print("P - DICE BIN")
    mean_results(paths_ref, paths_tgt, dice_bin, size)

def colors_compare(path_green, path_mag, path_save):
    #path_green = "../DataBases/Teste/New_Comp/07_y_green.png"
    #path_mag = "../DataBases/Teste/New_Comp/Rempe_A07_z_mag.png"
    y = cv2.imread(path_green)
    y = cv2.resize(y, (512, 512), cv2.INTER_CUBIC)
    x = cv2.imread(path_mag)
    x = cv2.resize(x, (512, 512), cv2.INTER_CUBIC)
    comp = y + x
    cv2.imwrite(path_save, comp)


def colors(c, path, path_save):
    # Load the aerial image and convert to HSV colourspace
    image = cv2.imread(path)
    #image = cv2.bitwise_not(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([0, 0, 25])
    brown_hi = np.array([0, 0, 255])

    # Mask image to only select browns
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to red where we found brown
    if (c):#green
        image[mask > 0] = (0, 255, 0)
    else:
        image[mask > 0] = (255, 0, 255)

    cv2.imwrite(path_save, image)

def compare ():
    #paths_ref = "../DataBases/Teste/Segmentadas/S_mov/S01_2.jpg"
    #paths_tgt = "../teste-segnew/1/01_z.tif"
    src = load_bin('../DataBases/Teste/Comp/Rempe/P01_1.jpg')
    #src = cv2.resize(src, (512, 512))
    tgt = load_bin('../DataBases/Teste/Comp/Rempe/p1.png')
    d = dice_bin(src, tgt)
    s = ssim_distance(src, tgt)
    m = mse_distance(src, tgt)
    print("Dice")
    print(d)
    print("SSIM")
    print(s)
    print("MSE")
    print(m)

def all_magenta():
    path = glob.glob("../DataBases/Segmentadas/P_s_mov/*.jpg")

    i = 0
    for p in path:
        ps = "../DataBases/Teste/Mov_mag/P_mag/{:02d}_z.png".format(i + 1)
        colors(False, p, ps)
        i = i + 1

def all_comp():
    paths_g = glob.glob("../DataBases/Teste/Ref_green/P_fix/*.png")
    paths_m = glob.glob("../DataBases/Teste/Mov_mag/P_mag/*.png")

    i = 0
    for g, m in zip(paths_g, paths_m):
        ps = "../DataBases/Teste/Before/P_comp/{:02d}_z.png".format(i + 1)
        colors_compare(g, m, ps)
        i = i + 1

if __name__ == "__main__":
    all_magenta()
    all_comp()
    #main(512)


