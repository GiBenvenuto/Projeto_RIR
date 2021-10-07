import numpy as np
import cv2, glob, csv
from Utils import plots as p
import matplotlib.pyplot as plt




def read_originals_control_points(cat, ind):
    if (cat == 2):
        ind = ind + 14
    elif (cat == 1):
        ind = ind + 63

    path = glob.glob("DataBases/FIRE/Ground_Truth/*.txt")
    file = open(path[ind])
    cps = csv.reader(file)
    A = []

    for lines in cps:
        line = lines[0]
        A.append(list(map(float, line.split(" "))))

    A = np.array(A)

    pts_ref = A[:,0:2]
    pts_warp = A[:,2:4]

    return pts_ref.astype(int), pts_warp.astype(int)

def read_control_points(cat, ind):
    if (cat == 2):
        ind = ind + 14
    elif (cat == 1):
        ind = ind + 63

    path = glob.glob("DataBases/FIRE/Ground_Truth_512/*.txt")
    path = sorted(path)
    file = open(path[ind])
    cps = csv.reader(file)
    A = []

    for lines in cps:
        line = lines[0]
        A.append(list(map(float, line.split(" "))))

    A = np.array(A)

    pts_ref = A[:,0:2]
    pts_warp = A[:,2:4]

    return pts_ref.astype(int), pts_warp.astype(int)


def add_point(m, x, y):
    m = cv2.circle(m, (x, y), 5, color= 255, thickness=-1)

    return m


def create_mask(pts, size):

    mask = np.zeros((size, size))
    for i in range(10):
        l = pts[i][0]
        c = pts[i][1]
        mask = add_point(mask, l, c)

    #mask = mask/255
    #mask = mask.reshape(mask.shape + (1,))
    return mask


if __name__ == "__main__":
    ref = plt.imread("01_y.tif")
    mov = plt.imread("01_x.tif")

    #pts, wa = read_originals_control_points(2,0)
    pts2, wa2, = read_control_points(2,0)
    mask = create_mask(pts2, 512)
    cv2.imwrite("mask_01_x.png", mask)
    mask2 = create_mask(wa2, 512)
    cv2.imwrite("mask_01_y.png", mask2)

    p.plot_img_gt(ref, pts2, 'g')

    p.plot_img_gt(mov, wa2, 'm')

    cv2.waitKey(0)




