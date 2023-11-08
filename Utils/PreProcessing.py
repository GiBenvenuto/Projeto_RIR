import numpy as np
import cv2
from Segmentation import segmentation
import csv
import glob

import matplotlib.pyplot as plt

def preprocessing(path):
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]


    if (not(height == width)):
        new_img = crop(img, height, width)

    seg_img = segmentation(new_img)
    new_img = cv2.resize(seg_img, (512, 512), cv2.INTER_CUBIC)

    #cv2.imshow("Teste", new_img)
    #cv2.waitKey(0)


    return new_img

def crop(img, height, width):
    cy = height/2
    cx = width/2

    if (width > height):
        crop_img = img[0:int(height), int(cx-cy):int(cx+cy)]
    else:
        crop_img = img[int(cy-cx):int(cy+cx), 0:int(width)]
    return crop_img

if __name__ == "__main__":
    paths = glob.glob("../DataBases/DRIVE/test/images/*.tif")
    j = 1
    k = 1


    for i in paths:
        img = cv2.imread(i)
        new_img = crop(img, img.shape[0], img.shape[1])
        #if j % 2 == 0:
        cv2.imwrite("../DataBases/DRIVE/test/croped_images/{:02d}_test.png".format(j), new_img)
        #else:
        #   cv2.imwrite("../DataBases/allQuality/croped_images/{:02d}_good.png".format(k), new_img)
        #    k = k + 1
        j = j + 1
