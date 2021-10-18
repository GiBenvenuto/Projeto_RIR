import cv2
import numpy as np
import glob
from ops import mkdir
import skimage.io as sk
from Utils import evaluations as ev

def opening(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    img_op = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img_op

def closing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    img_op = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_op

def main():
    #mkdir("Morph_Filters")

    for i in range(3):
        mkdir("Morph_Filters_clo/" + str(i))
        aux = glob.glob("../TESTES/teste_512_postprocessing/" + str(i) + "/*.tif")
        for j in range(0, len(aux)):
            img = ev.load_bin(aux[j])

            #op = opening(img=img)
            op = closing(img=img)
            sk.imsave("Morph_Filters_clo/" + str(i) + "/" + "/{:02d}_z.png".format(j+1), op)



if __name__ == '__main__':
    main()