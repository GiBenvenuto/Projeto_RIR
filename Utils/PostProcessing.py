import cv2
import numpy as np
import glob

def post_processing(path, cat, ind):
    image = cv2.imread(path)
    gray = image[:,:,1]
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    mask = np.zeros(gray.shape, dtype="uint8")
    for i in range(0, numLabels):

        '''if i == 0:
             text = "examining component {}/{} (background)".format(i + 1, numLabels)
        else:
             text = "examining component {}/{}".format( i + 1, numLabels)
             print("[INFO] {}".format(text))'''

        area = stats[i, cv2.CC_STAT_AREA]

        if (area < 10):
            #print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)


    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)


    pos = cv2.bitwise_and(mask, thresh)
    #cv2.imshow("Image", thresh)
    #cv2.imshow("Characters", mask)
    #cv2.imshow("Pos", pos)
    cv2.imwrite("TESTES/Teste_512_1603_cc_allcats_inv/" + str(cat) + "/{:02d}_z.tif".format(ind + 1), pos)
    #cv2.waitKey(0)


if __name__ == "__main__":
    paths = glob.glob("TESTES/Teste_512_1603_allcats_inv/2/*.tif")
    cont = 0

    for i in range(2, len(paths), 3):
        post_processing(paths[i], 2, cont)
        cont = cont + 1









