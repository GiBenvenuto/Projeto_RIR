import numpy as np
import IUWT_segmentation_for_testing as iuwt
import cv2

def segmentation (path):
    img = cv2.imread(path)
    II0=[]
    II0.append(img)
    num_pyr = 0

    while len(II0[-1])>900:
        II0.append(cv2.pyrDown(II0[-1]))
        num_pyr = num_pyr + 1

    II = cv2.cvtColor(II0[-1], cv2.COLOR_RGB2GRAY)
    #cv2.imshow("Teste", II)
    #cv2.waitKey(0)
    mask = (II > 5)
    mask = 1*mask

    BW1 = iuwt.IUWT_segmentation_for_testing(II0[-1], [2,3], 0.15, 200, 20, True, mask)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60))

    nmask = np.invert(mask)
    nmask = nmask.astype('uint8')
    border = cv2.dilate(nmask, se)
    border = (border > 254)
    border = 1*border
    BW1 = (BW1 <= 254)
    BW1 = 1*BW1
    BW1 = BW1 - border
    BW1 = (BW1 > 0)
    BW1 = 1 * BW1
    BW1 = BW1.astype('float')
    BW1 = cv2.resize(BW1, (512,512), cv2.INTER_CUBIC)
    cv2.imshow("Teste", BW1)

    #cv2.imwrite("teste.png", BW1)
    cv2.waitKey(0)
    return BW1



