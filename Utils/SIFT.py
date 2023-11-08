import numpy as np
import cv2
import csv

import matplotlib.pyplot as plt



# read images
imgy = cv2.imread('01_x.tif')
imgz = cv2.imread('01_z.tif')

img1 = cv2.cvtColor(imgy, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(imgz, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)



bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

k1 = []
k2 = []
for m in matches:
    k1.append(keypoints_1[m.queryIdx].pt)
    k2.append(keypoints_2[m.trainIdx].pt)




pts_ref = [np.array(k1[idx]) for idx in range(0, len(k1))]
pts_ref = np.array(pts_ref)

pts_warp = [np.array(k2[idx]) for idx in range(0, len(k2))]
pts_warp = np.array(pts_warp)

ref_file = open('ref1.csv', 'w')
warp_file = open('warp1.csv', 'w')

r_file = csv.writer(ref_file)
w_file = csv.writer(warp_file)

for i, j in zip(pts_ref, pts_warp):
    r_file.writerow(i)
    w_file.writerow(j)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()

