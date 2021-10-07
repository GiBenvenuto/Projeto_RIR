import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob
import cv2

def read_imgs_from_path (path):
    imgs = []
    for p in path:
        img = cv2.imread(p)
        imgs.append(img)

    return imgs

def create_hist (registred, height, width):
    h = [0, 0, 0]

    for image in registred:
        for i in range(0, height):
            for j in range(0, width):
                (b, g, r) = image[i, j]
                if (b, g, r) == (255, 255, 255):  # White
                    h[0] += 1;
                elif (b, g, r) == (255, 0, 255):  # Magenta
                    h[1] += 1;
                elif (b, g, r) == (0, 255, 0):  # Green
                    h[2] += 1;

    return h

def bars_hist():
    pathsA = glob.glob("../DataBases/Teste/GFEMR_inv/A_comp/*.png")
    pathsP = glob.glob("../DataBases/Teste/GFEMR_inv/P_comp/*.png")
    pathsS = glob.glob("../DataBases/Teste/GFEMR_inv/S_comp/*.png")

    compA = read_imgs_from_path(pathsA)
    compP = read_imgs_from_path(pathsP)
    compS = read_imgs_from_path(pathsS)

    hist = []

    hist.append(create_hist(compA, 512, 512))
    hist.append(create_hist(compS, 512, 512))
    hist.append(create_hist(compP, 512, 512))

    salve_hist_data(hist, "GFEMR_inv.csv")







def salve_hist_data(hist, name):
    csv.field_size_limit(393216)
    with open(name, 'w') as csvfile:
        for i in hist:
            row = str(i[0]) +","+ str(i[1]) + ","+ str(i[2])+ "\n"
            csvfile.write(row)

    csvfile.close()


def salve_training_data(loss, name):
    csv.field_size_limit(393216)
    with open(name, 'w') as csvfile:
        for i in loss:
            row = str(i) + "\n"
            csvfile.write(row)

    csvfile.close()


def load_training_data(name):
    file = open(name)
    loss = csv.reader(file)
    loss = list(loss)
    loss = np.hstack(loss)
    loss = loss.astype(np.float)

    return loss


def plot_graph(name):
    y = load_training_data(name)
    x = range(len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.ylim(0.0, 1.0)
    plt.title(name)
    plt.xlabel("Épocas")
    plt.ylabel("NCC")
    fig.savefig(name + ".png")
    plt.show()

def plot_img_gt (img, ref_pts, cor):
    implot = plt.imshow(img, 'gray')

    plt.scatter(x=ref_pts[:,0], y=ref_pts[:,1], c=cor, s=8)

    plt.show()


def plot_grid(x, y, ax=None, **kwargs):
  ax = ax or plt.gca()
  segs1 = np.stack((x, y), axis=2)
  segs2 = segs1.transpose(1, 0, 2)
  ax.add_collection(LineCollection(segs1, **kwargs))
  ax.add_collection(LineCollection(segs2, **kwargs))
  ax.autoscale()

if __name__ == '__main__':
    bars_hist()


