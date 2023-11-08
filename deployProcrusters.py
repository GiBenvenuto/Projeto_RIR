import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from FundusData import FundusDataHandler
from DeformUnet import STN as dUnet
from config import get_config
from ops import mkdir
from Utils import plots as p
import cv2, csv
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import distance


def euclidean(grid, mesh):
  return np.sqrt(((grid[:,0] - mesh[:,0])**2) + ((grid[:,1] - mesh[:,1])**2))



def main():

  sess = tf.compat.v1.Session()
  config = get_config(is_train=False)
  mkdir(config.result_ddir)


  dh = FundusDataHandler(is_train=False, is_lbp=False, im_size=config.im_size, db_size=config.db_size, db_type=0)
  ndh = FundusDataHandler(is_train=False, is_lbp=False, im_size=config.im_size, db_size=config.db_size, db_type=1)
  dunet = dUnet(sess, config, "dUnet", is_train=False)
  dunet.restore(config.ckpt_ddir)
  ini = time.time()

  i = 2


  #Para cada categoria da FIRE
  for i in range(3):
    result_i_dir = config.result_ddir+"/{}".format(i)
    mkdir(result_i_dir)
    batch_x, batch_y = dh.sample_pair(config.batch_size, istrain=False, label=i)
    batch_tf, batch_tm = ndh.sample_pair(config.batch_size, istrain=False, label=i)

    #dunet.varlist(batch_x, batch_y)
    #dunet.deploy(result_i_dir, batch_x, batch_y, batch_tf, batch_tm)
    v = dunet.grid(batch_x, batch_y)
    V = v[-1, :, :, :]
    X, Y = np.meshgrid(np.linspace(0, config.lsize, config.lsize),np.linspace(0, config.lsize, config.lsize))
    VX = V[0,:,:]
    VY = V[1,:,:]
    VX = (VX + 1.0) * (config.lsize) / 2.0
    VY = (VY + 1.0) * (config.lsize) / 2.0
    VX = np.clip(VX, 0, config.lsize-1)
    VY = np.clip(VY, 0, config.lsize-1)

    X = np.clip(X, 0, config.lsize-1)
    Y = np.clip(Y, 0, config.lsize-1)

    mesh = np.stack((X, Y))
    grid = np.stack((VX, VY))
    mesh = np.reshape(mesh, (2, config.lsize * config.lsize))
    mesh = np.transpose(mesh)
    grid = np.reshape(grid, (2, config.lsize * config.lsize))
    grid = np.transpose(grid)
    ams = euclidean(grid, mesh)

    idx = np.argpartition(ams, -10)[-10:]
    indices = idx[np.argsort((-ams)[idx])]

    nmesh = [mesh[x, :] for x in indices]
    ngrid = [grid[x, :] for x in indices]



    R, sca = orthogonal_procrustes(ngrid, nmesh)

    print(i)

    print(R)
    print(sca)

    R_All, sca_All = orthogonal_procrustes(grid, mesh)

    print(R_All)
    print(sca_All)

    print('\n')

    if i == 0:
      tgt = cv2.imread("DataBases/Originais/P_mov/P01_2.jpg")
      scr = cv2.imread("DataBases/Originais/P_fix/P01_1.jpg")
      scr = scr[:,:,1]
      tgt = tgt[:,:,1]

      M = np.zeros((2,3))
      M[:,:-1] = R_All

      fig, axs = plt.subplots(1, 4)
      axs[0].imshow(tgt, 'gray')
      axs[1].imshow(scr, 'gray')

      img = cv2.warpAffine(scr, M, (2912, 2912))
      axs[2].imshow(img, 'gray')

      M = np.zeros((2, 3))
      M[:, :-1] = R
      img = cv2.warpAffine(scr, M, (2912, 2912))
      axs[3].imshow(img, 'gray')

      plt.show()

      mesh_file = open('mesh.csv', 'w')
      grid_file = open('grid.csv', 'w')

      m_file = csv.writer(mesh_file)
      g_file = csv.writer(grid_file)

      for i, j in zip(grid, mesh):
        g_file.writerow(i)
        m_file.writerow(j)





  fim = time.time()
  print(fim-ini)


if __name__ == "__main__":
   main()
