import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from FundusData import FundusDataHandler
from DeformUnet import STN as dUnet
from config import get_config
from ops import mkdir
from Utils import plots as p
import Ground_Truth as gt
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

  #Para cada categoria da FIRE
  cat = 2
  ind = 1

  result_i_dir = config.result_ddir+"/{}".format(cat)
  mkdir(result_i_dir)
  batch_x, img_x, batch_y, img_y = dh.sample_pair_especific(cat, ind)
  #batch_tx, batch_ty = dh.sample_pair_choice(ind, cat)

  #dunet.varlist(batch_x, batch_y)
  dunet.deploy(result_i_dir, batch_x, batch_y, 1)

  v = dunet.grid(batch_x, batch_y)
  V = v[-1, :, :, :]
  X, Y = np.meshgrid(np.linspace(0, config.lsize, config.lsize),np.linspace(0, config.lsize, config.lsize))
  VX = V[0,:,:]
  VY = V[1,:,:]
  VX = (VX + 1.0) * (config.lsize) / 2.0
  VY = (VY + 1.0) * (config.lsize) / 2.0

  VX0 = np.floor(VX)
  VX1 = VX0 + 1
  VY0 = np.floor(VY)
  VY1 = VY0 + 1

  VX0 = np.clip(VX0, 0, config.lsize-1)
  VY0 = np.clip(VY0, 0, config.lsize-1)
  VX1 = np.clip(VX1, 0, config.lsize - 1)
  VY1 = np.clip(VY1, 0, config.lsize - 1)

  VX = (VX0 + VX1)/2
  VY = (VY0 + VY1)/2

  X = np.clip(X, 0, config.lsize - 1)
  Y = np.clip(Y, 0, config.lsize - 1)


  '''''''''

  pts_ref, pts_mov = gt.read_control_points(cat, ind)

  pts_grid = []
  for pt in pts_ref:
    pt_x = VX[pt[1]][pt[0]]
    pt_y = VY[pt[1]][pt[0]]
    pts_grid.append([pt_x, pt_y])

  pts_grid = np.array(pts_grid)




  mesh = np.stack((X, Y))
  grid = np.stack((VX, VY))
  mesh = np.reshape(mesh, (2, config.lsize*config.lsize))
  mesh = np.transpose(mesh)
  grid = np.reshape(grid, (2, config.lsize*config.lsize))
  grid = np.transpose(grid)
  ams = euclidean(grid, mesh)

  idx = np.argpartition(ams, -50)[-50:]
  indices = idx[np.argsort((-ams)[idx])]

  nmesh = [mesh[x, :] for x in indices]
  ngrid = [grid[x, :] for x in indices]



  R, sca = orthogonal_procrustes(ngrid, nmesh)

  print(R)
  print(sca)'''

  fig, ax = plt.subplots()
  img = plt.imread('black.png')
  ax.imshow(img, 'gray')


  #p.plot_grid(X, Y, ax=ax, color="blue", linewidths=0.2)
  p.plot_grid(VX, VY, ax=ax, color="lightblue", linewidths=0.2)
  #plt.scatter(x=pts_ref[:, 0], y=pts_ref[:, 1], c='g', s=20)
  #plt.scatter(x=pts_mov[:, 0], y=pts_mov[:, 1], c='m', s=20)
  #plt.scatter(x=pts_grid[:, 0], y=pts_grid[:, 1], c='b', s=20)
  #plt.quiver(X, Y, VX, VY)


  plt.show()
  #plt.savefig(config.result_ddir + "/" + str(i) + '/01_plot.png')

  '''

    fig = plt.figure()
    ax1 = plt.contourf(X, Y, VY)
    plt.colorbar(ax1)
    plt.show()
    '''


if __name__ == "__main__":
   main()
