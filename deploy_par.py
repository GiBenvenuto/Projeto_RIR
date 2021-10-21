import tensorflow as tf
from DeformUnet import STN as dUnet
from config import get_config
from ops import mkdir
import cv2, numpy as np
import glob

def main():
  sess = tf.compat.v1.Session()
  config = get_config(is_train=False)


  dunet = dUnet(sess, config, "dUnet", is_train=False)
  dunet.restore(config.ckpt_ddir)

  x = glob.glob("DataBases/Segmentadas/P_s_fix/*.jpg")
  y = glob.glob("DataBases/Segmentadas/P_s_mov/*.jpg")

  for i in range(len(x)):
    batch_x = []
    batch_y = []

    img = cv2.imread(x[i])
    img = img[:, :, 1]
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(img.shape + (1,))
    img = img / 255
    batch_x.append(img)
    batch_x = np.array(batch_x)

    img = cv2.imread(y[i])
    img = img[:, :, 1]
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(img.shape + (1,))
    img = img / 255
    batch_y.append(img)
    batch_y = np.array(batch_y)

    result_dir = "Teste_Todas/P"
    mkdir(result_dir)


    dunet.deploy(result_dir, batch_x, batch_y, i)




if __name__ == "__main__":
   main()
