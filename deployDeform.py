import tensorflow as tf

from FundusData import FundusDataHandler
from DeformUnet import STN as dUnet
from config import get_config
from ops import mkdir

def main():
  sess = tf.compat.v1.Session()
  config = get_config(is_train=False)
  mkdir(config.result_ddir)

  dunet = dUnet(sess, config, "dUnet", is_train=False)
  dunet.restore(config.ckpt_ddir)
  dh = FundusDataHandler(is_train=False, im_size=config.im_size, db_size=config.db_size, db_type=0)

  #Para cada categoria da FIRE
  for i in range(3):
    result_i_dir = config.result_ddir+"/{}".format(i)
    mkdir(result_i_dir)
    batch_x, batch_y = dh.sample_pair(config.batch_size, i)
    dunet.deploy(result_i_dir, batch_x, batch_y)


if __name__ == "__main__":
   main()
