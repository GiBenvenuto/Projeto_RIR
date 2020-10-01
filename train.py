import numpy as np
from fundus_data import FundusDataHandler
from config import get_config
import tensorflow as tf
from ops import mkdir
from Utils import plots as p
from DirnetTensorflow import DIRNet
#from DirnetAffineTensorflow import DIRNet


def train_tf():
    sess = tf.compat.v1.Session()
    config = get_config(is_train=True)  # Seta as configuracoes basicas (de treinamento)
    mkdir(config.tmp_dir)
    mkdir(config.ckpt_dir)
    dh = FundusDataHandler(is_train=True, im_size=config.im_size, db_size=config.db_size, db_type=0)
    reg = DIRNet(sess, config, "DIRNet", is_train=True)

    loss_file = []

    for i in range(config.iteration):
        batch_x, batch_y = dh.sample_pair(config.batch_size, 2)
        # label 1 para treinar com a categoria S (que tem as imagens mais parecidas)

        loss = reg.fit(batch_x, batch_y)
        loss_file.append(loss)
        print("iter {:>6d} : {}".format(i + 1, loss))
        if (i + 1) % 100 == 0:
            reg.deploy(config.tmp_dir, batch_x, batch_y)
            reg.save(config.ckpt_dir)

    # Salvando os dados para análise
    p.salve_training_data(loss_file, config.loss)

def plot():
    p.plot_graph()



if __name__ == "__main__":
    plot()