import numpy as np
from FundusData import FundusDataHandler
from config import get_config
import tensorflow as tf
from ops import mkdir
from Utils import plots as p
from DeformUnet import STN as dUnet
from AffineUnet import STN as aUnet




def train_tf():
    sess = tf.compat.v1.Session()
    config = get_config(is_train=True)  # Seta as configuracoes basicas (de treinamento)

    mkdir(config.tmp_ddir)
    mkdir(config.ckpt_ddir)
    dh = FundusDataHandler(is_train=True, im_size=config.im_size, db_size=config.db_size, db_type=0)
    dunet = dUnet(sess, config, "dUnet", is_train=True)


    dloss_file = []
    aloss_file = []

    for i in range(config.iteration):
        batch_x, batch_y = dh.sample_pair(config.batch_size, 1)
        # label 1 para treinar com a categoria S (que tem as imagens mais parecidas)

        dloss = dunet.fit(batch_x, batch_y)
        dloss_file.append(dloss)
        print("D-iter {:>6d} : {}".format(i + 1, dloss))

        if (i + 1) % 100 == 0:
            dunet.deploy(config.tmp_ddir, batch_x, batch_y)
            dunet.save(config.ckpt_ddir)


    # Salvando os dados para análise
    p.salve_training_data(dloss_file, config.dloss)
    p.plot_graph()



if __name__ == "__main__":
    train_tf()