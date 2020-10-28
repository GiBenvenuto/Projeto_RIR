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
    mkdir(config.tmp_adir)
    mkdir(config.ckpt_adir)

    dh = FundusDataHandler(is_train=True, im_size=config.im_size, db_size=config.db_size, db_type=0)

    aunet = aUnet(sess, config, "aUnet", is_train=True)

    dloss_file = []
    aloss_file = []

    for i in range(config.iteration):
        batch_x, batch_y = dh.sample_pair(config.batch_size, 1)
        # label 1 para treinar com a categoria S (que tem as imagens mais parecidas)

        aloss = aunet.fit(batch_x, batch_y)
        aloss_file.append(aloss)
        print("A-iter {:>6d} : {}".format(i + 1, aloss))

        if (i + 1) % 100 == 0:
            aunet.deploy(config.tmp_adir, batch_x, batch_y)
            aunet.save(config.ckpt_adir)



    # Salvando os dados para análise
    p.salve_training_data(aloss_file, config.aloss)
    p.plot_graph(config.aloss)


if __name__ == "__main__":
    train_tf()