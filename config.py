'''
Classe para setar as configurações de diretórios,
database e rede
'''
class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.im_size = [256, 256]#Dimensão da imagem
  config.db_size = 134
  config.ckpt_adir = "ckpt_aUnet_256"#Checkpoint
  config.ckpt_ddir = "ckpt_dUnet_256"


  if is_train:
    config.batch_size = 4
    config.lr = 1e-4
    config.iteration = 1000
    config.tmp_adir = "tmp_aUnet_256"
    config.aloss = "train_aUnet_256"

    config.tmp_ddir = "tmp_dUnet_256"
    config.dloss = "train_dUnet_256"

  else:
    config.batch_size = 10
    config.result_adir = "teste_aUnet_256"
    config.result_ddir = "teste_dUnet_256"

  return config
