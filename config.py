'''
Classe para setar as configurações de diretórios,
database e rede
'''
class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.im_size = [512, 512]#Dimensão da imagem
  config.db_size = 134
  config.ckpt_adir = "ckpt_Dirnet_unet_256"#Checkpoint
  config.ckpt_ddir = "ckpt_dUnet_512B"


  if is_train:
    config.batch_size = 8
    config.lr = 1e-4
    config.iteration = 10000
    config.tmp_adir = "tmp_aUnet_512"
    config.aloss = "train_aUnet_728"

    config.tmp_ddir = "tmp_dUnet_512B"
    config.dloss = "train_dUnet_512B"

  else:
    config.batch_size = 10
    config.result_adir = "teste_aUnet_512"
    config.result_ddir = "teste-dUnet_512B"

  return config
