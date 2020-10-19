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
  config.ckpt_dir = "ckpt_Affine_unet_256"#Checkpoint


  if is_train:
    config.batch_size = 4
    config.lr = 1e-4
    config.iteration = 1000
    config.tmp_dir = "tmp_Affine_unet_256"
    config.loss = "train-Affine_unet_256"
  else:
    config.batch_size = 10
    config.result_dir = "teste_Affine_Unet_256"

  return config
