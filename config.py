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
  config.ckpt_dir = "ckpt_teste"#Checkpoint


  if is_train:
    config.batch_size = 4
    config.lr = 1e-4
    config.iteration = 1000
    config.tmp_dir = "tmp_teste"
    config.loss = "trains-loss"
  else:
    config.batch_size = 10
    config.result_dir = "teste_teste"

  return config
