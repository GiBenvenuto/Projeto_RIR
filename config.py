'''
Classe para setar as configurações de diretórios,
database e rede
'''
class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.lsize = 512
  config.im_size = [config.lsize, config.lsize]#Dimensão da imagem
  config.db_size = 71 #tamanho da base
  config.ckpt_ddir = "CKPT/ckpt_dUnet_512_1603"  #iretório checkpoint rede deform


  if is_train:
    config.batch_size = 1
    config.lr = 1e-4
    config.iteration = 5000

    config.tmp_ddir = "TMP/tmp_dUnet_256_100seg" #Diretório deform temporário
    config.dloss = "Files/train_dUnet_256_100seg" #Diretório treinamento deform

  else:
    config.batch_size = 1
    config.result_ddir = "TESTES/Teste_512_1603_malha"

  return config
