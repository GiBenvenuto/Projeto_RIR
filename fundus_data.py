import numpy as np
import glob, cv2

class FundusDataHandler(object):
    """
    Membros :
      is_train - Opcao de treinamento
      path - caminho local da FIRE
      size - tamanho da FIRE
      im_size - dimensao das imagens
  """

    def __init__(self, is_train, im_size, db_size):
        self.is_train = is_train
        self.files = self.get_files()
        self.size = db_size
        self.im_size = (im_size[0], im_size[1])
        self.fire_fix = []
        self.fire_mov = []
        #Carrega as imagens
        self.get_data_cat()


    def get_files(self):
        '''
        FIRE
        Categoria A = 14 imagens - Sobreposicao >75% e Mudancas anatomicas
        Categoria S = 71 imagens - Sobreposicao >75%
        Categoria P = 49 imagens - Sobreposicao <75%
        '''

        #Caminho para as imagens segmentadas
        A_files = glob.glob("FIRE/A_s/*.jpg")
        S_files = glob.glob("FIRE/S_s/*.jpg")
        P_files = glob.glob("FIRE/P_s/*.jpg")

        '''
        Caminho para as imagens reais
        A_files = glob.glob("FIRE/A/*.jpg")
        S_files = glob.glob("FIRE/S/*.jpg")
        P_files = glob.glob("FIRE/P/*.jpg")
        '''

        files = []
        files.append(A_files) # indice 0
        files.append(S_files) # indice 1
        files.append(P_files) # indice 2

        return files

    def get_fire(self, choices, ind):
        fire_fix = []
        fire_mov = []

        j = 0
        #Seleciona os pares de imagens nas posicoes escolhidas

        for i in choices:
            #Imagens fixas
            img = self.load_img(ind, i)
            fire_fix.append(img / 255)
            j = i + 1
            #Imagens em movimento
            img = self.load_img(ind, j)
            fire_mov.append(img / 255)

        return fire_fix, fire_mov

    def load_img(self, ind, i):
        #Coverte os canais e redimensiona a imagem
        img = cv2.imread(self.files[ind][i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, self.im_size, interpolation=cv2.INTER_CUBIC)
        img = img.reshape(img.shape + (1,))

        return img


    def get_data_cat(self):

        if (self.is_train):
            percent = 60
        else:
            percent = 30

        for i in range(0, 3):
            n = len(self.files[i])
            lim = round(n * percent / 100)
            #Escolhe as amostras aleatóriamente
            choices = np.random.choice(n-1, int(lim / 2))
            #Começando das posições pares para escolher o par certo
            choices = [j - 1 for j in choices if j % 2 != 0]
            fix, mov = self.get_fire(choices, i)
            self.fire_fix.append(fix)
            self.fire_mov.append(mov)


    def get_data_aug(self):
        Aug_fix_files = glob.glob("FIRE/aug_fix_S/*.png")
        Aug_mov_files = glob.glob("FIRE/aug_moving_S/*.png")

        if (self.is_train):
            percent = 60
        else:
            percent = 30

        n = len(Aug_fix_files)
        lim = round(n * percent / 100)
        #Escolhe as amostras aleatóriamente
        choices = np.random.choice(n-1, int(lim / 2))
        #Começando das posições pares para não errar o par

        # Seleciona os pares de imagens nas posicoes escolhidas

        for i in choices:
            # Imagens fixas
            img = cv2.imread(Aug_fix_files[i])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, self.im_size, interpolation=cv2.INTER_CUBIC)
            img = img.reshape(img.shape + (1,))
            self.fire_fix.append(img / 255)
            # Imagens em movimento
            img = cv2.imread(Aug_mov_files[i])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, self.im_size, interpolation=cv2.INTER_CUBIC)
            img = img.reshape(img.shape + (1,))
            self.fire_mov.append(img / 255)

    def sample_pair_cat(self, batch_size, label=None):
        label = np.random.randint(3) if label is None else label
        x = []
        y = []
        #Escolhe a amostra de imagens que vao ser testadas
        choice = np.random.choice(len(self.fire_fix[label]), batch_size)

        for i in choice:
            x.append(self.fire_fix[label][i])
            y.append(self.fire_mov[label][i])

        x = np.array(x)
        y = np.array(y)
        return x, y


    def sample_pair_aug(self, batch_size):
        x = []
        y = []
        #Escolhe a amostra de imagens que vao ser testadas
        choice = np.random.choice(len(self.fire_fix), batch_size)

        for i in choice:
            x.append(self.fire_fix[i])
            y.append(self.fire_mov[i])

        x = np.array(x)
        y = np.array(y)
        return x, y
