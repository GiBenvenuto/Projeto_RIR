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

    def __init__(self, is_train, im_size, db_size, db_type):
        self.is_train = is_train
        self.size = db_size
        self.im_size = (im_size[0], im_size[1])

        self.files_fix, self.files_mov = self.get_fire_files(db_type)
        self.fire_fix = []
        self.fire_mov = []
        #Carrega as imagens
        self.get_data()


    def get_fire_files(self, type=0):
        '''
         FIRE
         Categoria A = 14 imagens - Sobreposicao >75% e Mudancas anatomicas
         Categoria S = 71 imagens - Sobreposicao >75%
         Categoria P = 49 imagens - Sobreposicao <75%
         '''
        files_fix = []
        files_mov = []

        if (type == 0):
            # Caminho para as imagens segmentadas
            A_files_fix = glob.glob("DataBases/Segmentadas/A_s_fix/*.jpg")
            A_files_mov = glob.glob("DataBases/Segmentadas/A_s_mov/*.jpg")

            S_files_fix = glob.glob("DataBases/Segmentadas/S_s_fix/*.jpg")
            S_files_mov = glob.glob("DataBases/Segmentadas/S_s_mov/*.jpg")

            P_files_fix = glob.glob("DataBases/Segmentadas/P_s_fix/*.jpg")
            P_files_mov = glob.glob("DataBases/Segmentadas/P_s_mov/*.jpg")

            files_fix.append(A_files_fix)
            files_fix.append(S_files_fix)
            files_fix.append(P_files_fix)
            files_mov.append(A_files_mov)
            files_mov.append(S_files_mov)
            files_mov.append(P_files_mov)

        elif(type == 1):
            #Caminho para as imagens reais
            A_files_fix = glob.glob("DataBases/Originais/A_fix/*.jpg")
            A_files_mov = glob.glob("DataBases/Originais/A_mov/*.jpg")

            S_files_fix = glob.glob("DataBases/Originais/S_fix/*.jpg")
            S_files_mov = glob.glob("DataBases/Originais/S_mov/*.jpg")

            P_files_fix = glob.glob("DataBases/Originais/P_fix/*.jpg")
            P_files_mov = glob.glob("DataBases/Originais/P_mov/*.jpg")

            files_fix.append(A_files_fix)
            files_fix.append(S_files_fix)
            files_fix.append(P_files_fix)
            files_mov.append(A_files_mov)
            files_mov.append(S_files_mov)
            files_mov.append(P_files_mov)

        else:
            #Caminho para as imagens S aumentadas
            S_files_fix = glob.glob("DataBases/Aumentadas/aug_fix_S/*.png")
            S_files_mov = glob.glob("DataBases/Aumentadas/aug_moving_S/*.png")

            files_fix.append(S_files_fix)
            files_mov.append(S_files_mov)



        return files_fix, files_mov

    def get_fire(self, choices, ind):
        fire_fix = []
        fire_mov = []

        j = 0
        #Seleciona os pares de imagens nas posicoes escolhidas

        for i in choices:
            #Imagens fixas
            img = self.load_img(ind, i, True)
            fire_fix.append(img / 255)

            #Imagens em movimento
            img = self.load_img(ind, i, False)
            fire_mov.append(img / 255)

        return fire_fix, fire_mov

    def load_img(self, ind, i, is_fix):
        #Coverte os canais e redimensiona a imagem
        if (is_fix):
            img = cv2.imread(self.files_fix[ind][i])
        else:
            img = cv2.imread(self.files_mov[ind][i])

        img = img[:,:,1]
        img = cv2.resize(img, self.im_size, interpolation=cv2.INTER_CUBIC)
        img = img.reshape(img.shape + (1,))

        return img

    def get_data(self):

        if (self.is_train):
            percent = 60
        else:
            percent = 30

        n = len(self.files_fix)
        for i in range (0, n):
            size = len(self.files_fix[i])
            lim = round(size * percent / 100)

            #Escolhe as amostras aleatóriamente
            choices = np.random.choice(size-1, int(lim))
            fix, mov = self.get_fire(choices, i)
            self.fire_fix.append(fix)
            self.fire_mov.append(mov)




    def sample_pair(self, batch_size, label=None):
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


