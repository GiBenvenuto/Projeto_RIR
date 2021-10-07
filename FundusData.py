import numpy as np
import glob, cv2
from Utils import Convolution as conv

class FundusDataHandler(object):
    """
    Membros :
      is_train - Opcao de treinamento
      path - caminho local da FIRE
      size - tamanho da FIRE
      im_size - dimensao das imagens
  """

    def __init__(self, is_train, is_lbp, im_size, db_size, db_type):
        self.is_train = is_train
        self.is_lbp = is_lbp
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

        elif(type == 2):
            #Caminho para as imagens S aumentadas
            S_files_fix = glob.glob("DataBases/P_Aug/P_Fix/*.png")
            S_files_mov = glob.glob("DataBases/P_Aug/P_Mov/*.png")

            files_fix.append(S_files_fix)
            files_mov.append(S_files_mov)

        else:
            S_files_fix = glob.glob("DataBases/Aug_teste/Mov_seg/*.png")
            S_files_mov = glob.glob("DataBases/Aug_teste/Mov_seg/*.png")

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
            img = self.load_img(ind, i, True, self.is_lbp)
            fire_fix.append(img)

            #Imagens em movimento
            img = self.load_img(ind, i, False, self.is_lbp)
            fire_mov.append(img)

        return fire_fix, fire_mov

    def load_img(self, ind, i, is_fix, is_lbp):
        #Coverte os canais e redimensiona a imagem
        if (is_fix):
            img = cv2.imread(self.files_fix[ind][i])
        else:
            img = cv2.imread(self.files_mov[ind][i])

        img = img[:,:,1]
        img = cv2.resize(img, self.im_size, interpolation=cv2.INTER_CUBIC)
        #img = img - cv2.GaussianBlur(img, (0, 0), 3) + 127

        if is_lbp:
            fe = conv.FeatureExtraction()
            fe.setImg(img)
            img = fe.convolution()
        #cv2.imshow("teste", img)
        img = img.reshape(img.shape + (1,))
        img = img/255


        return img

    def get_data(self):
        #print(self.files_fix[0])
        n = len(self.files_fix)
        for i in range(0, n):
           size = len(self.files_fix[i])

           if (self.is_train):
               choices = [x for x in range(int(size/2))]
           else:
               choices = [x for x in range(int(size))]

           fix, mov = self.get_fire(choices, i)
           self.fire_fix.append(fix)
           self.fire_mov.append(mov)




    def sample_pair(self, batch_size, istrain, label=None):
        label = np.random.randint(3) if label is None else label
        x = []
        y = []
        if istrain:
            choice = np.random.choice(len(self.fire_fix[label]), batch_size)
        else:
            choice = range(0, batch_size)

        for i in choice:
            x.append(self.fire_fix[label][i])
            y.append(self.fire_mov[label][i])

        x = np.array(x)
        y = np.array(y)
        return y, x


    def sample_pair_test(self):
        x = []
        y = []

        imgx = cv2.imread("DataBases/P1.png")

        imgx = imgx[:, :, 1]
        imgx = cv2.resize(imgx, self.im_size, interpolation=cv2.INTER_CUBIC)
        imgx = imgx.reshape(imgx.shape + (1,))
        imgx = imgx / 255

        x.append(imgx)

        imgy = cv2.imread("DataBases/P1_1.png")

        imgy = imgy[:, :, 1]
        imgy = cv2.resize(imgy, self.im_size, interpolation=cv2.INTER_CUBIC)
        imgy = imgy.reshape(imgy.shape + (1,))
        imgy = imgy / 255

        y.append(imgy)

        x = np.array(x)
        y = np.array(y)
        return y, x




    def sample_pair_choice(self, i, label=None):
        label = np.random.randint(3) if label is None else label
        x = []
        y = []

        x.append(self.fire_fix[label][i])
        y.append(self.fire_mov[label][i])

        x = np.array(x)
        y = np.array(y)
        return x, y


    def sample_par_path(self, path_x, path_y):
        img_x = self.get_image_path(path_x)
        img_y = self.get_image_path(path_y)

        x = [img_x]
        y = [img_y]

        x = np.array(x)
        y = np.array(y)

        return x, y





    def sample_pair_especific(self, cat, ind):
        x = []
        y = []


        x.append(self.fire_fix[cat][ind])
        y.append(self.fire_mov[cat][ind])

        x = np.array(x)
        y = np.array(y)
        img_x = self.get_fix_image(cat, ind)
        img_y = self.get_mov_image(cat, ind)

        return x, img_x, y, img_y

    def get_mov_image(self, ind, i):
        img = self.load_img(ind, i, False, False)
        return img

    def get_fix_image(self, ind, i):
        img = self.load_img(ind, i, True, False)
        return img

    def get_size_cats(self, i):
        return len(self.files_fix[i])

    def get_image_path(self, path):
        img = cv2.imread(path)
        img = img[:, :, 1]
        img = cv2.resize(img, self.im_size, interpolation=cv2.INTER_CUBIC)
        img = img.reshape(img.shape + (1,))
        img = img / 255

        return img




