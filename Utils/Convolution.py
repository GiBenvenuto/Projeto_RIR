import numpy as np
import math

class FeatureExtraction():
	height = width = 0


	def __init__(self):
		return


	def setImg(self, img):
		self.img = img
		self.height = self.img.shape[0]
		self.width = self.img.shape[1]


	def signal(self, i, j, center):
		val = self.check_border(i, j)
		if val < center:
			val = 0
		else:
			val = 1

		return val


	def check_border(self, i, j):
		if i < 0 or i >= self.height or j < 0 or j >= self.width:
			return 0
		return self.img[i, j]


	def lbp(self, i, j):
		cod = 0
		cont = 0
		cod = self.signal(i - 1, j - 1, self.img[i,j])*math.pow(2, 0)
		cod += self.signal(i - 1, j, self.img[i,j])*math.pow(2, 1)
		cod += self.signal(i - 1, j + 1, self.img[i,j])*math.pow(2, 2)
		cod += self.signal(i, j + 1, self.img[i,j])*math.pow(2, 3)
		cod += self.signal(i + 1, j + 1, self.img[i,j])*math.pow(2, 4)
		cod += self.signal(i + 1, j, self.img[i,j])*math.pow(2, 5)
		cod += self.signal(i + 1, j - 1, self.img[i,j])*math.pow(2, 6)
		cod += self.signal(i, j - 1, self.img[i,j])*math.pow(2, 7)
        
		return int(cod)


	def convolution(self):
		#print(self.height)
		#print(self.width)
		mAux = np.zeros((self.height, self.width))
		mAux = mAux.astype(np.uint8)

		for i in range(self.height):
			for j in range(self.width):
				mAux[i, j] = self.llbp(i,j, 9)
				#mAux[i, j] = self.riu_lbp(i, j)
				#mAux[i, j] = self.lbp(i,j)
				#mAux[i,j] = self.cs_lbp(i,j)
				#mAux[i,j] = self.lbp_circular(i,j,1,8)
				#mAux[i,j] = self.md_riulbp(i,j)
                
		return mAux
		

	def interpolation(self, i, iC, jC, r, P):

		x = iC + r*math.cos((iC*math.pi*i)/P)
		y = jC + r*math.sin((jC*math.pi*i)/P)

		if x != math.floor(x) or y != math.floor(y):
			x_inter = math.fabs(x - math.floor(x))
			y_inter = math.fabs(y - math.floor(y))

			p1 = (1 - x_inter) * (1 - y_inter)
			p2 = x_inter * (1 - y_inter)
			p3 = (1 - x_inter) * y_inter
			p4 = x_inter * y_inter

			valor = round(p1*self.check_border(int(math.floor(x-1)), int(math.floor(y))) + 
				p2*self.check_border(int(math.ceil(x-1)), int(math.floor(y))) + 
				p3*self.check_border(int(math.floor(x-1)), int(math.ceil(y))) + 
				p4*self.check_border(int(math.ceil(x-1)), int(math.ceil(y))))

		else:
			valor = self.check_border(int(x), int(y))

		return valor


	def signal2(self, val, center):
		if val > center:
			val = 1
		else:
			val = 0

		return val


	def lbp_circular(self, iC, jC, r, P):
		centro = self.img[iC, jC]
		cod = 0
		for i in range(P):
			valor = self.interpolation(i, iC, jC, r, P)
			cod += self.signal2(valor, centro)*math.pow(2, i)
			
		return int(cod)


	def cs_lbp(self, iC, jC):
		cod = 0

		g0 = self.check_border(iC, jC + 1)
		g4 = self.check_border(iC, jC - 1)
		cod += self.signal2(g0, g4)*math.pow(2, 0)

		g1 = self.check_border(iC + 1, jC + 1)
		g5 = self.check_border(iC - 1, jC - 1)
		cod += self.signal2(g1, g5)*math.pow(2, 1)

		g2 = self.check_border(iC + 1, jC)
		g6 = self.check_border(iC - 1, jC)
		cod += self.signal2(g2, g6)*math.pow(2, 2)

		g3 = self.check_border(iC + 1, jC - 1)
		g7 = self.check_border(iC - 1, jC + 1)
		cod += self.signal2(g3, g7)*math.pow(2, 3)

		return int(cod)

	def riu_lbp(self, i, j):
		cod = []
		cod.append(self.signal(i - 1, j - 1, self.img[i,j]))
		cod.append(self.signal(i - 1, j, self.img[i,j]))
		cod.append(self.signal(i - 1, j + 1, self.img[i,j]))
		cod.append(self.signal(i, j + 1, self.img[i,j]))
		cod.append(self.signal(i + 1, j + 1, self.img[i,j]))
		cod.append(self.signal(i + 1, j, self.img[i,j]))
		cod.append(self.signal(i + 1, j - 1, self.img[i,j]))
		cod.append(self.signal(i, j - 1, self.img[i,j]))

		(cont, aux) = self.uniform_lbp(cod)
		if cont > 2:
			return 9
		else:
			return aux


	def uniform_lbp(self, cod):
		cont = 0
		aux = 0

		for x in range(0,len(cod)):
			if x+1 < len(cod) and cod[x] != cod[x+1]:
				cont = cont + 1
			if cod[x] == 1:
				aux = aux + 1

		return (cont, aux)



	def signal3(self, val, center):
		if val < center:
			val = 0
		else:
			val = 1

		return val
		



	def llbph(self, iC, jC, N):
		aux_n = int(N/2)
		cod = 0
		center = self.img[iC, jC]
		for n in range(1, aux_n + 1):
			val = self.check_border(iC, jC + n)
			cod += self.signal3(val, center)*math.pow(2, n - 1)

			val = self.check_border(iC, jC - n)
			cod += self.signal3(val, center)*math.pow(2, n - 1)

		return cod




	def llbpv(self, iC, jC, N):
		aux_n = int(N/2)
		cod = 0
		center = self.img[iC, jC]
		for n in range(1, aux_n + 1):
			val = self.check_border(iC + n, jC)
			cod += self.signal3(val, center)*math.pow(2, n - 1)

			val = self.check_border(iC - n, jC)
			cod += self.signal3(val, center)*math.pow(2, n - 1)

		return cod


	def llbp(self, iC, jC, N):
		llbph = self.llbph(iC, jC, N)
		llbpv = self.llbpv(iC, jC, N)

		return math.floor(math.sqrt(math.pow(llbph, 2) + math.pow(llbpv, 2)))

	'''def llbpv(self, iC, jC, N):
		aux_n = int(N/2)
		cod = 0
		center = self.img[iC, jC]
		for n in range(0, aux_n):
			val = self.check_border(iC + (aux_n - n), jC)
			cod += self.signal(val, center)*math.pow(2, aux_n - n -1)

		for n in range(aux_n + 1, N):
			val = self.check_border(iC + (n - aux_n), jC)
			cod += self.signal(val, center)*math.pow(2, n -aux_n -1)

		return cod'''

	def md_riulbp(self, iC, jC):
		center = self.img[iC, jC]
		cod1 = []
		cod1.append(self.signal2(self.check_border(iC - 1, jC), center))
		cod1.append(self.signal2(self.check_border(iC, jC + 1), center))
		cod1.append(self.signal2(self.check_border(iC + 1, jC), center))
		cod1.append(self.signal2(self.check_border(iC, jC - 1), center))

		(cont, aux1) = self.uniform_lbp(cod1)
		if cont > 2:
			aux1 = 5

		cod2 = []
		cod2.append(self.signal2(self.check_border(iC - 1, jC - 1), center))
		cod2.append(self.signal2(self.check_border(iC - 1, jC + 1), center))
		cod2.append(self.signal2(self.check_border(iC + 1, jC + 1), center))
		cod2.append(self.signal2(self.check_border(iC + 1, jC - 1), center))

		(cont, aux2) = self.uniform_lbp(cod2)
		if cont > 2:
			aux2 = 5

		return aux1 + aux2
		





