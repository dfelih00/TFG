import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

class Imagen:

    #Atributos
    img = []
    bN = False #Blanco y negro
    pieza = "Pieza"
    tipo = "Desconocido"

    #Constructor
    def __init__(self, ruta="", img_nueva=[], blancoNegro=False, pieza="Pieza", tipo="Desconocido"):
        self.pieza = pieza
        self.tipo = tipo
        self.ruta = ruta
        if(len(img_nueva)!=0):
            self.img = img_nueva
        else:
            if(ruta[len(ruta)-3:len(ruta)]=="tif"):
                im = Image.open(ruta)
                imarray = np.array(im)
                if blancoNegro:
                    self.img = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)
                else:
                    self.img = cv2.cvtColor(imarray, cv2.IMREAD_COLOR)
            else:
                if blancoNegro:
                    self.img = cv2.imread(ruta,cv2.IMREAD_GRAYSCALE)
                else:
                    self.img = cv2.imread(ruta,cv2.IMREAD_COLOR)

        self.bN = blancoNegro

    #Metodos
    def sobely(self, blancoNegro=True):
        if blancoNegro:
            return Imagen(img_nueva=cv2.Sobel(self.img,cv2.CV_8U,0,1,ksize=3),blancoNegro=True)
        else:
            return Imagen(img_nueva=cv2.Sobel(self.img,cv2.CV_64F,0,1,ksize=3))

    def secciones(self, ancho=1, alto=1):
        secciones = []
        for i in range(0,ancho):
            for j in range(0,alto):
                secciones.append(self.seccion(ancho,alto,i,j))
        return secciones

    def seccion(self, ancho=1,alto=1,fila=0,columna=0):
        dim_ancho = int(self.img.shape[0]/ancho)
        dim_alto = int(self.img.shape[1]/alto)
        pos_ancho = dim_ancho*fila
        pos_alto = dim_alto*columna
        return Imagen(img_nueva=self.img[pos_ancho:pos_ancho+dim_ancho,pos_alto:pos_alto+dim_alto],blancoNegro=self.bN)

    def histograma(self):
        hist = cv2.calcHist([self.img],[0],None,[256],[0,256])
        return hist

    def mostrarHistograma(self, rgb=False):
        if rgb and not self.bN:
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([self.img],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.show()
        else:
            plt.hist(self.img.ravel(),256,[0,256])
            plt.show()

    def ecualizar(self):
        if self.bN:
            equ = cv2.equalizeHist(self.img)
            return Imagen(self.ruta, img_nueva=equ,blancoNegro=True, pieza=self.pieza, tipo=self.tipo)
        else:
            return None

    def clahe(self):
        if self.bN:
            clahe = cv2.createCLAHE()#puede tener parametros
            cl1 = clahe.apply(self.img)
            return Imagen(self.ruta, img_nueva=cl1,blancoNegro=True, pieza=self.pieza, tipo=self.tipo)
        else:
            return None

    def quitarFondo(self, Gaussian=False):
        img = cv2.medianBlur(self.img,5)
        if(Gaussian):
            th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            return Imagen(self.ruta, img_nueva=th2,blancoNegro=True, pieza=self.pieza, tipo=self.tipo)
        else:
            th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            return Imagen(self.ruta, img_nueva=th3,blancoNegro=True, pieza=self.pieza, tipo=self.tipo)

    def auto_canny(self, sigma=0.33):
    	# compute the median of the single channel pixel intensities
    	v = np.median(self.img)

    	# apply automatic Canny edge detection using the computed median
    	lower = int(max(0, (1.0 - sigma) * v))
    	upper = int(min(255, (1.0 + sigma) * v))
    	edged = cv2.Canny(self.img, lower, upper)

    	# return the edged image
    	return Imagen(self.ruta, img_nueva=edged, blancoNegro=True, pieza=self.pieza, tipo=self.tipo)

    def segmentacionOtsu(self):
        ret, thresh = cv2.threshold(self.img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        return Imagen(self.ruta, img_nueva=thresh, blancoNegro=True, pieza=self.pieza, tipo=self.tipo)

    def get_num_pixels(self):
        width, height = self.img.shape
        return width*height

    def porcentage_pixeles_blancos(self):
        #negros = self.img.any(axis=-1).sum()
        negros = 0
        for fila in self.img:
            for pixel in fila:
                if(pixel==255):
                    negros = negros+1
        total = self.get_num_pixels()
        return (negros*100)/total

    def FAST_descriptors(self):
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(self.img, None)
        return kp

    def imagen_descriptores(self, kp):
        imagen_descriptores = Imagen(img_nueva=self.img, blancoNegro=self.bN)
        imagen_descriptores.img = cv2.drawKeypoints(imagen_descriptores.img, kp, imagen_descriptores.img,color=(255,0,0))
        return imagen_descriptores

    def get_textura(self):
        self.glcm = greycomatrix(self.img, [1], [0],  symmetric = True, normed = True)
        info = []
        info.append(greycoprops(self.glcm, 'contrast')[0][0])
        info.append(greycoprops(self.glcm, 'dissimilarity')[0][0])
        info.append(greycoprops(self.glcm, 'homogeneity')[0][0])
        info.append(greycoprops(self.glcm, 'energy')[0][0])
        info.append(greycoprops(self.glcm, 'correlation')[0][0])
        info.append(greycoprops(self.glcm, 'ASM')[0][0])
        return info

    def erosion(self,kernel_size=3, iteraciones=1):
        if self.bN:
            kernel = np.ones((kernel_size,kernel_size),np.uint8)
            transformacion = cv2.erode(self.img,kernel,iterations = iteraciones)
            return Imagen(self.ruta, img_nueva=transformacion, blancoNegro=True, pieza=self.pieza, tipo=self.tipo)
        else:
            return self

    def lbp(self, numPoints=24, radius=8, eps=1e-7):
        lbp = local_binary_pattern(self.img, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

    #def lbp(self, numPoints=24, radius=8):
        #img = self.img
        #img = local_binary_pattern(self.img, numPoints, radius, method="uniform")
        #img = cv2.Sobel(self.img,cv2.CV_8U,0,1,ksize=9)
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        #return Imagen(self.ruta, img_nueva=img, blancoNegro=True, pieza=self.pieza, tipo=self.tipo)

    def blur(self, ksize=24):
        img = cv2.blur(self.img, (10,10)) 
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        return Imagen(self.ruta, img_nueva=img, blancoNegro=True, pieza=self.pieza, tipo=self.tipo)