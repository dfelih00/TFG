from matplotlib import pyplot as plt
from imagen import Imagen
import cv2
import numpy as np

prueba = Imagen("rebabas.tif", blancoNegro=True)
prueba = prueba.ecualizar()
otsu1 = prueba.segmentacionOtsu()
#secciones = otsu1.secciones(ancho=2)

imgErosion = otsu1.erosion(iteraciones=10)
#seccionesO = imgErosion.secciones(ancho=2)
imgErosion2 = imgErosion.erosion(iteraciones=10)
#secciones1 = imgErosion2.secciones(ancho=2)
secciones = imgErosion2.secciones(ancho=100)
"""
plt.subplot("131"),plt.imshow(otsu1.img, cmap = 'gray')
plt.title(str(otsu1.porcentage_pixeles_blancos())+' % blancos'), plt.xticks([]), plt.yticks([])
plt.subplot("132"),plt.imshow(imgErosion.img, cmap = 'gray')
plt.title(str(imgErosion.porcentage_pixeles_blancos())+' % blancos'), plt.xticks([]), plt.yticks([])
plt.subplot("133"),plt.imshow(imgErosion2.img, cmap = 'gray')
plt.title(str(imgErosion2.porcentage_pixeles_blancos())+' % blancos'), plt.xticks([]), plt.yticks([])
plt.show()
"""
blancos = []
for seccion in secciones:
    blancos.append(seccion.porcentage_pixeles_blancos())
plt.plot(blancos)
plt.title("Umbral")
plt.xlabel("Ancho de la imagen")
plt.ylabel("% Píxeles blancos")
plt.show()

i = 1
s = 1
for j in range(0,2):
    plt.subplot(int("23"+str(i))),plt.imshow(secciones[j].img, cmap = 'gray')
    plt.title('Otsu '+str(secciones[j].porcentage_pixeles_blancos())), plt.xticks([]), plt.yticks([])
    i = i+1
    plt.subplot(int("23"+str(i))),plt.imshow(seccionesO[j].img, cmap = 'gray')
    plt.title('Erosion '+str(seccionesO[j].porcentage_pixeles_blancos())), plt.xticks([]), plt.yticks([])
    i = i+1
    plt.subplot(int("23"+str(i))),plt.imshow(secciones1[j].img, cmap = 'gray')
    plt.title('Erosion '+str(secciones1[j].porcentage_pixeles_blancos())), plt.xticks([]), plt.yticks([])
    i = i+1
    s = s+1

plt.show()
