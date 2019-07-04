from matplotlib import pyplot as plt
from imagen import Imagen
import cv2
import numpy as np
import os
import csv
import datetime
from csvFile import CsvFile

def procesarImagen(imagen, lista):
    print("Procesando imagen "+imagen.ruta)
    #Preprocesado
    imagen_preprocesada = imagen.clahe()
    #Area interés
    imagen_interes = imagen_preprocesada.lbp()
    #Descriptores
    info_partial = [imagen_interes.ruta, imagen_interes.pieza, imagen_interes.tipo]
    info_partial2 = imagen_interes.get_textura()
    lista.append(info_partial+info_partial2)

#Variables
info_ok = []
info_defectos = []

#Leer imagenes
ruta = 'C:/Users/DavidAngel/Desktop/tfg/fotosPiezas';
for pieza in os.listdir(ruta):
    rutaPieza = ruta+'/'+pieza
    if os.path.isdir(rutaPieza):
        for tipo in os.listdir(rutaPieza):
            if tipo=="Ok" or tipo=="Defectos":
                rutaTipo = rutaPieza +'/'+tipo
                if os.path.isdir(rutaTipo):
                    for imagen in os.listdir(rutaTipo):
                        rutaImagen = rutaTipo + '/'+imagen
                        if imagen.endswith('.tif'):
                            if tipo=="Ok":
                                img = Imagen(rutaImagen, blancoNegro=True, pieza=pieza, tipo = tipo)
                                procesarImagen(img, info_ok)
                            else:
                                img = Imagen(rutaImagen, blancoNegro=True, pieza=pieza, tipo = tipo)
                                procesarImagen(img, info_defectos)
print("Se han encontrado "+str(len(info_ok))+" imágenes del tipo Ok")
print("Se han encontrado "+str(len(info_defectos))+" imágenes del tipo Defectos")

#Guardar los datos
print("Guardando datos Ok")
okFile = CsvFile(info_ok,"Ok")
print("Los datos Ok han sido guardados")

print("Guardando datos Defectos")
okFile = CsvFile(info_defectos,"Df")
print("Los datos Defectos han sido guardados")
