import os
from csvFile import CsvFile
import random
from clasificador import Clasificador
import numpy as np
import random
from csvFile import CsvFile
from matplotlib import pyplot as plt
from imagen import Imagen

def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1

def calculosFuncion(lista):
    blancos = np.where(lista > 99)
    negros = np.where(lista <= 1)
    if(blancos[0][0]==0 and negros[0][-1]==99):
        min = negros[0][0]
        max = blancos[0][-1]
    else:
        min = 0
        max = 0
    return (lista_sua,min,max)

okFile = ""
dfFile = ""

for file in os.listdir():
    if file.endswith(".csv"):
        #Recorremos todos porque queremos quedarnos con el último
        if file.startswith("data_Rebabas"):
            okFile = file

print("Procesamiento del fichero "+okFile)
#Leer datos ficheros
okFileData = CsvFile(file=okFile)
ok_data = okFileData.info

#fila = ok_data[0]
#lista = [float(i) for i in fila[3:103]]
"""
plt.plot(lista)
lista_sua = savitzky_golay(np.asarray(lista),5,1)
plt.plot(lista_sua)

blancos = np.where(lista_sua > 99)
negros = np.where(lista_sua <= 1)
print(lista_sua)
if(blancos[0][0]==0 and negros[0][-1]==99):
    min = negros[0][0]
    max = blancos[0][-1]
else:
    min = 0
    max = 0
"""
"""
lista_nueva, min, max = calculosFuncion(lista)
plt.plot(lista)
plt.plot(lista_nueva)
diferencia = abs(max-min)
print("[Min,max]: ["+str(min)+", "+str(max)+"] -> "+str(diferencia))
plt.text(max, 102, 'Max: '+str(max))
plt.text(min, 2, 'Min: '+str(min))
plt.plot([max, min], [100,0],marker='o', color='r', ls='')
"""
"""
for x in [5,6,7,12]:
    fila = ok_data[x]
    lista = [float(i) for i in fila[3:103]]
    lista_nueva, min, max = calculosFuncion(lista)
    plt.subplot("22"+str(x%4))
    plt.plot(lista)
    plt.plot(lista_nueva)
    diferencia = abs(max-min)
    medio = (max+min)/2
    medio_y = lista_nueva[int(medio)]
    print("[Min,max]: ["+str(min)+", "+str(max)+"] -> "+str(diferencia))
    plt.text(max, 102, 'Max: '+str(max))
    plt.text(min, 2, 'Min: '+str(min))
    plt.text(medio, medio_y, 'Medio: '+str(medio))
    plt.plot([max, min, medio], [100,0, medio_y],marker='o', color='r', ls='')

"""
plt.subplot(2,3,1)
#3array = [0,1,3,6,7]
array = [40,41,28,29,34]
for x in array:
    data = ok_data[x]
    lista = [float(i) for i in data[3:103]]
    texto = data[0]
    plt.plot(lista, label=texto[texto.rindex('/')+1:])

plt.legend()
plt.title("Umbral")

count = 2
for x in array:
    plt.subplot(2,3,count)
    count = count +1
    img = Imagen(ok_data[x][0], blancoNegro=True)
    img = img.ecualizar()
    #img = img.segmentacionOtsu()
    plt.imshow(img.img, cmap = 'gray')
    plt.title(ok_data[x][0])

plt.show()
