from umbral import Umbral
from matplotlib import pyplot as plt
import os
from csvFile import CsvFile

lista = []

u = Umbral()
uR = u.invertir()

okFile = ""
dfFile = ""

##Cargar informacion ficheros
for file in os.listdir():
    if file.endswith(".csv"):
        #Recorremos todos porque queremos quedarnos con el último
        if file.startswith("data_Rebabas"):
            okFile = file

print("Procesamiento del fichero "+okFile)
#Leer datos ficheros
okFileData = CsvFile(file=okFile)
ok_data = okFileData.info

array = [0]
#array = [0,1,3,6,7]
#array = [40,41,28,29,34]
for x in array:
    data = ok_data[x]
    lista = [float(i) for i in data[3:103]]
    texto = data[0]
    plt.plot(lista, label="FuncionImagen")
    plt.plot(uR.funcionMinimaDiferencia(lista), label="FuncionMinimaDif")


plt.plot(uR.puntos, label="FuncionUmbral")
plt.legend()
plt.show()
