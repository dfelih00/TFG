import os
from csvFile import CsvFile
import random
from clasificador import Clasificador
import numpy as np
import random
from csvFile import CsvFile

def getXY(all_data):
    X_data = []
    Y_data = []
    Y_data_num = []
    for x in all_data:
        partial = [float(i) for i in x[3:9]]
        X_data.append(partial)
        Y_data.append(x[2])
        if(x[2]=="Ok"):
            Y_data_num.append(1)
        else:
            Y_data_num.append(2)
    return (X_data, Y_data_num)


okFile = ""
dfFile = ""

for file in os.listdir():
    if file.endswith(".csv"):
        #Recorremos todos porque queremos quedarnos con el último
        if file.startswith("data_Ok"):
            okFile = file
        if file.startswith("data_Df"):
            dfFile = file

print("Procesamiento de los ficheros "+okFile+" y "+dfFile)
#Leer datos ficheros
okFileData = CsvFile(file=okFile)
ok_data = okFileData.info
dfFileData = CsvFile(file=dfFile)
df_data = dfFileData.info
#Desordenar los Datos
random.shuffle(ok_data)
random.shuffle(df_data)
#Juntar datos
#all_data = ok_data + df_data
#Desordenar los Datos
#random.shuffle(all_data)
ok_data_x, ok_data_y = getXY(ok_data)
df_data_x, df_data_y = getXY(df_data)
#Separar los datos en training y test
print("Leido "+str(len(ok_data))+" registros Ok y "+str(len(df_data))+" registros Defectos")
porcentaje = 0.3
csv_values = []
csv_values.append(["Porcentaje training","DecisionTreeClassifier","KNeighborsClassifier","RandomForestClassifier","Perceptron","SVC"])
while(porcentaje<=0.95):
    print("Porcentaje training "+str(porcentaje*100))
    #Agrupar los datos para el clasificador
    #X_data ,Y_data_num = getXY(all_data)

    #data_train = all_data[:int(len(all_data)*porcentaje)]
    #data_test = all_data[int(len(all_data)*porcentaje):]
    X_train = ok_data_x[:int(len(ok_data_x)*porcentaje)]+df_data_x[:int(len(df_data_x)*porcentaje)]
    Y_train_num = ok_data_y[:int(len(ok_data_y)*porcentaje)]+df_data_y[:int(len(df_data_y)*porcentaje)]
    X_test = ok_data_x[int(len(ok_data_x)*porcentaje):]+df_data_x[int(len(df_data_x)*porcentaje):]
    Y_test_num = ok_data_y[int(len(ok_data_y)*porcentaje):]+df_data_y[int(len(df_data_y)*porcentaje):]

    aciertos = []
    aciertos.append(str(int(porcentaje*100))+" %")
    clasificador = Clasificador(X_train,Y_train_num,X_test,Y_test_num)
    clasificador.clasificadorArbol()
    aciertos.append(clasificador.porcentaje_acierto)
    clasificador.clasificadorKVecinos()
    aciertos.append(clasificador.porcentaje_acierto)
    clasificador.clasificadorRandomForest()
    aciertos.append(clasificador.porcentaje_acierto)
    clasificador.clasificadorPerceptron()
    p = clasificador.porcentaje_acierto
    aciertos.append(clasificador.porcentaje_acierto)
    clasificador.SVC()
    aciertos.append(clasificador.porcentaje_acierto)

    csv_values.append(aciertos)
    porcentaje = porcentaje + 0.02

okFile = CsvFile(csv_values,"Clasificador")
