import csv
import datetime

class CsvFile:

    def __init__(self, info=None, etiqueta=None, file=None):
        if(file==None):
            fecha_actual = datetime.datetime.now()
            self.name_file = "data_"+etiqueta+"_"+str(fecha_actual.year)+"-"+str(fecha_actual.month)+"-"+str(fecha_actual.day)+"_"+str(fecha_actual.hour)+"-"+str(fecha_actual.minute)+".csv"
            with open(self.name_file, mode='w+', encoding="utf-8") as data_file:
                data_writer = csv.writer(data_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in info:
                    data_writer.writerow(row)
        else:
            self.name_file = file
            self.info = []
            with open(file, 'r',  encoding="utf-8") as csvfile:
                 spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
                 for row in spamreader:
                     if(len(row)>0):
                         self.info.append(row)
