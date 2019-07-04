
class Umbral:

    #Atributos
    puntos = []

    #Constructor
    def __init__(self, puntos=[]):
        if puntos == []:
            self.funcion()
        else:
            self.puntos = puntos

    def invertir(self):
        return Umbral(list(reversed(self.puntos)))

    #Positivo(->) derecha, Negativo(<-) izquierda
    def desplazar(self, desplazamiento=1):
        lista = []
        if(desplazamiento > 0):
            for i in range(0,len(self.puntos)):
                if i<desplazamiento:
                    lista.append(self.puntos[0])
                else:
                    lista.append(self.puntos[i-desplazamiento])
        elif(desplazamiento < 0):
            posicion = len(self.puntos)+desplazamiento
            for i in range(0,len(self.puntos)):
                if i<posicion:
                    lista.append(self.puntos[i-desplazamiento])
                else:
                    lista.append(self.puntos[len(self.puntos)-1])
        else:
            lista = self.puntos
        return lista

    #Funcion estimada
    def funcion(self):
        self.puntos = []
        min = 45
        max = 55
        dif = max-min
        for i in range(0,100):
            if(i<min):
                self.puntos.append(0)
            elif(i<max):
                self.puntos.append(100/(dif+1)*(i%min+1))
            else:
                self.puntos.append(100)

    def diferencia(self, ptos):
        dif = 0
        for i in range(0,len(self.puntos)):
            dif = dif + abs(self.puntos[i]-ptos[i])
        return dif

    def diferenciaMinima(self, ptos2, desplazamiento=1, division=10):
        funcion = ptos2
        funcionMin = ptos2
        difMin = self.diferencia(funcion)
        for i in range(0,division):
            umbral = Umbral(funcion)
            funcion = umbral.desplazar(desplazamiento)
            dif = self.diferencia(funcion)
            if dif < difMin:
                funcionMin = funcion
                difMin = dif
        return funcionMin

    def funcionMinimaDiferencia(self, ptos2, division=10):
        ptos = ptos2
        funcionMinimaIzq = self.diferenciaMinima(ptos,-1,int(division/2))
        diferenciaMinIzq = self.diferencia(funcionMinimaIzq)
        funcionMinimaDer = self.diferenciaMinima(ptos,1,int(division/2))
        diferenciaMinDer = self.diferencia(funcionMinimaDer)
        iterations = 0
        while diferenciaMinIzq != diferenciaMinDer and iterations<50:
            umbral2 = Umbral(ptos)
            if diferenciaMinIzq<diferenciaMinDer:
                ptos = funcionMinimaIzq
            else:
                ptos = funcionMinimaDer
            funcionMinimaIzq = self.diferenciaMinima(ptos,-1,int(division/2))
            diferenciaMinIzq = self.diferencia(funcionMinimaIzq)
            funcionMinimaDer = self.diferenciaMinima(ptos,1,int(division/2))
            diferenciaMinDer = self.diferencia(funcionMinimaDer)
            iterations= iterations+1
        return ptos
