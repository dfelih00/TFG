from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Perceptron

class Clasificador:

    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def clasificadorArbol(self):
        clf = DecisionTreeClassifier(max_depth=4)
        clf = clf.fit(self.X_train,self.Y_train)
        self.porcentaje_acierto = clf.score(self.X_test, self.Y_test)*100

    def clasificadorKVecinos(self, neighbors=2):
        clf = KNeighborsClassifier(n_neighbors=neighbors)
        clf = clf.fit(self.X_train,self.Y_train)
        self.porcentaje_acierto = clf.score(self.X_test, self.Y_test)*100

    def clasificadorRandomForest(self):
        clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        clf = clf.fit(self.X_train,self.Y_train)
        self.porcentaje_acierto = clf.score(self.X_test, self.Y_test)*100

    #Imortancia del vector de caracteristicas (X)
    def LDA(self):
        clf = LDA()
        clf.fit(self.X_train,self.Y_train)
        xbar = clf.coef_[0]
        xbar_sort = sorted(xbar, reverse=True)
        importance = []
        for value in xbar:
            for i in range(0,len(xbar_sort)):
                if value == xbar_sort[i]:
                    importance.append(i)
                    break
        return importance

    def clasificadorPerceptron(self):
        clf = Perceptron(tol=1e-3, random_state=0, validation_fraction=0.35)
        clf = clf.fit(self.X_train,self.Y_train)
        self.porcentaje_acierto = clf.score(self.X_test, self.Y_test)*100
