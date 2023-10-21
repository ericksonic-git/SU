# -*- coding: utf-8 -*-
"""
Created on Tue Oct 3 18:05:00 2023

@authors: Erick Davila & Manuel Sánchez
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as mtplpy
from joblib import dump, load
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import accuracy_score as AS, confusion_matrix as CM
#Máquina de vectores
from sklearn.svm import SVC



"""Importamos el conjunto de datos de una ubicación de Git"""
data = pd.read_csv('https://raw.githubusercontent.com/ericksonic-git/UNIR/main/DataSetUcayali_v2.csv')
data.shape
data.info()
datatrat = data

"""Calculamos los datos estadisticos (v.numéricas) y de frecuencias (v.categoricas)"""
for column in data:
    dt = ["Genero","depresion","Ansiedad","TrasPersonalidad","TrasBipolar",
          "ProblIdentSexual","IdeacionSuicida","Alcohol","TMCSustanciasPsicoa","TrasEsquizo","Esquizofrenia","Obesidad",
          "DolorCronico","Maltrato","AbusoSexual","CondSuicida"]
    if column in dt:
       print("\nCategorías y frecuencia de la columna:" + column)
       cats = pd.unique(data[str(column)])
       print(cats)
       frec = data[str(column)].value_counts()
       print(frec)
    else:
        print("\nDatos estadísticos de la columna: " + column)
        dtfstat = data[str(column)].describe()
        print(dtfstat)
        
"""Ahora procedemos a calcular y mostrar la matriz de correlación"""
mtrxcorr = data.corr()
print(mtrxcorr)
mtplpy.figure(figsize= (20,20))
sbn.heatmap(mtrxcorr, annot=True)
mtplpy.show()


"""Separamos variables"""
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(y.head(5))

"""Mostramos el nuevo arreglo con las variables depuradas"""
dt = pd.DataFrame.copy(datatrat)
dt.shape
dt.info(True,show_counts=True)

"""Normalizamos las variables de entrada"""

# scaler = SS()
# scaler.fit(X)
# scaled_x = scaler.fit_transform(X)
# X = pd.DataFrame(scaled_x, columns=X.columns)
# X.head(5)

"""OPCION 1"""
"""Aplicamos el clasificador basado en redes neuronales MPLClassifier"""
modelo = MLPC(hidden_layer_sizes = (16,40,12),max_iter=200,activation = 'relu',solver = 'adam')

scores = CVS(modelo,X,y,scoring="accuracy",cv=5,n_jobs=-1)
print("Resultados del modelo MLPC")
print(scores)
print("Promedio")
print(scores.mean())

"""Entrenamiento del modelo con muestras del 20%"""
x_train, x_test, y_train, y_test = TTS(X,y,test_size=0.2)
historial = modelo.fit(x_train,y_train)

"""Predición del modelo"""
y_pred= modelo.predict(x_test)
accur = AS(y_test,y_pred)

"""Matriz de confusión"""
print("Matriz de confusión")
c=CM(y_pred, y_test)
print(c)

plt.xlabel("No.de épocas")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.loss_curve_)

# """OPCION 2"""
# """Máquina de vectores de soporte"""
# modelo_2 = SVC(C=1, kernel='linear', gamma='auto', probability=True)
# scores_2 = CVS(modelo_2,X,y,scoring="accuracy",cv=5,n_jobs=-1)
# print("Resultados del modelo SVC")
# print(scores_2)
# print("Promedio SVC")
# print(scores_2.mean())

# """Entrenamiento del modelo con muestras del 20%"""
# x_train, x_test, y_train, y_test = TTS(X,y,test_size=0.1)
# historial = modelo_2.fit(x_train,y_train)
# """Predición del modelo"""
# y_pred = modelo_2.predict(x_test)
# accur = np.true_divide(np.sum(y_pred==y_test),y_pred.shape[0])*100
# cnf_matrix = CM(y_test,y_pred)
# print("Matriz de confusión")
# print(accur)
# print(cnf_matrix)


#Esto generaran el archivo del modelo y otro archivo del encoder
#Si tuvieramos alguna normalizacion o estandarizacion podriamos hacer similar
dump(modelo, 'modelo_suicidio.joblib')
# dump(encoder, 'label_encoder.joblib')
 
model_loaded = load('modelo_suicidio.joblib')
model_loaded.predict(x_test)
 
# encoder_loaded = load('label_encoder.joblib')
# type(encoder_loaded)








