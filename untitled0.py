import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter


#Leemos el CSV
df = pd.read_csv('poker-hand-tiny.csv')
print("Contenido de data set: ", df.shape)
print("Cantidad por clase:\n", df.Clase.value_counts())

#Limpiamos el CSV
for i in range(8,10):
    #Creamos lista con los indices a eliminar
    idmemor = df[df['Clase']== i].index
    #Eliminamos los indices del csv
    df = df.drop(idmemor)
    
#Define cada dataset
y_train = df.iloc[:,10]
x_train = df.iloc[:,0:10]

# CREACIÓN DATOS SINTÉTICOS
smote = SMOTE()
X_train, Y_train = smote.fit_sample(x_train, y_train)
print("Antes de SMOTE:", Counter(y_train),"\nDespués de SMOTE: ", Counter(Y_train))

#ÁRBOL DE DECISIÓN 
arboldecision = DecisionTreeClassifier()
parametros = {'max_depth': [10, 20, 40, 60]}

# VALIDACIÓN CRUZADA - Grid Search
clf = GridSearchCV(arboldecision, parametros, cv = 3)
clf.fit(X_train, Y_train)
clf.cv_results_
resultados = pd.DataFrame(clf.cv_results_)
means = clf.cv_results_['mean_test_score']
for mean, parametros in zip(means,  clf.cv_results_['params']):
    print("%0.3f for %r" %(mean, parametros))

print()