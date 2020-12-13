import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

############Leemos el CSV######################
df = pd.read_csv('poker-hand-tiny.csv')

############Limpiamos el CSV###################
for i in range(8,10):
    #Creamos lista con los indices a eliminar
    idmemor = df[df['Clase']== i].index
    #Eliminamos los indices del csv
    df = df.drop(idmemor)

#Define cada dataset
y_train = df.iloc[:,10]
x_train = df.iloc[:,0:10]

###################### CREACIÓN DATOS SINTÉTICOS #####################
smote = SMOTE()
X_train, Y_train = smote.fit_sample(x_train, y_train)
print(X_train, Y_train)

############# ÁRBOL DE DECISIÓN #################
arboldecision = DecisionTreeClassifier()
parametros = {'criterion': ['gini', 'entropy'],'splitter':['best', 'random'],'max_depth': [1,4,7,8,9,10]}

############# VALIDACIÓN CRUZADA - Grid Search ##############
clf = GridSearchCV(arboldecision, parametros, cv = 3)
clf.fit(X_train, Y_train)
clf.cv_results_
df = pd.DataFrame(clf.cv_results_)
print(df[['param_criterion', 'param_max_depth', 'param_splitter', 'mean_test_score', 'rank_test_score' ]])
df.to_csv(r'./nombrearchivo.csv')

######## NAIVE BAYES ##########
bayes = MultinomialNB()
val_cruzada = cross_validate(bayes, X_train, Y_train, cv = 3)

###### Resultados ###########
print('Resulatado Naive Bayes: ', val_cruzada['test_score']*100)


model = clf.best_estimator_
print('Resultado Arbol de Decisión:' , model.score(X_train,Y_train)*100)



