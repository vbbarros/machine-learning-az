# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:30:39 2018

@author: vbbarros
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])"""

#Encoding categorical data
#Basicamente precisamos transformar as colunas de string em numeros
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()"""
#X precisa usar o OneHotEncoder, o Y não
"""X[:, 0] = labelEncoder.fit_transform(X[:, 0])"""

#Considerando que machine learning iria interpretar os valores encode como grandeza
#Precisamos transformar cada País em uma coluna, 
#já que não faria sentido Espanha ter maior valor que França por exemplo
"""oneHotEncoder = OneHotEncoder(categorical_features= [0])
X = oneHotEncoder.fit_transform(X).toarray()
Y = labelEncoder.fit_transform(Y)"""

#Splitting the dataset into the Trining set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
#Basicamente, algumas contas do machine learning usam a distancia entre dois pontos
#Se um número é muito maior que o outro, o maior vai dominar essa distância
#Por isso, é preciso colocar os números em uma mesma escala
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
