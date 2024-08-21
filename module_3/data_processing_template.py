import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""

This is a template for data processing scripts.

"""

# plantilla para procesamiento de datos 05/07/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interfaces.CSVViewer import CSVViewer
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 


file_path = r'data/modulo1/Data.csv'
dataset = pd.read_csv(file_path)
viewer = CSVViewer(file_path)
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,3].values 


# Dividir el dataset en conjunto de dentrenamiento y conjunto de testing

X_train, X_test , Y_train , Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)

print(f"{X_train=}")
print(f"{Y_train=}")
print(" --- ")
print(f"{X_test=}")
print(f"{Y_test=}")

#print(Y)
# media = dataset['Age'].mean()
#print(media)

"""
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
viewer.load_data(dataset)
viewer.show_data()
"""