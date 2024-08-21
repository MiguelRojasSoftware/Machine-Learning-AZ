"""
This is a template for data processing scripts.
"""
# Interface para visualizar datos en formato CSV
# Añadir el directorio raíz al PYTHONPATH para permitir importaciones de módulos desde la raíz del proyecto
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# plantilla para procesamiento de datos: 05/07/2024
# ultima actualizacion: 08/08/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interfaces.CSVViewer import CSVViewer
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import train_test_split

# ruta de archivo de datos
file_path = r'C:/machine-learning/machine-learning-az/data/Data.csv'

#lectura de datos convirtiendose a dataset y importacion a interfaz 
dataset = pd.read_csv(file_path)
viewer = CSVViewer(file_path)

# ------------------------ division de datos  -----------------------------
"""
X = selecciona todas las filas (,) selecciona todas las columnas excepto la ultima
Y = selecciona todas las filas (,) selecciona solamente la columna 3
"""
X = dataset.iloc[ : , : -1 ].values 
Y = dataset.iloc[ : , 3 ].values



# ------- --completar datos con fit_transform atravez de la media----------
""" 
Imputacion de valores faltantes (nan) atravez de su media y utilizacion
"""
imputer = Imputer(missing_values = np.nan , strategy = 'mean') 
X[ : , 1 : 3] = imputer.fit_transform(X[:, 1 : 3]) 


# -------- completar datos a través de entrenamiento de datos ------------

"""
Separación de datos en conjuntos de entrenamiento y pruebas.
X_train, Y_train: Conjunto de datos utilizado para entrenar el modelo de inteligencia artificial.
X_test, Y_test: Conjunto de datos utilizado para evaluar el rendimiento del modelo.
El tamaño del conjunto de pruebas es el 20% del total, definido por test_size= 0.2
Por lo tanto el tamaño de datos usados para el entrenamiento sera el restante es decir 80%.
La semilla aleatoria (random_state=0) asegura la reproducibilidad del resultado.
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


viewer.load_data(dataset)
viewer.show_data()