"""

This is a template for data processing scripts.

"""

# plantilla para procesamiento de datos 05/07/2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interfaces.CSVViewer import CSVViewer
from sklearn.impute import SimpleImputer as Imputer


file_path = r'C:/machine-learning/machine-learning-az/data/modulo1/Data.csv'

dataset = pd.read_csv(file_path)

viewer = CSVViewer(file_path)

X = dataset.iloc[:,:-1].values 

Y = dataset.iloc[:,3].values 

#axis = 0 para las columnas y axis = 1 para las filas
imputer = Imputer(missing_values = np.nan , strategy = 'mean') 
X[ : , 1 : 3] = imputer.fit_transform(X[:, 1 : 3]) 


# media = dataset['Age'].mean()
#print(media)

viewer.load_data(X)
viewer.show_data()