
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import pandas as pd
from interfaces.CSVViewer import CSVViewer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer as Imputer


file_path = r'../data/Data.csv'

viewer = CSVViewer(file_path)

dataset = pd.read_csv(file_path)
viewer = CSVViewer(file_path)
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,3].values 

#axis = 0 para las columnas y axis = 1 para las filas
imputer = Imputer(missing_values = np.nan , strategy = 'mean') 
X[ : , 1 : 3] = imputer.fit_transform(X[:, 1 : 3]) 

viewer.load_data(X)
viewer.show_data()