
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from interfaces.CSVViewer import CSVViewer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer



file_path = r'../data/Data.csv'

viewer = CSVViewer(file_path)

dataset = pd.read_csv(file_path)
viewer = CSVViewer(file_path)
X = dataset.iloc[:,:-1].values 
Y = dataset.iloc[:,3].values 


#codificar datos categoricos

LabelEncoder_X = LabelEncoder()
X[:,0] = LabelEncoder_X.fit_transform(X[:,0])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(columnTransformer.fit_transform(X))
#print(Y)
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)

viewer.load_data(Y)
viewer.show_data()