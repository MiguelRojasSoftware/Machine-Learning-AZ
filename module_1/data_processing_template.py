"""
This is a template for data processing scripts.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interfaces.CSVViewer import CSVViewer
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import train_test_split

file_path = r'../data/Data.csv'


dataset = pd.read_csv(file_path)

viewer = CSVViewer(file_path)

X = dataset.iloc[ : , : -1 ].values 

Y = dataset.iloc[ : , 3 ].values

imputer = Imputer(missing_values = np.nan , strategy = 'mean') 
X[ : , 1 : 3] = imputer.fit_transform(X[:, 1 : 3]) 

X_train, X_test, Y_train, Y_test = train_test_split(X , Y , test_size=0.2, random_state = 0) 

print("train x :" , X_train)
print("test x :" , X_test)

print("train y :" , Y_train)
print("test y :" , Y_test)

viewer.load_data(dataset)
viewer.show_data()