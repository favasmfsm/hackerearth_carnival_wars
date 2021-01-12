#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('train.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#for missing datas
from sklearn.impute import SimpleImputer
imputer = SimpleImputer{}

#Encoding the variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



