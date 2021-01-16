#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('train.csv')
X= dataset.iloc[:, 1:-1].values
y= dataset.iloc[:, -1].values

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/10, random_state = 0)

# #for missing datas
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 3:])
X[:,3:]=imputer.transform(X[:,3:])

#Encoding the variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X= np.array(ct.fit_transform(X))


